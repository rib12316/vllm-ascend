#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Ascend GGUF (llama.cpp k-quant) Linear method for Ascend NPU.

GGUF uses a weight-loading contract (``is_gguf_weight`` /
``is_gguf_weight_type``) that is distinct from the packed-param pattern of
AWQ/GPTQ, so it does **not** use the ``AscendLinearScheme`` registry. Instead it
has a dedicated ``AscendGGUFLinearMethod`` that:

1. **create_weights** — reuses upstream vLLM's GGUF parameter contract verbatim
   (so ``GGUFModelLoader`` fills ``qweight`` / ``qweight_type`` unchanged).
2. **process_weights_after_loading** — dequantizes the quantized weight bytes to
   dense fp16/bf16 *once at load time* via pure-torch kernels
   (:mod:`gguf_dequant`), replacing vLLM's CUDA ``ggml_dequantize``. The
   universal MVP path: every supported block type → dense, then runs as a
   standard linear. (Simple symmetric types can later be repacked onto
   ``npu_weight_quant_batchmatmul`` for speed — see G-8.)
3. **apply** — dense ``F.linear(x, weight, bias)``.
"""

import torch
import torch.nn.functional as F
import torch_npu
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.gguf import GGUFUninitializedParameter
from vllm.model_executor.utils import set_weight_attrs

from .gguf_dequant import dequantize
from .gguf_repack import GGUF_NPU_REPACK_TYPES, repack_to_npu


class AscendGGUFLinearMethod(LinearMethodBase):
    """GGUF linear method that dequantizes to dense at load time (Pattern A).

    Args:
        quant_config: The Ascend GGUF config (currently carries no parameters;
            the quant type is per-tensor via ``qweight_type``).
    """

    def __init__(self, quant_config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)

        # Verbatim from upstream GGUFLinearMethod.create_weights: the
        # GGUFModelLoader recognizes these attrs and fills qweight/qweight_type.
        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        qweight_type = Parameter(
            torch.empty(len(output_partition_sizes), dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight_type,
            {
                "is_gguf_weight_type": True,
                "weight_type": 0,
                "shard_weight_type": {},
                "ignore_warning": True,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    # High-perf route: simple block types (Q8_0/Q4_0/Q4_1) are repacked to the
    # NPU op (weights stay quantized → real memory saving). False forces the
    # dense-dequant fallback (used by the embedding method, whose lookup/sampler
    # matmul needs a dense weight).
    high_perf: bool = True

    def process_weights_after_loading(self, layer: torch.nn.Module):
        """Repack simple types to the NPU op (high-perf) or dequant to dense (k-quants)."""
        dtype = self.params_dtype
        qweight = layer.qweight
        qweight_type = layer.qweight_type
        device = qweight.device

        # Gather (bytes, qtype) per shard on CPU, QKV-reordered.
        data_container = getattr(qweight, "data_container", None) or []
        if data_container:
            shard_ids = qweight.shard_id
            if "q" in shard_ids:  # QKV: loader order can be ['k','q','v'] → [q,k,v]
                shard_ids = ["q", "k", "v"]
            shard_specs = [
                (
                    data_container[qweight.shard_id_map[sid]].cpu(),
                    qweight_type.shard_weight_type.get(sid, qweight_type.weight_type),
                )
                for sid in shard_ids
            ]
        else:
            shard_specs = [(qweight.cpu(), qweight_type.weight_type)]

        if self.high_perf and shard_specs and all(qt in GGUF_NPU_REPACK_TYPES for _, qt in shard_specs):
            # HIGH-PERF: repack each shard to npu_weight_quant_batchmatmul format
            # (weights stay quantized → memory saving), concat along output dim.
            qws, scales, offsets = [], [], []
            group_size = None
            for bytes_, qt in shard_specs:
                qw, sc, off, group_size = repack_to_npu(bytes_.to(device), qt, dtype)
                qws.append(qw)
                scales.append(sc)
                offsets.append(off)
            layer.qweight = Parameter(torch.cat(qws, dim=1).to(device), requires_grad=False)
            layer.scales = Parameter(torch.cat(scales, dim=1).to(device), requires_grad=False)
            layer.offset = Parameter(torch.cat(offsets, dim=1).to(device), requires_grad=False)
            layer.group_size = group_size
            layer.output_size = sum(b.shape[0] for b, _ in shard_specs)  # logical total N
            layer.weight = Parameter(torch.empty(0, dtype=dtype, device=device), requires_grad=False)
        else:
            # DENSE fallback (k-quants Q4_K/Q5_K/Q6_K …): dequant on CPU, concat.
            dense = torch.cat([dequantize(b, qt, dtype) for b, qt in shard_specs], dim=0)
            layer.weight = Parameter(dense.to(device).contiguous(), requires_grad=False)
            layer.qweight = Parameter(torch.empty(0, dtype=dtype, device=device), requires_grad=False)

        layer.qweight_type = Parameter(torch.empty(0, dtype=torch.uint8, device=device), requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if layer.qweight.numel() > 0:  # high-perf (quantized) path
            if bias is not None and bias.dtype == torch.bfloat16:
                bias = bias.float()
            reshaped_x = x.reshape(-1, x.shape[-1])
            out = torch_npu.npu_weight_quant_batchmatmul(
                reshaped_x,
                layer.qweight,
                antiquant_scale=layer.scales,
                antiquant_offset=layer.offset,
                antiquant_group_size=layer.group_size,
                bias=bias,
            )
            return out.reshape(x.shape[:-1] + (layer.output_size,))
        # dense fallback (k-quants): layer.weight is dense [N, K].
        return F.linear(x, layer.weight, bias)


class AscendGGUFEmbeddingMethod(AscendGGUFLinearMethod):
    """GGUF embedding method: dequant to dense at load, plain embedding lookup.

    Embeddings (token_embd / lm_head) always use the dense path: the input
    embedding is a row lookup and the lm_head weight is matmul'd directly by the
    sampler, both needing a dense weight (so high_perf is forced off here).
    """

    high_perf = False

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, layer.weight)
