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
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.gguf import GGUFUninitializedParameter
from vllm.model_executor.utils import set_weight_attrs

from .gguf_dequant import dequantize


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

    def process_weights_after_loading(self, layer: torch.nn.Module):
        """Dequantize each GGUF shard to dense and concatenate along the output dim."""
        dtype = self.params_dtype
        qweight = layer.qweight
        qweight_type = layer.qweight_type
        device = qweight.device  # keep the dequantized weight on the model device

        shards = []
        data_container = getattr(qweight, "data_container", None) or []
        if data_container:
            # Fused / multi-shard layer (e.g. gate_up_proj, qkv_proj): the
            # loader placed each shard's raw quantized bytes in data_container.
            for sid in qweight.shard_id:
                idx = qweight.shard_id_map[sid]
                qw_bytes = data_container[idx]
                qtype = qweight_type.shard_weight_type.get(sid, qweight_type.weight_type)
                shards.append(dequantize(qw_bytes.to(device), qtype, dtype))
            # All shards share the input dim K (common case); concat along N.
            dense = torch.cat(shards, dim=0)
        else:
            # Single (non-fused) materialized quantized weight.
            qtype = qweight_type.weight_type
            dense = dequantize(qweight, qtype, dtype)

        layer.weight = Parameter(dense.to(device).contiguous(), requires_grad=False)
        # Release the quantized intermediates.
        layer.qweight = Parameter(torch.empty(0, dtype=dtype, device=device), requires_grad=False)
        layer.qweight_type = Parameter(torch.empty(0, dtype=torch.uint8, device=device), requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # layer.weight is dense [N, K] (output, input) after process.
        return F.linear(x, layer.weight, bias)


class AscendGGUFEmbeddingMethod(AscendGGUFLinearMethod):
    """GGUF embedding method: dequant to dense at load, plain embedding lookup.

    Inherits create_weights / process_weights_after_loading from
    AscendGGUFLinearMethod (so qweight/qweight_type are created and dequantized
    to dense ``layer.weight``). The GGUF VocabParallelEmbedding (e.g. lm_head)
    calls ``embedding(layer, x)``; since the weight is already dense, this is a
    plain lookup.
    """

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, layer.weight)
