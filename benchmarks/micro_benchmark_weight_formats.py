#!/usr/bin/env python3
"""Micro-benchmark: compare npu_weight_quant_batchmatmul performance
with different weight formats.

This script directly tests the NPU operator with:
1. AWQ-style packed int32 weights (8 x sint4 per int32, packed along output dim)
2. GPTQ-style int4pack weights (via npu_convert_weight_to_int4pack)
3. GPTQ-style int8 weights (unpacked int8, like GPTQ-Int8)

The goal is to determine if the int4pack format is the root cause of
GPTQ-Int4 being ~2x slower than AWQ/GPTQ-Int8.

Usage:
    cd /data/ascend/vllm-ascend
    source /data/ascend/.venv/bin/activate
    python benchmarks/micro_benchmark_weight_formats.py
"""

import time
import json
import sys

import torch
import torch_npu


def create_packed_int32_weight(K: int, N: int, group_size: int, dtype: torch.dtype, device: str):
    """Create AWQ-style packed int32 weight + scales + offset.

    Weight shape: (K, N // 8) int32, each int32 holds 8 sint4 values
    Scale shape: (K // group_size, N) float16
    Offset shape: (K // group_size, N) float16
    """
    pack_factor = 8
    N_packed = N // pack_factor
    num_groups = K // group_size

    # Create random int4 values [-8, 7], pack into int32
    qweight_int8 = torch.randint(-8, 8, (K, N), dtype=torch.int8, device=device)
    # Convert to uint4 [0, 15] for packing
    qweight_uint4 = (qweight_int8.to(torch.int32) + 8) & 0xF
    # Pack 8 values into each int32
    packed = torch.zeros((K, N_packed), dtype=torch.int32, device=device)
    for i in range(pack_factor):
        packed.bitwise_or_((qweight_uint4[:, i::pack_factor] & 0xF) << (4 * i))
    # XOR to convert to sint4 representation (same as AWQ)
    packed.bitwise_xor_(0x88888888)

    scales = torch.randn(num_groups, N, dtype=dtype, device=device).abs() + 0.01
    offset = torch.zeros(num_groups, N, dtype=dtype, device=device)

    return packed, scales, offset, N  # N is the output size


def create_int4pack_weight(K: int, N: int, group_size: int, dtype: torch.dtype, device: str):
    """Create GPTQ-style int4pack weight via npu_convert_weight_to_int4pack.

    Weight shape: result of npu_convert_weight_to_int4pack
    Scale shape: (K // group_size, N) float16
    Offset shape: (K // group_size, N) float16
    """
    num_groups = K // group_size

    # Create random int4 values [-8, 7]
    qweight_int8 = torch.randint(-8, 8, (K, N), dtype=torch.int8, device=device)
    qweight_int32 = qweight_int8.to(torch.int32)
    packed = torch_npu.npu_convert_weight_to_int4pack(qweight_int32)

    scales = torch.randn(num_groups, N, dtype=dtype, device=device).abs() + 0.01
    offset = torch.zeros(num_groups, N, dtype=dtype, device=device)

    return packed, scales, offset, N


def create_int8_weight(K: int, N: int, group_size: int, dtype: torch.dtype, device: str):
    """Create GPTQ-Int8 style weight (unpacked int8).

    Weight shape: (K, N) int8
    Scale shape: (K // group_size, N) float16
    Offset shape: (K // group_size, N) float16
    """
    num_groups = K // group_size

    qweight = torch.randint(-128, 128, (K, N), dtype=torch.int8, device=device)
    scales = torch.randn(num_groups, N, dtype=dtype, device=device).abs() + 0.01
    offset = torch.zeros(num_groups, N, dtype=dtype, device=device)

    return qweight, scales, offset, N


def benchmark_format(name: str, weight, scales, offset, output_size: int,
                     x: torch.Tensor, group_size: int, num_iters: int = 50):
    """Benchmark npu_weight_quant_batchmatmul with given weight format."""
    # Warmup
    for _ in range(5):
        _ = torch_npu.npu_weight_quant_batchmatmul(
            x, weight,
            antiquant_scale=scales,
            antiquant_offset=offset,
            antiquant_group_size=group_size,
        )
    torch.npu.synchronize()

    # Measure
    times = []
    for _ in range(num_iters):
        torch.npu.synchronize()
        start = time.perf_counter()
        out = torch_npu.npu_weight_quant_batchmatmul(
            x, weight,
            antiquant_scale=scales,
            antiquant_offset=offset,
            antiquant_group_size=group_size,
        )
        torch.npu.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_ms = sum(times) / len(times) * 1000
    min_ms = min(times) * 1000
    max_ms = max(times) * 1000

    print(f"  {name}:")
    print(f"    Weight shape: {weight.shape}, dtype: {weight.dtype}")
    print(f"    Avg: {avg_ms:.2f} ms, Min: {min_ms:.2f} ms, Max: {max_ms:.2f} ms")
    print(f"    Output shape: {out.shape}")
    print(f"    Memory: weight={weight.numel() * weight.element_size() / 1024 / 1024:.1f} MB")

    return {
        "name": name,
        "weight_shape": list(weight.shape),
        "weight_dtype": str(weight.dtype),
        "avg_ms": round(avg_ms, 2),
        "min_ms": round(min_ms, 2),
        "max_ms": round(max_ms, 2),
        "output_shape": list(out.shape),
        "weight_memory_mb": round(weight.numel() * weight.element_size() / 1024 / 1024, 1),
    }


def main():
    print("=" * 70)
    print("Micro-benchmark: npu_weight_quant_batchmatmul weight formats")
    print("=" * 70)

    device = "npu:0"
    dtype = torch.float16

    # Qwen2.5-0.5B model dimensions (typical linear layer)
    # MLP gate_up: input=896, output=4864 (gate 2432 + up 2432)
    # But we test with smaller sizes for faster iteration
    configs = [
        # (K, N, group_size, description)
        (896, 4864, 128, "Qwen2.5-0.5B MLP gate_up"),
        (1024, 4096, 128, "Standard 1K x 4K"),
        (2048, 4096, 128, "Standard 2K x 4K"),
        (4096, 4096, 128, "Standard 4K x 4K"),
    ]

    batch_size = 32  # typical decode batch
    num_iters = 50

    all_results = {}

    for K, N, group_size, desc in configs:
        print(f"\n--- {desc}: K={K}, N={N}, group_size={group_size}, batch={batch_size} ---")

        # Input tensor
        x = torch.randn(batch_size, K, dtype=dtype, device=device)

        # Memory before
        torch.npu.reset_peak_memory_stats()
        torch.npu.synchronize()
        mem_before = torch.npu.memory_allocated() / (1024 * 1024)

        results = []

        # 1. AWQ-style packed int32
        try:
            w1, s1, o1, out1 = create_packed_int32_weight(K, N, group_size, dtype, device)
            r1 = benchmark_format("AWQ packed_int32", w1, s1, o1, out1, x, group_size, num_iters)
            results.append(r1)
            del w1, s1, o1
        except Exception as e:
            print(f"  AWQ packed_int32: FAILED - {e}")
            results.append({"name": "AWQ packed_int32", "error": str(e)})

        # 2. GPTQ-style int4pack
        try:
            w2, s2, o2, out2 = create_int4pack_weight(K, N, group_size, dtype, device)
            r2 = benchmark_format("GPTQ int4pack", w2, s2, o2, out2, x, group_size, num_iters)
            results.append(r2)
            del w2, s2, o2
        except Exception as e:
            print(f"  GPTQ int4pack: FAILED - {e}")
            results.append({"name": "GPTQ int4pack", "error": str(e)})

        # 3. GPTQ-Int8 style unpacked int8
        try:
            w3, s3, o3, out3 = create_int8_weight(K, N, group_size, dtype, device)
            r3 = benchmark_format("GPTQ-Int8 int8", w3, s3, o3, out3, x, group_size, num_iters)
            results.append(r3)
            del w3, s3, o3
        except Exception as e:
            print(f"  GPTQ-Int8 int8: FAILED - {e}")
            results.append({"name": "GPTQ-Int8 int8", "error": str(e)})

        # Memory after
        torch.npu.synchronize()
        mem_peak = torch.npu.max_memory_allocated() / (1024 * 1024)
        print(f"    Peak NPU memory: {mem_peak:.1f} MB")

        all_results[f"K{K}_N{N}"] = {
            "description": desc,
            "config": {"K": K, "N": N, "group_size": group_size, "batch_size": batch_size},
            "results": results,
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<30} {'AWQ int32':>12} {'GPTQ int4pack':>14} {'GPTQ-Int8':>12} {'Ratio int4pack/AWQ':>20}")
    print("-" * 90)

    for key, data in all_results.items():
        results = data["results"]
        awq_ms = next((r.get("avg_ms") for r in results if r.get("name") == "AWQ packed_int32"), None)
        int4pack_ms = next((r.get("avg_ms") for r in results if r.get("name") == "GPTQ int4pack"), None)
        int8_ms = next((r.get("avg_ms") for r in results if r.get("name") == "GPTQ-Int8 int8"), None)

        ratio = f"{int4pack_ms / awq_ms:.2f}x" if awq_ms and int4pack_ms else "N/A"
        awq_str = f"{awq_ms:.2f}" if awq_ms else "FAIL"
        int4pack_str = f"{int4pack_ms:.2f}" if int4pack_ms else "FAIL"
        int8_str = f"{int8_ms:.2f}" if int8_ms else "FAIL"

        print(f"{data['description']:<30} {awq_str:>12} {int4pack_str:>14} {int8_str:>12} {ratio:>20}")

    # Save results
    output_path = "benchmarks/results/micro_benchmark_weight_formats.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
