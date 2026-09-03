#!/usr/bin/env python3
"""Fused (stacked GEMM) vs per-expert (loop GEMM) MoE execution — the real cost
of de-fusing experts. Directly answers: is the fused path faster, and by how much?

fused   = one big GEMM over stacked expert weights (vLLM-style math)
per-expert = N small GEMMs in a Python loop (moesim hook-style)
"""
from __future__ import annotations

import time

import torch


def bench(fn, iters=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden, intermediate, num_experts, num_tokens = 512, 1024, 8, 64
    torch.manual_seed(0)

    W13 = torch.randn(num_experts * 2 * intermediate, hidden,
                      dtype=torch.float16, device=device)
    W2 = torch.randn(num_experts * intermediate, hidden,
                     dtype=torch.float16, device=device)
    w13_list = [W13[i * 2 * intermediate:(i + 1) * 2 * intermediate]
                for i in range(num_experts)]
    w2_list = [W2[i * intermediate:(i + 1) * intermediate]
               for i in range(num_experts)]
    x = torch.randn(num_tokens, hidden, dtype=torch.float16, device=device)

    def fused_forward():
        h = x @ W13.T
        h = h.view(num_tokens, num_experts, 2 * intermediate)
        g, u = h.chunk(2, dim=-1)
        y = torch.nn.functional.silu(g) * u
        out = torch.einsum('tei,eio->teo', y, W2.view(num_experts, intermediate, hidden))
        return out.reshape(num_tokens, num_experts * hidden)

    def perexpert_forward():
        out = torch.zeros(num_tokens, num_experts * hidden,
                          dtype=torch.float16, device=device)
        for i in range(num_experts):
            h = x @ w13_list[i].T
            g, u = h.chunk(2, dim=-1)
            y = torch.nn.functional.silu(g) * u
            out[:, i * hidden:(i + 1) * hidden] = y @ w2_list[i]
        return out

    # sanity: same result
    assert torch.allclose(fused_forward(), perexpert_forward(), atol=1e-2), "mismatch"

    fused_ms = bench(fused_forward)
    perexpert_ms = bench(perexpert_forward)

    print(f"=== fused vs per-expert MoE (device={device}, {num_tokens} tokens, "
          f"{num_experts} experts, hidden {hidden}, intermediate {intermediate}) ===")
    print(f"{'method':14s} {'ms/forward':>11s}")
    print(f"{'fused (stacked)':14s} {fused_ms:11.4f}")
    print(f"{'per-expert (loop)':14s} {perexpert_ms:11.4f}")
    print(f"\nfused is {perexpert_ms/fused_ms:.1f}x faster than per-expert loop.\n")
    print("read: this is the de-fusing cost. On GPU, one big GEMM beats N small "
          "GEMMs in a Python loop. Our per-expert hook pays this cost in exchange "
          "for per-expert placement freedom — the tradeoff is only worth it when "
          "placement freedom enables offloading a model that otherwise would not fit.")


if __name__ == "__main__":
    main()
