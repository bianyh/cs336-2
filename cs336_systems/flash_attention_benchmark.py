import math
import torch
import triton
import triton.testing
from itertools import product
from tqdm import tqdm

# ===== import YOUR implementations =====
from flashattention_pytorch import flash_attention_pytorch
from flashattention_triton import flash_attention_triton


# ============================================================
# Naive PyTorch Attention (reference, optional)
# ============================================================
def pytorch_attention(q, k, v, causal=True):
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)

    scores = torch.matmul(q, k.transpose(-1, -2)) * scale

    if causal:
        T = q.shape[-2]
        mask = torch.triu(
            torch.ones(T, T, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))

    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, v)


# ============================================================
# Benchmark helpers
# ============================================================
def bench_forward_flash(fn, q, k, v):
    def fwd():
        fn.apply(q, k, v, True)
    return triton.testing.do_bench(fwd)


def bench_backward_flash(fn, q, k, v):
    out = fn.apply(q, k, v, True)
    grad = torch.randn_like(out)

    def bwd():
        out.backward(grad, retain_graph=True)
    return triton.testing.do_bench(bwd)


def bench_fwd_bwd_flash(fn, q, k, v):
    def fwd_bwd():
        out = fn.apply(q, k, v, True)
        out.sum().backward()
    return triton.testing.do_bench(fwd_bwd)


def bench_forward_naive(q, k, v):
    def fwd():
        pytorch_attention(q, k, v, True)
    return triton.testing.do_bench(fwd)


def bench_backward_naive(q, k, v):
    out = pytorch_attention(q, k, v, True)
    grad = torch.randn_like(out)

    def bwd():
        out.backward(grad, retain_graph=True)
    return triton.testing.do_bench(bwd)


def bench_fwd_bwd_naive(q, k, v):
    def fwd_bwd():
        out = pytorch_attention(q, k, v, True)
        out.sum().backward()
    return triton.testing.do_bench(fwd_bwd)


# ============================================================
# Main benchmark
# ============================================================
def main():
    device = "cuda"
    torch.manual_seed(0)

    batch_size = 1
    seq_lens = [128, 256]
    d_models = [16, 32]
    dtypes = [torch.bfloat16, torch.float32]

    results = []

    for T, D, dtype in tqdm(product(seq_lens, d_models, dtypes)):
        # Skip insane naive cases
        if T > 8192:
            run_naive = False
        else:
            run_naive = True

        q = torch.randn(batch_size, T, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # ---------------- FlashAttention PyTorch ----------------
        fwd_pt = bench_forward_flash(flash_attention_pytorch, q, k, v)
        bwd_pt = bench_backward_flash(flash_attention_pytorch, q, k, v)
        fb_pt  = bench_fwd_bwd_flash(flash_attention_pytorch, q, k, v)

        # ---------------- FlashAttention Triton ----------------
        fwd_tr = bench_forward_flash(flash_attention_triton, q, k, v)
        bwd_tr = bench_backward_flash(flash_attention_triton, q, k, v)
        fb_tr  = bench_fwd_bwd_flash(flash_attention_triton, q, k, v)

        # ---------------- Naive PyTorch ----------------
        if run_naive:
            fwd_nv = bench_forward_naive(q, k, v)
            bwd_nv = bench_backward_naive(q, k, v)
            fb_nv  = bench_fwd_bwd_naive(q, k, v)
        else:
            fwd_nv = bwd_nv = fb_nv = None

        results.append({
            "T": T,
            "D": D,
            "dtype": str(dtype).replace("torch.", ""),
            "pt_fwd": fwd_pt,
            "pt_bwd": bwd_pt,
            "pt_fb": fb_pt,
            "tr_fwd": fwd_tr,
            "tr_bwd": bwd_tr,
            "tr_fb": fb_tr,
            "nv_fwd": fwd_nv,
            "nv_bwd": bwd_nv,
            "nv_fb": fb_nv,
        })

    # ============================================================
    # Print Markdown Table
    # ============================================================
    print("\n## FlashAttention Benchmark (H100, batch=1, causal=True)\n")
    print(
        "| T | D | dtype | "
        "PT fwd (ms) | PT bwd (ms) | PT fwd+bwd (ms) | "
        "TR fwd (ms) | TR bwd (ms) | TR fwd+bwd (ms) | "
        "Naive fwd (ms) | Naive bwd (ms) | Naive fwd+bwd (ms) |"
    )
    print("|---|---|-------|" + "----|" * 9)

    for r in results:
        def fmt(x):
            return "-" if x is None else f"{x:.3f}"

        print(
            f"| {r['T']} | {r['D']} | {r['dtype']} | "
            f"{fmt(r['pt_fwd'])} | {fmt(r['pt_bwd'])} | {fmt(r['pt_fb'])} | "
            f"{fmt(r['tr_fwd'])} | {fmt(r['tr_bwd'])} | {fmt(r['tr_fb'])} | "
            f"{fmt(r['nv_fwd'])} | {fmt(r['nv_bwd'])} | {fmt(r['nv_fb'])} |"
        )


if __name__ == "__main__":
    main()
