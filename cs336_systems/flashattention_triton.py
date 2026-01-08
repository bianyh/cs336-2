import torch
import triton
import triton.language as tl
import math
from .flashattention_pytorch import flash_backward_torch

# uv run pytest -k test_flash_forward_pass_triton

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    # 比如 Q 的 layout 是 (B, T, D)，那么：
    # stride_qb：跨一个 batch 要跳多少元素
    # stride_qq：跨一个 query
    # stride_qd：跨一个 dim
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # 当前kernel负责：
    # 第 batch_index 个batch
    # 第 query_tile_index * Q_TILE_SIZE : (query_tile_index+1)*Q_TILE_SIZE 个 query
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # 声明 Block pointers 块指针
    # “这是一个指向 Q[b, i:i+Bq, :] 的二维视图”
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    # order 是一个整数元组，表示维度的重要性/连续性顺序。数值越小的维度在内存中越连续（ stride 越小）。
    # order=(1, 0)就是行主序模式，按行读取

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # 加载Qi
    Qi = tl.load(Q_block_ptr)

    # 初始化FlashAttention的几个需要递推存储的变量
    mi = tl.full((Q_TILE_SIZE,), -float("inf"), tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), tl.float32)
    Oi = tl.zeros((Q_TILE_SIZE, D), tl.float32)

    # 循环每一个key tile
    for j in range(0, N_KEYS, K_TILE_SIZE):
        Kj = tl.load(K_block_ptr)
        Vj = tl.load(V_block_ptr)

        # S = Qi @ Kj * scale
        S = tl.dot(Qi, Kj) * scale

        # 此时S的形状为[Q_tile, K_tile]
        if is_causal:
            # q_idx: shape (Q_TILE_SIZE,)
            # k_idx: shape (K_TILE_SIZE,)
            q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = j + tl.arange(0, K_TILE_SIZE)
            # 扩展成（Q_tile, K_tile）
            q_idx = q_idx[:, None]
            k_idx = k_idx[None, :]
            # 当两个张量比较：
            # (Q, 1) 和
            # (1, K)
            # 它们会被自动广播成：(Q, K)
            mask = k_idx > q_idx
            # 应用mask
            S = tl.where(mask, -float("inf"), S)



        # m_ij
        mij = tl.max(S, axis=1)
        mi_new = tl.maximum(mi, mij)

        # p = exp(S - mi_new)
        p = tl.exp(S - mi_new[:, None])
        
        # l_new
        li_new = tl.exp(mi - mi_new) * li + tl.sum(p, axis=1)

        Oi = (
            Oi * (li * tl.exp(mi - mi_new) / li_new)[:, None]
            + tl.dot(p.to(Vj.dtype), Vj) / li_new[:, None]
        )

        mi = mi_new
        li = li_new

        # 增长指针地址
        K_block_ptr = tl.advance(K_block_ptr, (0, K_TILE_SIZE))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # 把运算结果写入 O 和 L
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_ptr = (
        L_ptr
        + batch_index * stride_lb
        + (query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)) * stride_lq
    )

    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty))
    tl.store(L_ptr, mi + tl.log(li))



class flash_attention_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, T, D = Q.shape
        O = torch.empty_like(Q)
        L = torch.empty((B, T), device=Q.device, dtype=torch.float32)


        # query_tile_index = tl.program_id(0)
        # batch_index = tl.program_id(1)
        grid = (triton.cdiv(T, 16), B)

        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            T, T,
            1.0 / math.sqrt(D),
            D=D,
            Q_TILE_SIZE=16,
            K_TILE_SIZE=16,
            is_causal=is_causal,
        )
        
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.scale = 1 / math.sqrt(Q.shape[-1])
        ctx.is_causal = is_causal
        return O
    
    
    @staticmethod
    def backward(ctx, dO):
        """
        Q, K, V: (B, T, D)
        O, dO: (B, T, D)
        L: (B, T) # log-sum-exp
        """
        Q, K, V, O, L = ctx.saved_tensors
        scale = ctx.scale

        dQ, dK, dV = flash_backward_torch(Q, K, V, O, dO, L, scale, ctx.is_causal)
        
        return dQ, dK, dV, None
        
