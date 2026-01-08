import torch
import math

# uv run pytest -k test_flash_forward_pass_pytorch
# uv run pytest -k test_flash_backward

# 需要自己实现前向传播和反向传播
class flash_attention_pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        orig_ndim = Q.ndim
        # QKV : (Batch_size, Head_nums, Token_Seq_length, Dim)
        if Q.ndim == 2:
            Q = Q.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
            K = K.unsqueeze(0).unsqueeze(0)
            V = V.unsqueeze(0).unsqueeze(0)
        elif Q.ndim == 3:
            Q = Q.unsqueeze(1)  # (B, 1, T, D)
            K = K.unsqueeze(1)
            V = V.unsqueeze(1)
        B, H, T, D = Q.shape
        device = Q.device
        dtype = Q.dtype

        # 每个block中有多少个query（Q）[一次处理的Q数量]
        BLOCK_Q = 16
        # 每个 block 里有多少个 key/value（K, V）
        BLOCK_KV = 16
        # Query block × Key block = attention tile
        # 即一个注意力瓦片
        # 这个注意力瓦片就是每次在SRAM中计算后写回HBM的大小

        # 这是标准Attention中需要除以的那个维度开方
        scale = 1.0 / math.sqrt(D)

        # FlashAttention与普通Attention开始不一样的地方
        # O 是最终 attention 输出
        # L 是每个 query 的 logsumexp（softmax 的“归一化常数”）存的是此前log(sum（exp(S)）)
        # S 是 QK/scale
        # softmax(S_i) = exp(S_i - logsumexp(S_i)) = e^S / ∑e^S
        # L 的存在，是 FlashAttention 能“边算边 softmax”而不存整张 attention matrix 的关键。
        # L 是可以递推求得的
        O = torch.zeros_like(Q)
        L = torch.zeros((B, H, T), device=device, dtype=dtype)

        # 主循环
        for b in range(B):
            for h in range(H):
                Q_bh = Q[b, h]
                K_bh = K[b, h]
                V_bh = V[b, h]
                for i in range(0, T, BLOCK_Q):
                    # 得到Q的块
                    Qi = Q_bh[i: i+BLOCK_Q]

                    # 把一会儿需要递归复写的几个先声明一下
                    # 最大值
                    mi = torch.full(
                        (Qi.shape[0],),
                        -float("inf"),
                        device=device,
                        dtype=dtype
                    )
                    # softmax的分母
                    li = torch.zeros(
                        (Qi.shape[0],),
                        device=device,
                        dtype=dtype
                    )
                    # softmax的分子
                    Oi = torch.zeros(
                        (Qi.shape[0], D),
                        device=device,
                        dtype=dtype
                    )

                    # 下一层循环，对kv的循环
                    for j in range(0, T, BLOCK_KV):
                        Kj = K_bh[j: j+BLOCK_KV]
                        Vj = V_bh[j: j+BLOCK_KV]

                        S = Qi @ Kj.T * scale

                        # 计算当前block的最大值
                        mij = torch.max(S, dim=1).values
                        mi_new = torch.maximum(mi, mij)

                        P = torch.exp(S - mi_new[:, None])
                        li_new = torch.exp(mi - mi_new) * li + torch.sum(P, dim=1)


                        Oi = (
                            Oi * (torch.exp(mi - mi_new) * li / li_new)[:, None]
                            + torch.matmul(P, Vj) / li_new[:, None]
                        )

                        mi = mi_new
                        li = li_new

                    O[b, h, i:i+BLOCK_Q] = Oi
                    L[b, h, i:i+BLOCK_Q] = mi + torch.log(li)

        # restore output shape
        if orig_ndim == 2:
            O = O.squeeze(0).squeeze(0)
            Q = Q.squeeze(0).squeeze(0)
            K = K.squeeze(0).squeeze(0)
            V = V.squeeze(0).squeeze(0)
            L = L.squeeze(0).squeeze(0)            
        elif orig_ndim == 3:
            O = O.squeeze(1)
            K = K.squeeze(1)
            V = V.squeeze(1)
            Q = Q.squeeze(1)
            L = L.squeeze(1)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.scale = 1.0 / math.sqrt(Q.shape[-1])
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

        dQ, dK, dV = flash_backward_torch(Q, K, V, O, dO, L, scale)

        return dQ, dK, dV, None
        
def flash_backward_torch(Q, K, V, O, dO, L, scale, is_causal = False):
    # 1、重算 S 和 P
    S = Q @ K.transpose(-1, -2) * scale
    if is_causal:
        T = Q.shape[1]
        mask = torch.triu(
            torch.ones(T, T, device=Q.device, dtype=torch.bool),
            diagonal=1
        )
        S = S.masked_fill(mask, float("-inf"))

    P = torch.exp(S - L.unsqueeze(-1)).to(Q.dtype)

    # 2、dV
    dV = P.transpose(-1, -2) @ dO

    # 3、计算D=dO*O
    D = torch.sum(dO * O, dim=-1)

    # 4、dS
    dOV = dO @ V.transpose(-1, -2)
    dS = P * (dOV - D.unsqueeze(-1))

    # 5、dQ, dK
    dQ = dS @ K * scale
    dK = dS.transpose(-2, -1) @ Q * scale

    return dQ, dK, dV