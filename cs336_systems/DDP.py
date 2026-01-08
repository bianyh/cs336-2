import torch
import torch.distributed as dist
from typing import List

# uv run pytest tests/test_ddp_individual_parameters.py
# uv run pytest tests/test_ddp.py

class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        # 修改：列表现在存储三元组 (handle, grad_tensor, parameter)
        # 必须显式持有 grad_tensor 的引用，因为它是异步通信的目标缓冲区
        self._work_handles = []

        # 1️⃣ 参数广播（确保初始权重一致）
        self._broadcast_parameters()

        # 2️⃣ 注册梯度 hook（逐参数、异步通信）
        self._register_gradient_hooks()

    def _broadcast_parameters(self):
        """
        将 rank 0 的参数广播到所有 rank
        """
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

    def _register_gradient_hooks(self):
            # 这里的 seen_storages 是为了处理“权重共享”的情况。
            # 比如有的模型 Embedding 层和输出层共用同一个权重矩阵，
            # 我们不想对同一块内存重复注册两次通信任务。
            seen_storages = set()

            for param in self.module.parameters():
                if not param.requires_grad:
                    continue

                # 获取参数底层数据的内存地址指针
                storage_ptr = param.data_ptr()
                if storage_ptr in seen_storages:
                    continue
                seen_storages.add(storage_ptr)

                # --- 高能预警：闭包函数 ---
                # 为什么需要 make_hook(p)？
                # 因为 Python 的循环变量绑定问题。如果我们直接用 param，
                # 所有的 hook 最后都会指向循环的最后一个参数。
                # 用 make_hook 把当前的 p "锁" 住。
                def make_hook(p):
                    
                    # 这就是当 loss.backward() 走到这一层时，实际执行的代码
                    # grad 参数就是 PyTorch 刚刚算出来的这一层的梯度
                    def hook(grad):
                        
                        # 1. 发射异步任务 (Fire and Forget)
                        # dist.all_reduce: 让所有卡把这个 grad 加起来。
                        # async_op=True: 关键！这句话说完立刻返回，不等待通信结束。
                        # 此时，网卡开始在后台疯狂传输数据，而 CPU 继续去算下一层的梯度。
                        # 注意：NCCL 会直接修改 grad 这块内存中的数据。
                        work = dist.all_reduce(
                            grad,
                            op=dist.ReduceOp.SUM,
                            async_op=True,
                        )
                        
                        # 2. 记账
                        # 我们把这笔“正在进行”的交易记录下来。
                        # 必须保存 grad 的引用，否则 Python 垃圾回收机制可能会把
                        # 正在通信的 grad 张量给销毁了，导致报错。
                        self._work_handles.append((work, grad, p))
                        
                        # 3. 【核心 trick：偷梁换柱】
                        # PyTorch 的逻辑是：拿到 hook 的返回值 -> 累加到 param.grad 上。
                        # 此时，上面的 grad 正在被网卡读写（脏数据），不能给 PyTorch 用。
                        # 所以，我们返回一个 全零张量 (zeros_like)。
                        # 意思是告诉 PyTorch：“这一层的梯度增量是 0，你啥也别干。”
                        # 真正的梯度去哪了？在上面的 grad 变量里，正被网卡处理呢。
                        # 注意！！！
                        # hook(grad) 的返回值
                        # ↓
                        # 会被 autograd 当作
                        # “这个节点对 param.grad 的增量”
                        # 也就是说：
                        # param.grad += hook_return_value
                        return torch.zeros_like(grad)
                    
                    return hook

                # 正式注册
                param.register_hook(make_hook(param))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """
        在 optimizer.step() 之前调用，
        确保所有梯度通信完成，并进行平均，最后回填到 param.grad
        """
        # 1. 遍历所有保存的 handle 和梯度
        for work, reduced_grad, param in self._work_handles:
            # 等待通信完成
            work.wait()

            # 此时 reduced_grad 中已经是 Sum 后的结果了
            # 进行平均： Grad = Sum / World_Size
            reduced_grad.div_(self.world_size)

            # 2. 手动将正确的结果赋值给 param.grad
            # 因为 hook 返回了 0，现在的 param.grad 应该是 0 (或者之前的累加值)
            # 我们直接使用 copy_ 将计算好的梯度覆盖进去
            if param.grad is None:
                param.grad = reduced_grad
            else:
                param.grad.copy_(reduced_grad)

        # 3. 清空 handle，为下一个 batch 做准备
        self._work_handles.clear()



class _Bucket:
    def __init__(self, params: List[torch.nn.Parameter], world_size: int):
        self.params = params
        self.world_size = world_size

        # flatten buffer
        self.sizes = [p.numel() for p in params]
        self.offsets = torch.cumsum(
            torch.tensor([0] + self.sizes[:-1]), dim=0
        ).tolist()

        device = params[0].device
        dtype = params[0].dtype
        total_size = sum(self.sizes)

        self.buffer = torch.zeros(total_size, device=device, dtype=dtype)

        self.ready_count = 0
        self.work = None

    def mark_ready(self):
        self.ready_count += 1
        return self.ready_count == len(self.params)

    def launch_allreduce(self):
        self.work = dist.all_reduce(
            self.buffer, op=dist.ReduceOp.SUM, async_op=True
        )

    def finalize(self):
        self.work.wait()
        self.buffer.div_(self.world_size)

        # scatter back
        for p, offset, size in zip(self.params, self.offsets, self.sizes):
            grad_view = self.buffer[offset : offset + size].view_as(p)
            if p.grad is None:
                p.grad = grad_view.clone()
            else:
                p.grad.copy_(grad_view)

class DDPBucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()

        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)

        self.buckets: List[_Bucket] = []
        self.param_to_bucket = {}

        self._broadcast_parameters()
        self._build_buckets()
        self._register_hooks()

    def _broadcast_parameters(self):
        with torch.no_grad():
            for p in self.module.parameters():
                dist.broadcast(p.data, src=0)

    def _build_buckets(self):
        params = list(self.module.parameters())[::-1]  # reverse order

        current_bucket = []
        current_size = 0

        for p in params:
            if not p.requires_grad:
                continue

            param_bytes = p.numel() * p.element_size()

            if current_bucket and current_size + param_bytes > self.bucket_size_bytes:
                bucket = _Bucket(current_bucket, self.world_size)
                self._register_bucket(bucket)
                current_bucket = []
                current_size = 0

            current_bucket.append(p)
            current_size += param_bytes

        if current_bucket:
            bucket = _Bucket(current_bucket, self.world_size)
            self._register_bucket(bucket)

    def _register_bucket(self, bucket: _Bucket):
        self.buckets.append(bucket)
        for p in bucket.params:
            self.param_to_bucket[p] = bucket

    def _register_hooks(self):
        for bucket in self.buckets:
            for p, offset, size in zip(
                bucket.params, bucket.offsets, bucket.sizes
            ):

                def make_hook(param, bucket, offset, size):
                    def hook(grad):
                        # copy into bucket
                        bucket.buffer[offset : offset + size].copy_(grad.view(-1))

                        if bucket.mark_ready():
                            bucket.launch_allreduce()

                        return torch.zeros_like(grad)

                    return hook

                p.register_hook(make_hook(p, bucket, offset, size))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for bucket in self.buckets:
            if bucket.work is not None:
                bucket.finalize()
