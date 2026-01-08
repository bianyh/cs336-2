# benchmark.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import argparse
import time
import torch
from cs336_basics.model import BasicsTransformerLM
import statistics


MODEL_CONFIGS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7b": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def parse_args():
    parser = argparse.ArgumentParser("Transformer benchmarking")

    parser.add_argument("--model-size", type=str, default="small",
                        choices=["small", "medium", "large", "xl", "2.7b"])
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--do-backward", type=bool, default=False)

    return parser.parse_args()


def main(args):
    # 开始记录内存消耗
    # torch.cuda.memory._record_memory_history(max_entries=10000000)

    device = torch.device("cuda:2")

    config = MODEL_CONFIGS[args.model_size]

    model = BasicsTransformerLM(
        context_length=256,
        rope_theta=10000.0,
        vocab_size=10_000,
        **config
    ).to(device)

    # 进行compile优化
    # model = torch.compile(model)


    if args.do_backward:
        model.train()
    else:
        model.eval()


    input_ids = torch.randint(
        0, 10_000,
        (args.batch_size, args.seq_len),
        device=device
    )


    # Warmup
    for _ in range(args.num_warmup):
        output = model(input_ids)
        if args.do_backward:
            loss = output.mean()
            loss.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()


    # 正式开始计时
    times = []

    for _ in range(args.num_steps):
        start = time.perf_counter()

        output = model(input_ids)
        if args.do_backward:
            loss = output.mean()
            loss.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        end = time.perf_counter()
        times.append(end - start)



    mean = statistics.mean(times)
    stdev = statistics.stdev(times)

    mode = "forward+backward" if args.do_backward else "forward-only"
    print(f"{mode} - {args.model_size} pass: mean={mean*1000:.2f} ms, std={stdev*1000:.2f} ms")

    # 保存记录的内存变化到pickle文件中
    # torch.cuda.memory._dump_snapshot('/home/bianyuhan/LLM Learning/cs336/cs336-2/memory_snapshot.pickle')
    # 可以使用 https://docs.pytorch.org/memory_viz 网站进行该文件的查看、分析
    # 停止内存记录
    # torch.cuda.memory._record_memory_history(enabled=None)



if __name__ == "__main__":
    args = parse_args()

    # main(args)

    for i in ["small", "medium", "large", "xl", "2.7b"]:
        args.model_size = i
        main(args)
