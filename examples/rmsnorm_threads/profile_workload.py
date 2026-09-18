"""Isolate one warmed RMSNorm launch for Nsight Compute, or recheck its timing."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess

import run as experiment  # Reuse the lesson's import, CUDA and cache setup.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=int, choices=(1024, 256), default=1024)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--pair", action="store_true", help="Profile three alternating pairs in one process")
    parser.add_argument("--output", type=Path, default=Path("/tmp/rmsnorm-profile-timing.json"))
    args = parser.parse_args()

    import torch
    from flashinfer.jit.core import gen_jit_spec
    from flashinfer.testing import bench_gpu_time

    flags = []
    for directory in (experiment.cuda / "lib", Path("/usr/lib/wsl/lib")):
        if directory.is_dir():
            flags.extend([f"-L{directory}", f"-Wl,-rpath,{directory}"])
    module = gen_jit_spec(
        "rmsnorm_threads_lesson",
        [Path(__file__).with_name("launch.cu")],
        extra_include_paths=[experiment.ROOT / "include", experiment.ROOT / "csrc"],
        extra_cuda_cflags=["-DENABLE_BF16", "-DENABLE_FP8", "--ptxas-options=-v"],
        extra_ldflags=flags,
    ).build_and_load()
    torch.manual_seed(42)
    x = torch.randn(1024, 4096, device="cuda", dtype=torch.float16)
    w = torch.randn(4096, device="cuda", dtype=torch.float16)
    outputs = {cap: torch.empty_like(x) for cap in (1024, 256)}
    xf = x.float()
    ref = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6) * w.float()).half()
    for cap in outputs:
        module.run(outputs[cap], x, w, cap)
        torch.cuda.synchronize()
        torch.testing.assert_close(outputs[cap], ref, rtol=1e-3, atol=1e-3)

    if args.benchmark:
        samples = {1024: [], 256: []}
        telemetry = []
        for trial in range(7):
            for cap in ((1024, 256) if trial % 2 == 0 else (256, 1024)):
                times = bench_gpu_time(
                    module.run, input_args=(outputs[cap], x, w, cap),
                    use_cuda_graph=True, num_iters_within_graph=100,
                    cold_l2_cache=False, dry_run_time_ms=20, repeat_time_ms=60,
                )
                samples[cap].append(statistics.median(times) * 1000)
                # Outside the timed interval; a post-trial snapshot, not a trace of kernel clocks.
                snapshot = subprocess.run(
                    ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False,
                )
                telemetry.append({"trial": trial, "cap": cap,
                                  "post_trial_gpu_state": snapshot.stdout.strip(),
                                  "query_error": snapshot.stderr.strip()})
        result = {
            "gpu": torch.cuda.get_device_name(), "M": 1024, "D": 4096,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "timing": "unprofiled CUDA graph, 100 calls, no forced L2 flush",
            "trial_us": samples,
            "median_us": {cap: statistics.median(v) for cap, v in samples.items()},
            "resources": {cap: list(module.resources(4096, cap)) for cap in samples},
            "post_trial_telemetry": telemetry,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    else:
        # With ncu --profile-from-start off, setup, validation and warmup are excluded.
        for _ in range(20):
            module.run(outputs[args.cap], x, w, args.cap)
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        caps = (1024, 256, 256, 1024, 1024, 256) if args.pair else (args.cap,)
        for cap in caps:
            module.run(outputs[cap], x, w, cap)
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        print(f"Target execution completed: cap={args.cap}, M=1024, D=4096; "
              "check the ncu exit status/log for counter collection success")


if __name__ == "__main__":
    main()
