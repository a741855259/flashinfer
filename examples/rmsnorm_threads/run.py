"""Compare thread caps using the same unmodified FP16 RMSNorm CUDA kernel."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Keep generated build artifacts outside the source/package directories.
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/flashinfer-rmsnorm-threads")
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
cuda = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cu13"
if cuda.joinpath("bin/nvcc").exists():
    os.environ.setdefault("CUDA_HOME", str(cuda))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/rmsnorm-threads"))
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args()
    if args.trials < 2:
        parser.error("--trials must be >= 2")

    import torch
    from flashinfer.jit.core import gen_jit_spec
    from flashinfer.testing import bench_gpu_time

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    torch.manual_seed(42)
    source = Path(__file__).with_name("launch.cu")
    # pip's CUDA toolkit uses lib/, whereas a system toolkit normally uses lib64/.
    link_flags = []
    for library_dir in (cuda / "lib", Path("/usr/lib/wsl/lib")):
        if library_dir.is_dir():
            link_flags.extend([f"-L{library_dir}", f"-Wl,-rpath,{library_dir}"])
    module = gen_jit_spec(
        "rmsnorm_threads_lesson",
        [source],
        extra_include_paths=[ROOT / "include", ROOT / "csrc"],
        extra_cuda_cflags=["-DENABLE_BF16", "-DENABLE_FP8", "--ptxas-options=-v"],
        extra_ldflags=link_flags,
    ).build_and_load()
    args.output.mkdir(parents=True, exist_ok=True)
    props = torch.cuda.get_device_properties(0)
    metadata = {
        "gpu": props.name,
        "compute_capability": torch.cuda.get_device_capability(),
        "sm_count": props.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "norm_source_sha256": hashlib.sha256(
            (ROOT / "include/flashinfer/norm.cuh").read_bytes()
        ).hexdigest(),
        "timing": "CUDA graph, 100 calls/graph, repeated inputs, no forced L2 flush",
        "trials": args.trials,
        "pdl": False,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    rows = []
    for m in (1, 32, 1024):
        for d in (1024, 4096, 16384):
            x = torch.randn(m, d, device="cuda", dtype=torch.float16)
            w = torch.randn(d, device="cuda", dtype=torch.float16)
            xf = x.float()
            ref = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
                   * w.float()).half()
            outputs = {cap: torch.empty_like(x) for cap in (1024, 256)}
            errors = {}
            for cap, out in outputs.items():
                module.run(out, x, w, cap)
                torch.cuda.synchronize()
                torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
                errors[cap] = (out.float() - ref.float()).abs().max().item()
            if d == 1024:
                # Both caps launch exactly the same shape: a timing-noise control.
                torch.testing.assert_close(outputs[1024], outputs[256], rtol=0, atol=0)
            measurements = {1024: [], 256: []}
            for trial in range(args.trials):
                order = (1024, 256) if trial % 2 == 0 else (256, 1024)
                for cap in order:
                    times = bench_gpu_time(
                        module.run,
                        input_args=(outputs[cap], x, w, cap),
                        use_cuda_graph=True,
                        num_iters_within_graph=100,
                        cold_l2_cache=False,
                        dry_run_time_ms=20,
                        repeat_time_ms=60,
                    )
                    measurements[cap].append(statistics.median(times) * 1000)
            speedup = statistics.median(measurements[1024]) / statistics.median(measurements[256])
            for cap in (1024, 256):
                threads, rounds, smem, regs, local, blocks = module.resources(d, cap)
                row = dict(M=m, D=d, cap=cap, threads=threads, rounds=rounds,
                           smem_bytes=smem, registers_per_thread=regs,
                           local_bytes_per_thread=local, resident_blocks_per_sm=blocks,
                           median_us=statistics.median(measurements[cap]),
                           min_trial_us=min(measurements[cap]),
                           max_trial_us=max(measurements[cap]),
                           max_abs_error=errors[cap], speedup_256=speedup,
                           trial_us=json.dumps(measurements[cap]))
                rows.append(row)
                print(f"M={m:4} D={d:5} cap={cap:4} threads={threads:4} "
                      f"rounds={rounds} time={row['median_us']:.3f} us "
                      f"error={errors[cap]:.6g} speedup256={speedup:.3f}", flush=True)
            with (args.output / "results.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    main()
