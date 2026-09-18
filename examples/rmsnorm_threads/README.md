# RMSNorm 线程上限实验

本实验直接调用仓库的 `flashinfer::norm::RMSNormKernel<8, half>`，对比
1024（原 launcher 策略）和 256 两种线程上限。两个方案使用同一个编译后的
kernel，只改变 block 维度以及对应的归约 shared memory 大小。没有修改生产
kernel，也没有使用 CuTe DSL 后端。这次测的是普通 RMSNorm，不是融合残差版本。

## 运行

在仓库根目录，使用已经安装 FlashInfer 依赖的 CUDA Python 环境：

```bash
/root/Infer/.venv/bin/python examples/rmsnorm_threads/run.py \
    --output /tmp/rmsnorm-threads-results
```

首次使用会 JIT 编译。脚本会为当前虚拟环境补齐 bin 搜索路径，并识别 pip CUDA
的 `nvidia/cu13` 路径；其他安装方式可通过 `CUDA_HOME` 指定 toolkit。
编译缓存默认位于 `/tmp/flashinfer-rmsnorm-threads`。

## 实验控制

- FP16，epsilon=1e-6，M={1,32,1024}，D={1024,4096,16384}。
- 输入和权重固定，预分配独立输出；普通 RMSNorm 不修改输入。
- 对每个配置检查 FP32 reference，rtol=atol=1e-3。
- D=1024 时两种上限都启动 128 个线程，是无实际变更的对照组，并检查输出逐位一致。
- 关闭 PDL；CUDA Graph 内含 100 次调用，减少 Python 提交开销。
- 7 个 trial，交替测试两种配置的先后顺序；报告 trial 中位数的中位数。
- 重复使用相同 buffer，不主动清空 L2。大 tensor 未必能装进 L2，不能称为全部命中缓存。
- 这些结果不包含 JIT、分配和 reference 计算时间，也不代表端到端模型延迟。

`results.csv` 保存各 trial 延迟、误差和资源信息；`metadata.json` 保存设备、
软件版本与 norm.cuh 的 SHA256。`local_bytes_per_thread` 是 CUDA 属性查询结果，
并非独立的 spill 计数；本次 nvcc 的 ptxas 输出另确认 spill load/store 均为 0。

## 本次实测（2026-09-16）

后续 profiling 发现不同条件下结论会反转，不能把下表外推为稳定加速。详见
[PROFILING.md](PROFILING.md) 的新测量和限制。

RTX 2050 4 GB，16 SM，SM86，PyTorch 2.14.0+cu130，CUDA 13.0，WSL。
原始结果保存在本目录 `results/`，18 个配置全部通过正确性检查。
最大绝对误差为 0.00390625，满足上述结合相对误差的逐元素判定。

| M | D | 原策略 μs | cap=256 μs | 原策略/256 |
|---:|---:|---:|---:|---:|
| 1 | 1024 | 1.659 | 1.659 | 1.000 |
| 1 | 4096 | 2.007 | 2.263 | 0.887 |
| 1 | 16384 | 4.424 | 5.489 | 0.806 |
| 32 | 1024 | 3.389 | 3.430 | 0.988 |
| 32 | 4096 | 5.458 | 5.868 | 0.930 |
| 32 | 16384 | 33.213 | 33.843 | 0.981 |
| 1024 | 1024 | 71.700 | 70.246 | 1.021 |
| 1024 | 4096 | 313.948 | 252.446 | 1.244 |
| 1024 | 16384 | 1465.784 | 982.845 | 1.491 |

### 怎样解读

1. 小 M 时更少的线程意味着更多循环，但没有足够多的 block 利用额外驻留容量。
   单行更慢符合这一解释；这不是 profiler 对原因的独立证明。
2. D=4096：512 线程可驻留 3 block/SM，256 线程可驻留 6 block/SM，
   二者都是 48 warp/SM。因此加速不能简单归因于更高 occupancy。block 粒度、
   每线程工作量、归约组织和访存执行均变化，精确归因需要进一步 profiling。
3. D=16384：1024 线程可驻留 1 block/SM，即 32 warp；256 线程可驻留
   6 block/SM，即 48 warp。较高的驻留容量可能有助于大 M 下隐藏延迟。
4. 同一个 kernel 二进制每线程均使用 40 个寄存器，无 spill。线程减少不代表
   每线程寄存器数必然增加：本实现通过运行时循环复用局部变量，并不保留整行。
5. 理论驻留量来自 CUDA occupancy API，不是运行期间的实测 active warps。
6. D=1024 两方案实际上相同，也出现约 1–2% 差异，提醒我们不应解释微小差别。
   CSV 中保留 trial 范围；设备未锁频，本次结果也不应外推到其他 GPU。

当前结论支持按 workload 选择策略，不支持把生产实现的上限一律改成 256。
