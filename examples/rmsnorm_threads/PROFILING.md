# RMSNorm profiling：区分测量事实与性能假设

## 采集完成后的结论（2026-09-17）

用户开放 Windows 性能计数器后，采集成功。原先“256 线程更快”的结论必须限定
运行条件；本次没有得到一个对所有条件成立的加速根因。

### 可比性检查与补充实验

1. 首轮不控制缓存/频率，两个进程的 SM 频率约为 1240 与 251 MHz，不能直接归因。
2. 请求 `--clock-control base --cache-control all` 后，两个进程报告频率仍分别为
   178 与 735 MHz。设置相同不等于实际条件相同，不能只看命令行。
3. 因此使用 `--pair` 在同一进程交替执行 1024,256,256,1024,1024,256 六次调用，
   缩减到 10 个关键指标，每次 3 passes。三次/配置的频率均接近 735 MHz。
   下表是这些调用各指标的中位数（不同于前面的无 profiler 图计时）。

| 指标 | 512 线程 | 256 线程 |
|---|---:|---:|
| SM 频率 MHz | 734.807 | 734.803 |
| profiler 单次时长 μs | 203.104 | 253.920 |
| 执行的 warp 指令数 | 3,062,784 | 2,440,192 |
| 实际 occupancy % | 86.774 | 84.965 |
| DRAM 读取 MB（十进制） | 8.809 | 13.207 |
| DRAM 写入 MB（十进制） | 8.005 | 8.048 |
| 每 scheduler 每活跃周期发射数 | 0.33 | 0.21 |
| 每 scheduler 每活跃周期 eligible warp | 0.912 | 0.380 |
| long scoreboard / issue-active 比值 | 18.298 | 36.371 |
| barrier / issue-active 比值 | 5.320 | 6.529 |

全部数据和 `.ncu-rep` 在 `profiling_results/`。`paired.csv` 第二行是单位，
后面六行才是结果。stall 指标原名为
`smsp__average_warps_issue_stalled_*_per_issue_active.ratio`，它们按 issue-active
归一化，不是 kernel 时间百分比，更不能相加当作总耗时。

### 证据支持的解释

- 所有采集中的执行指令数都一致地减少 20.33%。这支持“较小 block 减少总体
  warp 指令工作”的判断，但不能把全部减少量归因于 shuffle；包括其他控制工作。
- 在频率接近的冷缓存单次 launch 条件下，256 线程 DRAM 读取多约 49.9%，
  实际 occupancy 略低，eligible warp 与发射率更低，long-scoreboard 等待比值更高。
  因而数据支持“减少指令未抵消更差的访存/等待表现”，此时 256 线程慢约 25.0%。
- DRAM 读取数是实际采集值；不能仅凭它确定新增读取来自 input 第二遍、weight
  或哪一级缓存。精确定位还需按内存指令归因或专门的数据复用对照实验。
- 第一轮缓存命中率方向也不支持“256 的缓存一定更好”，但因频率与 replay 条件
  不匹配，不把该轮命中率作精确因果解释。
- 无 profiler 的原图计时曾两次测到约 1.246×，但后续带计时外遥测的图复测变为
  986.522 对 960.070 μs，仅 1.028×。计时外时钟快照为约 210–300 MHz。
  这说明结果对设备状态/执行条件敏感；快照不是 kernel 执行期间的连续时钟记录，
  无法证明是温度、功耗还是某一项驱动策略导致了全部漂移。
- 不应把 profiler 的 203/254 μs 与图基线 314/252 μs 直接计算交叉加速比。

最终学习结论：这个实验已经证明**更少指令不保证更快、理论 occupancy 不决定
实际表现**。它没有证明一个跨条件稳定的 256 线程优化，也没有严格解释此前全部
1.246× 加速。避免据此修改生产默认策略。

### 重现同进程关键指标采集

```bash
/tmp/nsight-compute/opt/nvidia/nsight-compute/2025.3.1/ncu \
  --profile-from-start off --kernel-name regex:RMSNormKernel --launch-count 6 \
  --metrics gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second,smsp__inst_executed.sum,sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__bytes_write.sum,smsp__issue_active.avg.per_cycle_active,smsp__warps_eligible.avg.per_cycle_active,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio \
  --cache-control all --clock-control base \
  --export /tmp/rmsnorm-profiling/paired-new \
  /root/Infer/.venv/bin/python examples/rmsnorm_threads/profile_workload.py --pair
```

下面保留第一次采集受阻时的记录和一般判读指南，历史状态不代表当前状态。

## 首次尝试的历史记录（权限已解除）

目标：解释 M=1024、D=4096 时，512 线程与 256 线程的性能差异。

- 无 profiler 的 CUDA Graph 计时和正确性校验已完成。
- Nsight Compute 2025.3.1 已从 NVIDIA 官方包解压到 `/tmp/nsight-compute`。
- 已尝试采集 512 线程配置，驱动返回 `ERR_NVGPUCTRPERM`。
- **目前没有成功采集实际 occupancy、stall、缓存命中率或硬件指令计数。**
- 尚未用 profiler 验证根因；不能把下面的源码推导写成 profiler 结论。

## 已有证据

RTX 2050，16 SM，驱动 596.36，CUDA 13.0，WSL。

| 项目 | 原策略（cap=1024） | 实验策略（cap=256） |
|---|---:|---:|
| block 线程数 | 512 | 256 |
| warp/block | 16 | 8 |
| rounds | 1 | 2 |
| 寄存器/线程 | 40 | 40 |
| 动态 shared memory/block | 64 B | 32 B |
| 理论驻留 block/SM | 3 | 6 |
| 理论驻留 warp/SM | 48 | 48 |
| 本次独立计时中位数 | 314.491 μs | 252.431 μs |

加速比为 1.246×。原始 7 轮数据保存在 `profiling_results/timing.json`。
计时重复使用相同 buffer，不主动清 L2；两种配置均通过 FP32 reference 校验。
设备没有锁频，第一轮原策略较快，保留所有 trial 便于判断波动。

## 从源码可以推导什么

本次两种配置使用同一个 kernel 二进制，不能解释为编译器给 256 线程版本
生成了不同的寄存器分配。不同的是运行时 blockDim 和由此得到的 rounds。

对于 D=4096，每行始终执行同样数量的有效逐元素平方和与输出运算，
源码层面仍然读取 input 两遍、weight 一遍，写 output 一遍。
逻辑全局访问量均为 `4 * M * D * sizeof(half) = 32 MiB`。
**这不是实测 DRAM 流量**：weight 跨行复用、缓存以及访存事务会改变物理流量。

第一层归约，每个 warp 执行 5 次 shuffle；第二层固定一个 warp 执行 5 次：

| 源码层面每行的 warp 级 shuffle 调用次数 | 512 线程 | 256 线程 |
|---|---:|---:|
| 第一层 | 16×5=80 | 8×5=40 |
| 第二层 | 5 | 5 |
| 合计 | 85 | 45 |

这个计数是控制流推导，不是实测硬件指令计数，更不能直接转换成耗时。
总 block 数仍为 1024，每 block 仍有两处 barrier；只是参与的 warp 数不同。
256 线程版本的遍历循环轮数翻倍，可能抵消部分控制开销收益。

合理假设：较少的 warp 级归约工作、更细的 block 粒度及执行时的访存行为
共同影响了延迟。现有证据不能确定各项贡献，更不能证明瓶颈就是某一种 stall。

## 解锁 WSL 的硬件计数器

实测错误见 `profiling_results/cap1024.log`。在 WSL 内用 root 运行 ncu 仍报错。

在 Windows NVIDIA 控制面板中，以管理员权限打开：

1. 桌面 → 启用开发者设置。
2. 开发者 → 管理 GPU 性能计数器。
3. 允许所有用户访问 GPU 性能计数器，应用。

部分新版 NVIDIA App 也在 System → Advanced → Developer 中提供对应设置。
具体以本机界面为准。

官方说明：
- https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/system-requirements.html
- https://developer.nvidia.com/ERR_NVGPUCTRPERM

## 复现步骤

先做不带 profiler 的计时；不要在 ncu 下运行 `--benchmark` 来得出加速比：

```bash
/root/Infer/.venv/bin/python examples/rmsnorm_threads/profile_workload.py \
  --benchmark --output /tmp/rmsnorm-profiling/timing.json
```

开放权限后，分别将下方 `CAP` 替换为 1024 和 256。每次只捕获一个已预热的
目标调用。验证、初始化及预热均在 `cudaProfilerStart` 之前：

```bash
/tmp/nsight-compute/opt/nvidia/nsight-compute/2025.3.1/ncu \
  --profile-from-start off --kernel-name regex:RMSNormKernel --launch-count 1 \
  --section SpeedOfLight --section Occupancy --section SchedulerStats \
  --section WarpStateStats --section MemoryWorkloadAnalysis --section InstructionStats \
  --cache-control none --clock-control none \
  --export /tmp/rmsnorm-profiling/capCAP \
  --log-file /tmp/rmsnorm-profiling/capCAP.log \
  /root/Infer/.venv/bin/python examples/rmsnorm_threads/profile_workload.py --cap CAP
```

已有同名报告时请使用新输出名称。采集成功须同时确认 ncu 退出码、日志和报告；
Python 打印“Target execution completed”只表示 kernel 已执行，不代表计数器采集成功。

读取报告：

```bash
/tmp/nsight-compute/opt/nvidia/nsight-compute/2025.3.1/ncu \
  --import /tmp/rmsnorm-profiling/cap1024.ncu-rep --page details
```

## 获取报告后怎样判读

| 章节 | 要回答的问题 | 不应直接下的结论 |
|---|---|---|
| Occupancy | 实际活跃 warp 是否接近同为 100% 的理论值？ | 高 occupancy 一定快 |
| SchedulerStats | 每周期就绪/发射 warp 是否变化？ | 活跃 warp 都能立即执行 |
| WarpStateStats | long scoreboard、barrier 等等待有何变化？ | 某类 stall 占比就是耗时占比 |
| MemoryWorkloadAnalysis | DRAM/L2/L1 流量、事务与命中是否变化？ | 源码 load 字节数等于 DRAM 字节数 |
| InstructionStats | 较少 warp 的归约是否体现在指令计数上？ | 指令减少多少，时间就减少多少 |
| SpeedOfLight | 吞吐与资源峰值的关系如何？ | profiler 时长就是前面的无 profiler 时长 |

`--cache-control none` 接近原实验不主动清缓存的策略，但多次 replay 之间缓存状态
可能不同，ncu 也会警告潜在不一致。若结果异常，需要以 `--cache-control all`
另做可复现的冷缓存对照；不要把两种缓存策略的指标混在一起比较。
profiler 捕获的是单次普通 launch，基线是图内重复调用，二者执行环境也不同。

参考：https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
