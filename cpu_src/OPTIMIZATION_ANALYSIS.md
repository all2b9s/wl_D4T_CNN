# C++ 推理性能审计与优化路线

> 历史审计快照：下文的“当前/原版/尚未实现”指 job 3418136 时的代码。
> 后续实现已加入 F4 Winograd、向量数学、寄存器内融合 epilogue、首层 skip 融合、worker 缓冲区复用、
> 持久线程池、fixW 直接 pooling，以及 AttnW token 累加和 QK/AV 向量化。
> 新版状态与验收请看 [VALIDATION.md](VALIDATION.md)，不要用下文的历史测速
> 代表新版。row-streaming 属于原方案并行效率不达标时的备选，仍未启用。

x86 补充（2026-10-20）：F4/F2 和寄存器内融合 epilogue/首层 skip 已移植到
AVX2/AVX512；本次未编译、未验证、未测速，用户将在 Delta 验证。
以下 Grace 测速不能作为 x86 性能数据。

日期：2026-10-19。模型：真实训练的 **fixW s4** 和 **AttnW** `last.pth`；网络全程 float32，T=2，64×64，5 层，32 通道。结论来自代码、分段计时、`perf`、汇编和隔离构建实验。

**主要结论：当前 C++ 已有高效的双数卷积，但原方案假设的向量 GELU、融合 epilogue、工作区复用尚未实现。原版约三分之一时间消耗在 GELU，约八分之一在 LayerNorm，不能按 FLOP 占比估计它们的时间。**

本次只在独立副本中实验，没有替换正式 `_native` 或修改已有 `src/`。向量 GELU + 向量 LayerNorm + 消除残差复制，将相同 512 个源的 72 核吞吐提升 **43.6% / 41.6%**。再使用整批，s4 达到 **514.7 gal/s**，AttnW 达到 **477.5 gal/s**。后者距离 500 下限仍差 4.5%，需要再提高约 4.7% 吞吐。

## 1. 精度口径与实验范围

采用用户指定的标准，对匹配的 N 个源分别计算：

```text
b_e[i]   = mean_over_sources(CPU.e[:,i]   - reference.e[:,i])
b_R[i,j] = mean_over_sources(CPU.R[:,i,j] - reference.R[:,i,j])
通过条件：每个分量 abs(b) < 1e-4，且所有输出有限。
```

这是有符号差的平均；不同源可相互抵消，不同 e/R 分量不能相互抵消。不是 mean(abs(error))。逐点离群、分位数、位置导数继续保留为诊断。float64 golden 验证保留原来的严格逐点门槛。

对之前 512 个真实源重新评估：

| 比较 | s4 最大 e 分量均值偏差 | s4 最大 R 分量均值偏差 | AttnW 最大 e 分量均值偏差 | AttnW 最大 R 分量均值偏差 |
|---|---:|---:|---:|---:|
| 原版 C++ − strict FP32 GPU | 4.27e-9 | 1.46e-7 | 1.57e-8 | 1.01e-6 |
| 本次 combined − strict FP32 GPU | 4.33e-9 | 1.22e-7 | 1.85e-8 | 9.70e-7 |

**两模型均通过。** 原版与 Torch CPU 也都通过。s4 与默认 TF32 GPU 的 R11 均值偏差为 **+1.34e-4**，仍不通过；AttnW 与默认 TF32 的最大 R 均值偏差为 3.55e-5，通过。数值参考应继续使用关闭 TF32 的 GPU。

所有本次表列实验在 512 个源上都通过 e/R 均值标准。combined 另外检查了 64 个源的 T=4：最大 R 均值偏差分别为 1.30e-7 / 9.40e-7；同一实验版 T=2 与 T=4 的 e/R 逐位一致。位置导数最大均值偏差为 4.20e-8 / 1.46e-7，仅作诊断。

这证明这些样本上的数值一致性，不等同于完成曝光级 m/c 科学校准。整批测试重复已有源，也不增加独立样本数。

实现位置：[validation.py](validation.py)、[CLI](__main__.py)、[均值标准测试](tests/test_mean_acceptance.py)、[旧结果复评](benchmarks/reassess_pair.py)。三个验收规则测试通过。

## 2. 测量方法和可比性

- interactive job **3418136**，**gh051**，独占一个 72 核 CPU 域，CPU affinity 72–143；机器其他域可能有别的作业。本次 GPU 结果复用之前逐源匹配的 strict FP32 输出。
- GCC 14.2，Grace/Neoverse V2；基础编译参数 `-O3 -std=c++17 -fPIC -fvisibility=hidden -pthread`。
- 同一组 512 个 held-out 真实 q-stamps，来自 exposure 940–943；两模型复用完全相同输入。模型及源身份继承之前的 provenance。
- 单核测 32 个源，72 核测 512 个源；先 warmup，然后三次计时取中位数，全部原始耗时保留。包含 Python 模型调用及输入/输出处理，不含导出、读文件、检测、q-stamp 生成。
- 独立进程加载各实验 `.so`；构建完成后依次计时，无并发编译污染基准。另有分段计时版本，不用它替代无插桩吞吐结果。
- 整批实验为 512/576/1152/2304 个源，循环已有 512 个源，考察 batch 尾部和稳态吞吐。
- `perf stat` 是进程范围：包含加载、warmup 和推理；单核计时主体 64 个源，多核主体 1152 个源。计数器不是某一个算子的精确归因，不能当作 DRAM 带宽测量。

之前 gh009 原版 344.6 / 321.4 gal/s，本次 gh051 原版 327.3 / 306.3 gal/s，相差约 5%。所有优化比值都用本次同节点基线，不能混用两节点结果放大收益。每组三次计时不能消除跨实验时段的频率/共享资源变化；例如重复的 batch=512 测试有约 1–2% 漂移。

## 3. 原版时间究竟花在哪里

单核分段计时，单位 ms/gal；相邻阶段独立计时，`total` 包含整个原生单源推理。

| 阶段 | s4 原版 | 占比 | AttnW 原版 | 占比 | s4 combined | AttnW combined |
|---|---:|---:|---:|---:|---:|---:|
| 隐藏层卷积 | 86.84 | 48.0% | 87.25 | 45.4% | 86.96 | 86.55 |
| 第一层卷积 | 1.13 | 0.6% | 1.14 | 0.6% | 1.13 | 1.13 |
| GELU | 58.10 | 32.1% | 57.52 | 29.9% | 10.62 | 10.71 |
| Channel LayerNorm | 21.90 | 12.1% | 22.00 | 11.4% | 8.33 | 8.42 |
| residual / skip | 7.86 | 4.3% | 6.95 | 3.6% | 5.63 | 5.16 |
| attention 分支整体 | — | — | 13.10 | 6.8% | — | 13.03 |
| orbit 累加 | 1.87 | 1.0% | 1.88 | 1.0% | 1.92 | 1.88 |
| 其余：分配、输入、pool、head 等 | 3.21 | 1.8% | 3.41 | 1.8% | 2.93 | 2.27 |
| **总计** | **180.90** | | **192.15** | | **117.53** | **129.17** |

attention 内部也有 LN/GELU，但本表计入 attention 总计，避免重复统计。插桩版与对应无插桩版吞吐差约百分之一以内。

![实测阶段耗时](runs/profile_3418136/stage_breakdown.png)

`perf record` 独立采样也支持这个分解：s4 的 NEON 卷积占 49.87%，`expf` 占 19.27%，`erff` 占 10.29%，LayerNorm 占 11.68%；AttnW attention 采样占 6.39%。这不是只看源代码得到的猜测。

### 原方案中需要修正的判断

1. [原计划](../CPU_INFERENCE_PLAN.md) 用大致 FLOP 数推断 conv≈95%、LN≈2%、GELU≈2%。标量 `erff`/`expf` 的函数调用、分支和串行指令不能等价成几十个廉价 FMA；实际 GELU≈30–32%。
2. 原计划明确要求 vector erf、融合 LN/GELU/residual、两块 buffer 复用。现有 `block()` 仍是独立的卷积→整幅 LN→整幅 GELU→残差，并且每层分配输出、复制 skip。性能预估的前提没有全部实现。
3. attention 的 FLOP 少于 1%，不代表耗时少于 1%；实际占 6.8%，主干加速后升至约 10%。
4. 7–8 gal/s × 72 = 504–576 gal/s，隐含约 100% 并行效率。若按方案的最低合格线 70%，只能得到 353–403 gal/s。单核目标、并行效率门槛和绝对吞吐目标必须分别记录。
5. P1 中“单核 trunk ≥20 gal/s”与同一文档的 7–8 gal/s 估计不一致。若指当前 T=2、8 orbit、直接卷积，20 gal/s 要求约 143 GFLOP/s，已超过下面按最高标称频率计算的单核上界；除非改变算术工作量或测试定义，否则不能用该值做门槛。

### 卷积本身是否很差

考虑边缘少做的卷积，每层有效空间邻域数为 `(3×64−2)^2 = 36100`。CNN 的直接卷积工作量是：

```text
2 FLOP/FMA × 3 dual components × 8 orbits × 36100 × (32 + 4×32²)
= 7.153 GFLOP/gal
```

隐藏层 7.098 GFLOP / 86.84 ms ≈ **81.7 GFLOP/s**。Arm V2 指令表给出的 128-bit FP32 FMLA 吞吐为 4 条/周期，即理论 32 FLOP/周期；用 `lscpu` 的最高 3.456 GHz 得到 110.6 GFLOP/s 的上界，当前隐藏层已约为其 74%。这只是算术峰值参照，3.456 GHz 不是本次验证过的持续频率。[Arm V2 优化指南](https://documentation-service.arm.com/static/668bc0a369e89f01e39c4668)

因此最初应先优化卷积之外的 51% 时间。即使让原版卷积快 20%，整模型也只有约 9% 加速；向量化 GELU 单项实测就带来 26–29% 的 72 核吞吐提升。

## 4. 已做的优化实验：有效与无效都保留

单位 gal/s，均为同节点、相同 512 个源（单核列用 32 个源）。

| 隔离版本 | s4 单核 | s4 72 核 | AttnW 单核 | AttnW 72 核 | 解释 |
|---|---:|---:|---:|---:|---|
| baseline | 5.545 | 327.307 | 5.169 | 306.266 | 正式版二进制 |
| 相同参数重编译 | 5.530 | 326.510 | 5.242 | 305.196 | 基本一致，排除旧二进制是主要原因 |
| 删除隐藏层 skip 深拷贝 | 5.630 | 337.381 | 5.307 | 315.952 | 约 +3%，输出逐位一致 |
| `-mcpu=neoverse-v2 -fno-math-errno` | 4.423 | 269.836 | 4.236 | 257.716 | 退化，见寄存器分析 |
| V2 + `-ffast-math` | 4.732 | 284.537 | 4.551 | 273.678 | 比上一行好，但仍差于 baseline |
| T=2 tile 从 32 通道改成 16 | 4.345 | 262.334 | 4.178 | 250.393 | 重读输入/循环开销增加；不是越小越好 |
| 2 pixels × 16 channels tile | 4.403 | 263.667 | 4.212 | 249.564 | 此具体实现无收益，不能推广成所有空间分块无效 |
| 向量 GELU | 7.536 | 422.687 | 6.936 | 386.421 | 主要收益来源 |
| 向量 LayerNorm | 5.999 | 347.870 | 5.639 | 326.008 | 约 +6–7% 多核收益 |
| **GELU + LN + no-copy** | **8.523** | **469.926** | **7.747** | **433.728** | **+43.6% / +41.6% 多核收益** |
| V2 + `-fno-schedule-insns` | 5.567 | 330.252 | 5.251 | 309.481 | 基本恢复 baseline，支持调度导致退化的判断 |

![各实验吞吐](runs/profile_3418136/variant_throughput.png)

### 为什么面向 V2 编译反而更慢

`kernels.cpp` 的 T=2、32 输出通道 tile 有 `3×8=24` 个 NEON 累加寄存器。基线汇编内循环每个输入通道做 24 次向量 FMLA，权重逐块装入，累加器不落栈。

V2 编译器调度提前加载更多权重，造成寄存器压力；保存的 `tuned_kernels.s` 内循环出现：

```asm
ldr q21, [sp, 192]
str q21, [sp, 192]
ldp q30, q26, [sp, 160]
stp q30, q26, [sp, 160]
```

每个输入通道迭代额外读写 3 个 128-bit 累加器，即 96 bytes 栈流量。插桩隐藏层卷积从 **86.84 ms 增至 130.13 ms**。禁用第一阶段指令调度后总吞吐基本恢复。应逐 kernel 检查汇编和性能，不能直接全局打开 `-mcpu=native`、更激进展开或 fast-math。

这里 fast-math 行同时带 V2 flags，不能据此断言 generic `-ffast-math` 必然退化。GCC 的 fast-math 还包含 finite-only、重结合等假设；本项目有显式 `isfinite` 输入检查，数学近似应限制在具体 kernel 内。[GCC 优化选项](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)

### 向量 GELU 和 LN 具体改了什么

- GELU：NEON 四 lane，使用 erf 的五系数近似和系统 `libmvec` 向量 exp；正向和导数共享 `exp(-x²/2)`。双数仍按目标 erf-GELU 的 `Φ(x)+xφ(x)` 计算，不改变训练网络的激活类型。这里 Φ 是数值近似，导数也因此有近似误差；它不是对该近似多项式再自动求导。
- GELU 当前证据是两 checkpoint、512 个源、T=2 及额外 T=4 检查，**没有宣称完整输入域的严格误差界**。生产化应补极端输入/近似函数和有限差分验证。tanh、double、标量尾部继续原路径。
- LN：C=32 显式 NEON channel reduction、方差、tangent mean/covariance 和输出；复用中心化向量。改变 float32 加法顺序，但上述 e/R 均值通过。
- combined 使用基础 `-O3`，没有全局 fast-math。实验引入了 `libmvec` 向量 ABI 依赖，正式实现要处理系统可用性和 portable fallback。

## 5. 多核为什么还损失性能

### batch 尾部

512/72 = 7 个完整波次 + 最后 8 个任务。若每个源成本完全相同，这一批的调度利用率上界是 `512/(72×8)=88.89%`。它不能被解释成内存带宽损失。当前是动态调度，实际还受源成本差异、启动和频率影响。

| 模型 / 版本 | batch=512 | 576 | 1152 | 2304 |
|---|---:|---:|---:|---:|
| s4 baseline | 330.7 | 356.1 | 355.4 | 360.2 |
| s4 combined | 475.2 | 504.1 | 507.3 | **514.7** |
| AttnW baseline | 312.0 | 334.4 | 334.0 | 338.1 |
| AttnW combined | 443.4 | 467.5 | 470.6 | **477.5** |

本表是单独的 batch sweep，所以 512 结果与前表略有漂移。对应相同 batch=2304，combined 提升约 **42.9% / 41.2%**；不能把 batch 增益再次算到这些比值里。

![批次大小与吞吐](runs/profile_3418136/batch_throughput.png)

实际 pipeline 默认 `batch_size=64`、`nthreads=1`。将 nthreads 改成 72 而保留 batch=64，永远至少有 8 核拿不到任务；并且每批都创建/销毁线程。生产部署应显式选择 72 的倍数，先测 576、1152，再权衡内存和首批延迟。

### 工作集超出缓存

硬件报告：每核 64 KiB L1D、1 MiB L2；每 72 核域约 114 MiB L3。T=2 的一张 64²×32 特征图含 3 个分量，大小 **1.5 MiB**，已经放不进单核 L2。

现有代码在处理隐藏层时同时持有 scalar/e1/e2 三张 canonical field，再加 x/y/skip-copy 三张特征图，约 **9 MiB/core**，72 核约 648 MiB，远大于共享 L3。T=4 对应约 15 MiB/core。这里是活跃张量的容量估算，不等于全部字节每一步都来自 DRAM。

每个源有 32 次隐藏层 skip-copy，复制有效负载 **48 MiB**（读+写约 96 MiB）；40 个 block 的输出构造还会清零约 **60 MiB**，随后卷积又完全覆盖。分配计时本身不包含所有后续 cache pollution，因此不能只按分配函数的 1–2 ms 判断缓存价值。

### PMU 支持什么结论

下表为 `stall_backend_mem / cycles`，进程范围硬件计数之比，并非可直接从墙钟时间扣除的百分比：

| 设置 | s4 | AttnW |
|---|---:|---:|
| 原版 1 核 | 1.81% | 1.63% |
| combined 1 核 | 3.05% | 2.20% |
| 原版 72 核 | 5.73% | 6.03% |
| combined 72 核 | **10.99%** | **8.78%** |

同时，s4 单核 instructions 从约 1664 亿降到 1200 亿，cycles 从 373 亿降到 245 亿；72 核 IPC 从 4.18 升到 4.40，但内存停顿比重增大。阶段计时也显示 combined 隐藏卷积从单核约 87 ms 增至多核约 102 ms，orbit 累加从约 1.9 ms 增至约 5 ms。

这些证据支持“标量数学开销消除后，缓存/内存访问是下一轮重点”。尚未测内存控制器/DRAM 带宽，**不能声称已饱和 Grace 内存带宽**，也不能把所有单核→多核退化都归因于内存。

## 6. 逐项代码优化清单

下列“潜在收益”是局部范围或优先级依据，未测项目不当作承诺，也不能把各项百分比简单相加。

| 优先级 | 位置 / 问题 | 具体方案 | 证据、收益边界与验证点 |
|---|---|---|---|
| P0 | `model.h::activate` 标量 erf/exp | 将已验证的向量实现整理为独立 fp32 数学 kernel；向量 exp 复用；保留 fallback | 已测最有效；补宽输入域误差、尾部、极端值、T=4；不把 GELU 换成另一种激活 |
| P0 | `layernorm` 串行 channel reduction | C32 NEON 特化，generic/double fallback，后续 AVX 对应实现 | 已测约 22→8 ms；验证不同 channel、eps、方差极小时 tangent |
| P0 | `block` 的 `X skip=x` | 同维度直接读 x；仅第一层生成投影 skip | 已测约 +3%，逐位一致；注意 workspace 改写后仍保留 x 的生命周期 |
| P0 | `api.py` / pipeline batch | 使用 72 倍数；报告 batch、core count、T 和吞吐中位数 | 已测整批约 +6–9%；不能把循环样本称作新增真实数据 |
| P1 | `Tensor` 零初始化、每层分配 | 每 worker workspace；x/y ping-pong；卷积覆盖的 buffer 不预清零；只有累加器清零 | 去除 40 次输出分配/清零及 allocator 压力；设计 explicit overwrite/accumulate 两种接口，避免读未初始化内存 |
| P1 | Conv/LN/GELU/residual 分离 | 每像素/小 tile 的 conv 输出写入对齐 scratch，立刻 LN→GELU→skip，再写最终 y | 从整幅多次遍历变成热数据短生命周期；不要强求 24 conv 累加器与全部 LN 临时值同时驻留寄存器，避免重现 spill |
| P1 | `measure` 最后一层与 orbit 累加分离 | 最后层 epilogue 直接累加最终使用的 canonical/token/pooled 数据 | 降低最终 x 写回和再读；trace 模式保留完整慢路径，验证每个 D4 变换/符号 |
| P1 | fixW 仍构造 scalar/e1/e2 三张场 | Gaussian pool 的 D4 对称性允许各 orbit 输出直接加权 pool，再乘 sign1/sign2 累加 | 不需要完整 scalar/e1/e2，也不需要动态 kernel 的 tangent-zero arrays；显著减小工作集，准确收益待测；保留输出 head 的非线性顺序 |
| P1 | `attention_kernel` QK/AV 访问模式 | head_dim=8 向量化；K/V 合适转置或 packing；AV 改为 p,q,j 并跨 j 累加向量；小型单线程 GEMM 作对照 | 当前全分支 13 ms；combined 后约占 10%，这是 AttnW 特有的下一步目标；必须传播 Q、K、V 和 softmax 的所有 tangent |
| P1 | attention scores 过大 | 按 query 16/32 行分块，每块做 QK→softmax→AV，复用 score scratch | 256²×3×4=768 KiB/head；16 query 可降到 48 KiB，QKV/上下文也更易留在缓存；不改变 softmax 归一化的 key 范围 |
| P1 | AttnW scalar full-resolution 场 | 因 adaptive average pooling 是线性的，最后层按 canonical 坐标直接累加 16×16 scalar tokens | scalar 1.5 MiB→约 96 KiB；e1/e2 仍需等待 object-specific kernel。非整除尺寸的 adaptive pool 可能重叠，必须用原来的区间定义 |
| P2 | `parallel.h` 每批建线程 | 持久线程池 + worker workspace + 每 worker affinity；比较固定块/动态小 chunk | source 级原子调度每约 0.1 s 一次，原子本身不是主要瓶颈；更大价值是缓存和 AnaCal lease 跨批保留。公开 API 并发调用不能共享可变 scratch |
| P2 | 全层特征图超 L2 | 5 层 row-streaming，每层 3-row buffer，边算边做 epilogue | T=2 每层约 72 KiB，5 层约 360 KiB（另有权重/scratch/fields）；层间 halo、残差行生命周期和上下边界要显式调度。先 fusion 再决定是否值得复杂度 |
| P2 | 卷积地址、边界、packing | 内部像素和边框拆开；C32/D3,D5 编译期特化；受控空间 tile；weight layout 与 tile 联动 | 当前两种简单 tile 已失败；每次检查最内层 spill、指令数、单核/72 核。当前权重 `[ky,kx,ic,oc]` 与 C32 tile 已匹配，不能只换布局不换循环 |
| P2 | ISA / 编译器 | kernel 局部 flags，比较 GCC/Clang、LTO/PGO；NEON/SVE 或 AVX 自己测 | V2 flags 负收益已有证据。向量更宽不自动更快；x86 当前只有小输出块、runtime nd，需单独的寄存器分块，不能套用 Grace 数据 |
| P2 | attention 辅助算子 | 预计算固定坐标、adaptive-pool 边界、bilinear 索引/系数、taper；softmax 向量 exp 与 reduction | 归入 13 ms attention 总预算，不能每项再算一遍 13 ms；对低分辨率 logits 和 full-size softmax 分别剖析 |
| P2 | 5×5 Gaussian blur | 经验证 kernel 可分离时用 1×5 + 5×1，25 taps→10 | checkpoint kernel 是已舍入的 2D 值，数学可分离不等于浮点逐位可分离；验证 replicate padding、primal/tangent 和最终均值 |
| P2 | q preparation 与模型之间的数组 | 保留 AnaCal double FFT，在写最终 q-stamp 时可提供 fp32 输出；或同一 source worker 直接接网络 | 当前先输出 5-plane float64，再转 float32；T=2 不需要传 x1/x2 给 CNN。该项影响端到端 pipeline，不在本次纯网络吞吐里 |
| P3 | 小对象、key lookup、head | constructor 缓存 layer 参数指针，减少 `std::map`/字符串和重复 head_width；T2/T4 分享 immutable weights | head 仅约 0.1 ms，normalize 约 0.02 ms，不能解释百毫秒缺口；权重 double→float 转换发生在加载，不是每源热点 |
| P3 | q 检查和输出复制 | 避免已满足 dtype/contiguity 的重复转换；在可信内部接口合并检查 | q 有限性扫描和 e/R 输出很小，不应先删公共入口检查；输出仅几个数，不是主要内存流量 |

### attention 的具体算术与缓存问题

QK 和 AV 各自的 primal + 两条 tangent 需要约 20.97 MFLOP，共约 41.94 MFLOP，另有 projection/FFN/softmax 等。即便 FLOP 规模小，当前 `p,j,q` 的 AV 循环使 V 沿 q 以 96 floats 的跨度读取，归约串行，无法有效利用 contiguous head 通道。

每个 head scores 768 KiB，加上 QKV 288 KiB、tokens/context/LN 各约 96 KiB，活跃数据超过私有 L2。query tiling 比单纯删除几个数学函数更有针对性。双数 softmax 必须保留 `s_dot = s*(z_dot - sum(s*z_dot))`；把 learned pooling kernel 的梯度 detach 会改变 R，不能作为性能优化。

### 端到端还要单独测什么

本次 500 gal/s 目标只针对已准备好的 q-stamps。`api.py::measure_sources` 每批先 q preparation 再网络，每个阶段各建一个线程池。AnaCal 的 `ImageQLease` 使用 `thread_local` 缓存，短命线程使跨批复用失效。因此持久 worker 还能保留 FFT/q preparation 对象，而不仅仅节省 thread start。

建议分别记录 detection、PSF/noise/q preparation、network、catalog write 和 calibration。若做 q+network 同 worker 流水线，先明确哪些对象可共享、哪些必须每线程独立；避免与外层 exposure 并行叠加造成 oversubscription。288 核整节点应按四个 NUMA CPU 域分别初始化/运行，不能直接按本次单域吞吐乘四承诺结果。

## 7. 不应直接采用的“捷径”

- **删除 8 个 D4 orbit 中的大部分计算**：一般训练的卷积核不是 D4-equivariant，同一张卷积输出不能简单旋转后复用为另一个 orbit。变换卷积权重可能改变访存组织，但仍需对应的网络计算，并增加 packed weights。
- **只因 s4 Gaussian 很窄就裁 stamp**：有效感受野、padding、输入归一化及模型训练支持都可能改变。可以做有误差预算的独立实验，不能假定小于 1e-4。
- **改 GELU 为其他激活、减 channel/layer/grid、量化/bfloat16**：属于模型/数值策略变化。C32→C24 的隐藏卷积 FLOP 约变为 56%，但精度收益/损失需重新训练或独立验证，不是现有 checkpoint 的等价实现优化。
- **默认 FTZ/DAZ**：未测 denormal 是当前热点；先统计/计数再选择是否仅对推理线程启用，必须恢复线程状态并验证 tangent 尾部。
- **直接上 im2col/GEMM 或 Winograd**：可作备选对照，im2col 增加 packing/内存，Winograd 改变运算顺序和误差。现有直接卷积已经较高效，需要端到端证据才值得替换。
- **把 shared head 理解成少算一次 head**：e1/e2 输入不同且 head 非线性，分享权重不等于可复用输出；该部分本身也不是热点。

## 8. 建议下一轮实现顺序与收益判断

1. **将已验证的三项改动正式化**：vector GELU、C32 vector LN、skip no-copy，保留 scalar/double/T4/非 C32 路径；整理 vector math 的平台依赖和误差检查。目标复现相同 512 batch 上约 470 / 434 gal/s，而不是只看最有利 batch。
2. **workspace + epilogue fusion**：先 per-pixel 小 scratch，不追求所有中间量同时在寄存器中；保持第一层 skip 和 LN tangent 计算顺序。加 full pipeline 的 persistent workers/batch sweep。
3. **分模型优化输出路径**：s4 直接 orbit weighted pooling；AttnW scalar token 直接累加 + attention query tiling/AV 向量化。保留 trace 的完整张量路径用于定位问题。
4. **再判断卷积和 row-streaming 的投入**：用优化后的分段时间和 PMU 决定，避免继续优化已不占主要比例的开销。

按 combined 单核 profile，AttnW attention 约 13/129 ms。若该分支快 2 倍，单核理论加速约 5.3%；若其他条件保持不变，batch=2304 的 477.5 可能接近 503 gal/s。**这是 Amdahl 估算，不是已实现结果。** 72 核还有内存停顿，实际要测。

同理，combined 的卷积约占 s4 总时间 75%、AttnW 68%；到这个阶段，卷积再快 20% 的整模型收益约 14% / 13%，才值得重排优先级。各百分比随优化状态改变，不能沿用最初的分解。

当前已经证明 s4 的 500 下限可达。AttnW 的 500 有明确、有限的剩余差距；580 和整节点 2000 还没有足够实测证据。未来报告应固定模型、T、batch、核数、计时范围和实际均值验收，分别给出单核、并行效率与最终吞吐。

## 9. 文件、原始证据与复现

实验目录：[runs/profile_3418136/](runs/profile_3418136/)。该目录被 Git 忽略，分享结果时需要一并保留。

- [summary.json](runs/profile_3418136/summary.json)、[全部实验吞吐表](runs/profile_3418136/SUMMARY.md)：可机读结果。
- `baseline/combined/vector_*.json` 等：每次原始计时、完整分量均值、max/rms；`*_predictions.npz`：512 源逐源预测。
- `profile.json`、`combined_profile.json`：分阶段单核/72 核结果；计时是每 worker 的 elapsed time，多核时可包含调度和资源等待。
- `perf*_*.csv`、`perf_*_report.txt`、`*.perf.data`：硬件计数与符号采样；`rebuild_kernels.s`、`tuned_kernels.s`：寄存器调度证据。
- `combined_T4_checks.json`：额外 T=4 验证；旧结果复评在 [mean_acceptance.json](runs/paired_3417691/mean_acceptance.json)。
- `experimental_builds.tar.gz` + `.sha256`：原版快照、全部实验源码和 `.so`、运行脚本；各 `*_build.json` 和 `build_*.log` 记录编译选项与日志。向量 GELU 的额外 `-lmvec` 链接参数见构建脚本。
- `lscpu.txt`、`numa.txt`、`allocation.txt`、`accounting.txt`：环境及作业记录。allocation 已释放。

可复用脚本：[profile_native.py](benchmarks/profile_native.py)、[perf_native.py](benchmarks/perf_native.py)、[summarize_profile.py](benchmarks/summarize_profile.py)、[check_profile_t4.py](benchmarks/check_profile_t4.py)。`finish_profile.sh` 保留本次现场的最终收集命令。

在已分配的同类计算节点上，可从归档基线重跑；使用新的输出目录，避免覆盖原始结果：

```bash
repo=/projects/bdsp/wenyinli/codes/single_galaxies
python_bin=/u/wenyinli/.conda/envs/anacal_clone/bin/python
scratch=/tmp/d4cnn_recheck_${SLURM_JOB_ID}
mkdir -p "$scratch"
tar -xzf "$repo/cpu_src/runs/profile_3418136/experimental_builds.tar.gz" -C "$scratch"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$scratch" MPLCONFIGDIR="$scratch/matplotlib"
"$python_bin" -B "$repo/cpu_src/benchmarks/profile_native.py" \
  --base "$scratch/base" --work "$scratch/rebuild" \
  --data "$repo/cpu_src/runs/paired_3417691" \
  --output "$repo/cpu_src/runs/profile_recheck_${SLURM_JOB_ID}"
```

该脚本会重建全部实验，默认跑 baseline/tuned 的 batch sweep；combined 的 batch sweep 用 `--worker --balanced --variant combined_balanced --base "$scratch/rebuild/combined"` 再运行，其他路径参数相同。汇总脚本读取三种 batch sweep 后生成 PNG/PDF。不要在登录节点执行推理、构建或批量绘图。
