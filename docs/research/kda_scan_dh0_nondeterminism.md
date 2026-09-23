# A5K-03：`scan_fused` 的 seed 槽位在 BHV 边界提前复用

当前证据定位到一个跨核所有权缺口：奇数个 chunk 结束后，`seed_ub`
重新从槽 0 开始，但 `CvMutex(3, depth=2)` 的信用仍连续流转。下一组
BHV 可以用上一组槽 1 归还的信用覆盖槽 0，而上一组最终 state 更新还在读槽 0。
事件模型指出了这一对无序访问；历史真机错误值与“读到了下一组最后一个 chunk
的 seed”这一预先记录的预测高度吻合。

两张 CANN 9.1 健康卡和一张 CANN 9.2 健康卡的共同恒等 VF 时序干预
均支持这一因果解释。原始入口的 50 次新重放没有自然复现不确定性；CANN 9.2
的无扰动注释副本仅首次 anchor 出错。不能据此声称问题消失或已经修复。
完整私有归档已恢复并逐文件校验；实际张量与回执的对应核验也已完成。
本任务只定位：所有实验都在 `kda_scan_diag`，上游、生产 kernel 和公开调度未改。

## 来源与范围

- 项目起点：`5a085751595fe6223bed785ac36b6d839216d125`，任务分支 `task/A5K-03`。
- 主机检查基线快进至 `9b8c7e8e8433d3cb00d6890a48b3d4bc2207c01a`；仅 PM 元数据变化，
  423 个原始真机源码文件 SHA256 全部不变。
- library：`90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5`。
- upstream kernels：`b3b3f9c16df7c4626ed3c081032a1be5a753d0b1`。
- 原始 `projects/a5/kda_bwd/kernels/scan_fused.py` 的 SHA256：
  `deca5ef18267904baa6ede10b17d41684413a229983f76e48bb10463092fdb03`。
- 固定输入从 FMT-02 的种子清单在运行时重新生成；每次核对公共输入、八个 scan 输入、
  实际返回的四个输出。历史张量只用于历史故障分析，不替代本任务真机运行。

数值参考使用 Torch CPU FP32；没有新增或放宽预算，没有性能结论。
每个 `block_dim` 使用独立进程。生产依赖全部 50 个 vendor（包括九个 backward）
以及该进程使用的实验 vendor，均在首次自定义 kernel 执行前完成编译。

## 具体无序访问

以下行号均指上述原始源码。每组 cube 和两个 AIV 负责 BHV 序列；
两个 AIV 各处理 64 行 state。`B2/HV4/bd4` 使每组 cube 连续处理两个 BHV。

| 位置 | 操作 | 所有权与存储 |
| --- | --- | --- |
| 212 | 声明 `seed_mutex` | `CvMutex(3, depth=2)`，FIX 发布、V 归还 |
| 241 | 声明 `seed_ub` | 两个 UB 槽，各 `64×128` FP32，每个 AIV 独立 |
| 276–278 | 每个 BHV 的第一次 seed 发布 | `lock → FIX 写 seed_ub[0] → ready` |
| 334–336 | 下一个 chunk 的 seed 预取 | 索引 `rev_c + 1`，按两个槽取模 |
| 353–360 | 消费 seed、更新 state、归还信用 | 最终 chunk 调用 `update_dstate_final_vf` |
| 160–161 | 最终 VF 读取 seed 的两半行 | 消费者的实际读访问 |
| 362–363 | 写出 dh0，执行本侧 `bar_all` | 不能替代跨 AIC/AIV 的所有权归还 |

以 C=3 为例，物理槽序列是 `0,1,0 | 0,1,0`。两信用协议连续推进，
第四次生产可取得第二次消费归还的信用。第二次消费读的是槽 1，第四次生产却
写槽 0，因此第三次消费与第四次生产之间没有必需的顺序。

精确无序对是：上一 BHV 的最终 VF（358 行，实际 seed load 在 160–161 行）
与下一 BHV 的初始 `l0c_to_ub`（277 行）。模型在两个 AIV 上分别报告 UB WAR。
计数平衡检查通过并不能证明存储复用安全。

最终 `state_h_ub → dh0` 本身生成了 V 释放、MTE3 获取同一个本地缓冲 mutex 的代码。
因此，本调查没有把“Python 中没有显式 barrier”当作缺同步的证据。
本地自动同步的 32-ID 命名空间分别属于各 AIC/AIV，不能建立这里缺失的跨侧边。

六条跨侧握手均逐一核对：state bridge、dv bridge、dvdelta、corr、dv0 的深度为 1；
seed 为深度 2。此次缺陷涉及 seed 的跨 BHV 槽序，其他握手和 32-ID 上限均未修改。

## 预测、单变量干预与负对照

执行模型之前，任务记录已提出 H1：奇数 C 且同核重复 BHV 时出现上述 WAR；
偶数 C 或单个 BHV 不出现这一处边界竞争；只有最终 dh0 受影响。

实验 `ring` 只把四处 seed UB 索引改为连续相位
`(bhv - bhv_begin) * C + rev_c`，并相应处理首次发布和预取。
缓冲数量、信用、计算、cast、预取位置和 drain 均不变。
`negative` 只增加无关注释。三份实验副本采用不同入口名以便在同一进程预编译，
规范化入口名后，negative 与 baseline 的 AST 和生成的 cube/vector CCE 源码相同。

| 输入与核组数 | 原始模型 WAR 数 | 连续槽号模型 WAR 数 |
| --- | ---: | ---: |
| B2/HV4/C3，bd1 | 14 | 0 |
| B2/HV4/C3，bd2 | 12 | 0 |
| B2/HV4/C3，bd3 | 10 | 0 |
| B2/HV4/C3，bd4 | 8 | 0 |
| B2/HV4/C1，bd4 | 8 | 0 |
| B2/HV4/C2，bd4 | 0 | 0 |
| B1/HV2/C3，bd1 | 2 | 0 |

19 份原始/槽位干预模型运行均完成，没有 deadlock，事件计数检查为空。
完整真机时序实验后，另执行三份 SPINS64 缩小模型，baseline/ring/negative
分别仍为 2/0/2 个同源 WAR；总计 22 份，未出现额外 hazard。
注释负对照在缩小形状和原始失败形状仍分别报告 2 和 8 处 WAR。
B1/HV1/C3/bd1 的原始模型没有这处竞争。
缩小模型保留完整 tile、两个 AIV、奇数槽回绕和每核两个 BHV；没有用它替代完整真机 workload。
模型给出的周期恰好不重叠也不构成安全保证：这里的判据是事件建立的先后关系。

## 历史真机字节证据与错误形态

重新 SHA 核验 FMT-02 的原始捕获后，12 次历史直接重放中的 11 次异常
全部位于零起始 `B=1, HV=2`，即全局 BHV6、第四个 cube 处理的前一个 BHV。
每次有 1,917–8,184 个 BF16 元素偏离稳定卡；`dAqk/dh/dv` 保持逐位一致。

在拟合任何数值之前，预测错误 seed 来自下一 BHV7 的最后一个 chunk。
CPU FP32 计算保持原来的 scale、当前 corr、decay 和先前 state，只替换该 seed：

```text
bf16(seed_next_head_last / sqrt(128) + prior_state * exp(g_last * ln(2)) - current_corr)
```

该预测在共计 52,001 个异常元素中逐位解释 51,990 个，每个异常 trial 仅余一个元素。
剩余误差的相对 L2 为 `5.95e-6` 到 `1.25e-5`，符合此次 CPU/设备计算的舍入差异量级；
这是观测值，不是另设精度预算。竞争假设“下一 head 的第一个 chunk”及
“同一 head 的中间 chunk”每次仅匹配 1–7 个元素。
正常 CPU 物理参考与稳定卡 dh0 的相对 L2 是 `4.380261816550046e-5`。
可重放分析脚本为 `kda_scan_diag/counterfactual.py`，不执行 NPU kernel。

为什么只有 dh0：最后一次 state 更新之后只发布 dh0。
`dAqk`、当前 `dh` 和 `dv` 已经产生；下一 BHV 又从自己的 dht 初始化 state。
竞争因此可污染当前 BHV 的部分最终 state 行，而不传播到这三个输出。
生产 autograd 在 `initial_state` 需要梯度时返回此 dh0，故不是不可见的临时值。

## 新真机运行与结论边界

CANN 9.1.0-beta.1、compiler `20260509_173000235`、Python 3.12.14、
Torch 2.12.0+cu130、torch_npu 2.12.0 环境中：

- 两张健康对照卡各完成完整 Kimi T4096 训练和原始 scan 的独立 anchor + 50 次重放。
- 不填输出的首轮：两卡各 6 个形状 × 3 份副本 × 50 次，共 1,800 次。
- 按历史方式填 NaN 的重放：两卡各独立运行 bd1–4；第三张卡补 bd4。
  每个进程同样是 18 个 family、每 family 50 次，共 8,100 次。
- 上述 9,900 次实验副本重放均有限、无跨次偏差，三份副本的 anchor 逐位一致。
  每个进程完整 workload 通过，编译回执包含全部 50+3 个 vendor，运行和 guard 退出均为 0。

每个 family 的零偏差是 0/50，条件 iid 假设下单侧 95% Clopper–Pearson 上界为
`0.05815507911697225`。不合并不同卡、bd、形状或时序为一个独立同分布总体。
这些负结果不能证明没有竞态，也不能确认干预降低了自然发生的故障率。
历史复现卡和历史对照卡当前健康异常，未在其上执行，也未复位共享设备。

CANN 9.2.0、compiler `20260805_101249091`、Python 3.12.13、
Torch 2.12.0+cpu、torch_npu 2.12.0 在另一张健康卡独立编译并完成：
原始入口 50 次稳定重放、同样 18 个无扰动 family（900 次）、15 个时序 family
（750 次）。三个进程均先完整训练，退出/guard 为 0、结束后健康。
无扰动 B2/HV4/C3 的注释负对照 anchor 有 32,742 个 dh0 元素不同，随后
50 次均与 baseline 的正确 anchor 逐位相同；其他 17 个 family 均稳定。
这是首次调用异常，不能记为 50/50 次错误，也不冒充原始入口自然重放。

两套 CANN 的 OPP 目录均为 `ascend910_93`、`ascend910b`、`ascend950`；
完整原始版本行及 version.info SHA 在各 job 的 environment.json。
未从 CANN 9.1 复制编译产物到 CANN 9.2。

### 共同恒等 VF 时序干预：跨卡、跨 CANN 的新因果证据

在执行前记录 H2：只延迟最终 VF 的 seed 读取，应增加原始槽序发生提前覆盖的机会；
连续槽号应对这一延迟保持数值不变。baseline、ring、negative 都增加相同的辅助 VF，
在最终更新之前重复 load/store 同一组 64 个 FP32 decay 值，间以 `VST_VLD` barrier。
没有算术运算、dtype 转换、新缓冲或信用变化；只有三份副本共同具有的 SPINS 标量。

两卡均先完成完整 Kimi 训练，再执行 B1/HV32/C64/T4096、SPINS=4096 的三份实验 scan。
四输出全部与原始逐位相同，并通过既有 CPU FP32 物理 stage 预算：
`atol=rtol=0.004, relative_l2≤0.01`；dh0 的实测相对 L2 为 `3.3611388062126935e-5`。
随后才运行固定 B2/HV4/C3 的缩小实验，每行每份副本每张卡均为独立 anchor + 50 次。

| SPINS | baseline / 注释负对照相对未扰动结果 | 连续槽号副本 |
| ---: | --- | --- |
| 0 | 两卡均 0/50 错误 | 两卡均 0/50 错误 |
| 64 | 两卡 baseline 与负对照的 anchor 和后续调用不同；C 卡 baseline 有 1/50 次错误，另三组为 0/50 | 两卡均 0/50，含 anchor 全部相同 |
| 256、1024、4096（各自独立统计） | 两卡、两份副本均 50/50 次 dh0 错误 | 两卡均 0/50，与未扰动结果逐位一致 |

所有 family 的 `dAqk/dh/dv` 都与未扰动结果逐位一致。SPINS=64 的 anchor 是采样前
单独的一次调用，不计入 50 次发生率；“50 次都偏离错误 anchor”不等于“50 次都错误”。
大 SPINS 下 baseline 和负对照每次都产生相同的错误，因此仍完整捕获张量并记录为错误，
没有把重复性当成正确性。

这是在新健康卡上对既有无序边的受控干预，不能冒充未改动原始 kernel 的自然复现。
它与原始模型的无序对、历史错误张量的特定 seed 替换特征共同支持同一个机制。
完整四输出哈希、每次差异、所有错误张量及源文件运行前后哈希都已保留。

CANN 9.2 同组实验的大 SPINS 结果一致：256/1024/4096 时 baseline 和负对照
各 50/50 次 dh0 错误，ring 各 0/50；SPINS0 和 64 的 baseline/负对照
各有错误 anchor，但随后 50 次均正确。其他三个输出始终相同。
完整 C64 实验扫描的三份副本也逐位等于原始且通过原有 stage 预算。
三张卡、两套环境的 SPINS4096 错误 anchor 各有 65,492 个变化元素，分布在
BHV0/2/4/6；同一个无拟合 seed 替换预测各解释 65,491 个，余下一个的
相对 L2 约 `3.742233e-6`。CPU 分析与设备执行分开记录。

### 影响面

这条机制需要奇数 C，并且一个 cube 组连续承担多个 BHV；边界前的 BHV
暴露于下一 BHV 的 seed 写入。B2/HV4 的 bd1/2/3/4 都满足，模型危险对数
分别 14/12/10/8（每个边界两个 AIV）。偶数 C 连续维持交替槽号；单个 BHV
没有后继覆盖。这只排除了上述特定边界的竞争，不构成其它形状的完整安全证明。
历史样本仅 BHV6 出错；受控长延迟让所有四个边界前 BHV 出错，变化可覆盖
部分行或几乎整个 128×128 state，符合覆盖进度依赖时序的机制。

## 后续 kernel 批次的建议

建议在后续获批批次维持跨 BHV 的 seed 槽相位，或设计明确归还全部在途槽的边界协议。
不能简单把 seed 信用减为 1：当前预取在消费之前，需要另行分析循环进度，可能造成死锁。
本任务的 ring 副本只用于检验机制，不接入生产；正式修复需独立完成完整前后向、
缓存、形状与跨 bd 验收。结论不扩展到未测试的库修订、设备或 CANN 环境。

## 回执与重放入口

完整索引：[manifest.json](../../kernels/projects/a5/kda_scan_diag/evidence/manifest.json)；
阶段汇总：[validation-summary.json](../../kernels/projects/a5/kda_scan_diag/evidence/closeout/validation-summary.json)；
重放方式：[诊断工具 README](../../kernels/projects/a5/kda_scan_diag/README.md)。

| 证据 | 入口 |
| --- | --- |
| 运行前 H1/H2 原文、源行更正 | [preregistered-predictions.json](../../kernels/projects/a5/kda_scan_diag/evidence/provenance/preregistered-predictions.json) |
| 无关注释负对照的 AST/CCE 相同性 | [negative-controls.json](../../kernels/projects/a5/kda_scan_diag/evidence/provenance/negative-controls.json) |
| CANN 9.1 的 9,900 次无扰动副本重放 | [verified-sweeps.json](../../kernels/projects/a5/kda_scan_diag/evidence/native/verified-sweeps.json) |
| 两卡共同扰动的逐条件统计 | [verified-stress.json](../../kernels/projects/a5/kda_scan_diag/evidence/native/verified-stress.json) |
| CANN 9.2 的三阶段结果 | [verified-cross.json](../../kernels/projects/a5/kda_scan_diag/evidence/native/verified-cross.json) |
| 历史错误的 CPU 反事实分析 | [counterfactual.json](../../kernels/projects/a5/kda_scan_diag/evidence/historical/counterfactual.json) |
| 新错误的 CPU 分析，两卡 / 跨 CANN | [fresh-stress-counterfactual.json](../../kernels/projects/a5/kda_scan_diag/evidence/historical/fresh-stress-counterfactual.json)、[cross-stress-counterfactual.json](../../kernels/projects/a5/kda_scan_diag/evidence/historical/cross-stress-counterfactual.json) |
| 交付代码与实际真机代码的对应 | [source-delivery.json](../../kernels/projects/a5/kda_scan_diag/evidence/closeout/source-delivery.json) |

共记录 18 个完整 workload 在先的进程：原始入口 150 次、无扰动实验副本
10,800 次、共同扰动副本 2,250 次测量调用；每个 family 另有不计入次数的 anchor。
不同条件不合并估计故障概率。每次实际张量、四输出哈希、发生偏差的行分布分别存档。
主机全测为 `1225 passed, 6 skipped`；诊断工具自身 11 项测试只覆盖哈希、统计和脱敏。
首次主机全测因未改动基线的 README/PM 看板不同步出现一次失败；保留失败证据，
快进 PM 的元数据修复后通过，真机源码未变。编译中既有 D-084 性能警告未被隐藏，
本任务没有性能结论或相关优化；seed WAR 是调查对象，不能作为生产通过项。

归档收尾：[archive-restoration.json](../../kernels/projects/a5/kda_scan_diag/evidence/closeout/archive-restoration.json)
记录三份私有归档的哈希及 6,923 个文件的恢复核验。原环境归档已传回并恢复；
跨环境归档先在分配的 Docker 内恢复到新目录逐文件核验；随后异地增量备份也完成，
81,473,264 字节增量包加既有原环境归档，可恢复全部 2,281 个跨环境文件，
每个文件均与原始清单的 SHA256/大小一致。模型、trace、生成源码、旧版实际运行脚本
及失败记录另行归档。
[capture-correspondence.json](../../kernels/projects/a5/kda_scan_diag/evidence/closeout/capture-correspondence.json)
核对恢复后的 264 份 anchor、1,350 份偏差/错误 trial 与全部四输出哈希、
差异元素计数及逐行形态；包含稳定重复的错误，未仅检查文件存在。
