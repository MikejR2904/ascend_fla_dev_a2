# Gated Delta Rule

Chunked gated delta rule for Ascend a5 (DAV_3510), as a single fused PyPTO-Pro kernel.

## Kernel list

| 文件 | Kernel | Host entry | 描述 |
|---|---|---|---|
| `gated_delta_rule_fwd_impl.py` | `gated_delta_rule_fwd_kernel` | `gated_delta_rule_fwd` | 前向全算子：chunk 内 WY 构造 + chunk 间串行 scan，单 `@pl.jit`、单 launch |

## 接口

```python
from pypto_gym.ops.pypto_pro.experimental.ops_transformer.gated_delta_rule.gated_delta_rule_fwd_impl import (
    gated_delta_rule_fwd,
)

o = gated_delta_rule_fwd(q, k, v, g, beta)
```

| 张量 | Shape | dtype | 说明 |
|---|---|---|---|
| `q`, `k` | `[B, T, H, K]` | bfloat16 | |
| `v` | `[B, T, H, V]` | bfloat16 | |
| `g` | `[B, T, H]` | float32 | 对数衰减，语义上 `<= 0` |
| `beta` | `[B, T, H]` | bfloat16 | **logits**，`sigmoid` 在 kernel 内部施加 |
| `o` (返回) | `[B, T, H, V]` | bfloat16 | 内部全程 fp32 运算，仅最终一次 cast |

`scale = 1/sqrt(K)`；无 q/k L2 norm；`H == HV`；无 initial/final state。
已验证范围：`K, V <= 256`，`T` 任意（尾块由 kernel 处理），`C = 64` 为编译期常量。

## Kernel 结构

单 kernel、单 launch，两个阶段由一次 grid-wide `sync_all(HARD, MIX)` 隔开。
两个阶段的两个 AIV sub-block 都参与计算。

```
chunk-prep 阶段（并行度 = B*H*ceil(T/C) 个 chunk）
  preprocess (Vec)  beta_act / q_s / k_f32 / k_beta / v_beta / g 列 /
                    下三角 ones / k^T
  gram       (Cube) decay = g @ upper_ones, kk_gram = k_beta @ k^T,
                    qk_gram = q_s @ k^T
  wy         (Vec)  decay_exp / decay_mask / wy_t（前向替换求逆）/
                    attn_intra / q_decay / k_beta_decay / decay_tail
  uw         (Cube) u = wy_t @ v_beta, w = wy_t @ k_beta_decay
        —— 四步在同一 block 内用 INTRA_BLOCK 握手 e0..e3 衔接

          ── sync_all(HARD, MIX) ──

scan 阶段（并行度 = B*H*ceil(V/VT) 个 (bh, v_tile) 工作项）
  每个工作项内部对 chunk 串行（S 是真递推）；cube/vector 用 8 事件
  depth-1 INTRA_BLOCK 握手 e0..e7 交替：
      v_prime = w @ S_prev
      v_new   = u - v_prime          v_scaled = v_new * decay_tail
      o       = q_decay @ S_prev + attn_intra @ v_new
      S       = S_prev * decay_exp[C-1] + k^T @ v_scaled
```

状态 `S[K, V]` 经 GM workspace 传递而非常驻 L1——这是 `K, V = 256` 能放下的原因。

## 常量

| 常量 | 值 | 说明 |
|---|---|---|
| `C` | 64 | Chunk size |
| `TILE_COL` | 128 | K/V 列 tile 宽度（K<=256 由列循环覆盖） |
| `VT` | 64 | scan 阶段 V tile 宽度，同时是工作项划分维 |
| `KT` | 128 | scan 阶段 K 收缩块宽度 |
| `HC` | 32 | `C//2`，`DualModeSplitM` 下每个 AIV 承担的行数 |

## 性能

真机 msprof（Ascend950PR_9589，32 cube / 64 vector，CANN 9.1.0-beta.1-950），
8 个验收 case 的 kernel Task Duration 合计：

| 版本 | TOTAL (µs) | `mid` (T2048, H8, K128, V128) | `mid` aic_mac_ratio |
|---|---:|---:|---:|
| 初版 | 1134.8 | 638.0 | 0.32 |
| 工作项扁平化 + 双 AIV | 670.7 | 349.3 | 0.45 |
| **当前**（k^T 折入 chunk-prep，去掉第二道 barrier） | **646.0** | **346.6** | **0.45** |

三处提速全部来自工作分配与阶段结构，数学、容差、golden 均未改动。

剩余瓶颈是 `mid`（占总量 53%），且**不是任何单一 pipe**——最高的 `mac` 也只有 0.45。
它是 dependency-bound：scan 阶段对 32 个核只能暴露 `B*H*vtiles = 16` 个工作项，
每项又是 32 步串行递推。已实测排除的两条路：把 `o` 提到独立的宽并行阶段
（scan 省下的与新阶段的 per-item cube 占用相抵，反而 +12 µs），以及倾斜的
chunk-prep 分配（两阶段之间有 grid barrier，makespan = phase1 + phase2，
不均分只会拉长 phase1）。

`aic_mte2_ratio` 长期在 0.1–0.17，这是正确的：L2 命中 97–100%、主存带宽 3–5 GB/s，
工作集完全驻留片上。把 MTE2 利用率推高只能靠制造访存，不是优化。

完整的两轮 msprof 归档（7 组 aic-metrics + sample-based）与逐项解读见
[`docs/perf/PERF_REPORT.md`](docs/perf/PERF_REPORT.md)。Stage 3.5 的 KAL 预构建
产物与其边界发现见 [`kal/README.md`](kal/README.md)。

## 硬件行为注意事项

以下三条在本机以**错误数值结果**（而非编译错误）的形式出现过，改动时容易重新引入：

- matmul 的 N 小于 64 会算错 → V 在 host entry 里补齐到 64 的倍数，scan 阶段所有
  V tile 都是满宽。
- L0A / L0C tile 声明成 `[128, .]` 但以 `valid_shape [64, .]` 使用会算错 →
  scan 的 K 输出块固定 64 行。
- transpose-load（`order=[2,1]`）的 tile 可以做 matmul 的 RIGHT 操作数，但不能做
  LEFT → `k^T` 在 preprocess 里用 vector transpose 显式物化，再按普通 layout 载入。

另外，`auto_mutex` 为每个 traced block 生成一个 `<pyname>__mutexid_N`；在两个
**兄弟**运行时分支里复用同一个 Python tile handle 名会得到
`error: use of undeclared identifier '<name>__mutexid_0'`。每个分支用各自的名字。

## 目录

```
gated_delta_rule_fwd_impl.py   kernel + host entry（唯一的运行时文件）
docs/perf/                     真机 msprof 归档：baseline/ 与 optimized/ + PERF_REPORT.md
kal/                           Stage 3.5 KAL 预构建产物（参考资料，非运行时）
```

## 已知限制与注意事项

- **平台**：仅在 Ascend 950PR（DAV_3510 / `NpuArch=3510`，32 cube / 64 vector）上验证。
  其它 SoC 未验证，`@pytest.mark.soc("950")` 据此标注。
- **形状范围**：已验证 `K, V <= 256`；`C = 64` 是编译期常量，不随入参变化。`T` 任意，
  尾块在 kernel 内处理。`H == HV`，不支持 GQA 式的 K/V 头折叠。
- **语义范围**：`beta` 按 logits 传入（kernel 内施加 sigmoid），`allow_neg_eigval=False`；
  无 q/k L2 norm；不支持传入 initial state 或返回 final state。
- **输出补齐**：`V` 在 host entry 内补齐到 64 的倍数（matmul N<64 会算错），返回前切片还原。
  该切片是 view，不产生拷贝。
- **精度判据**：输出是 bf16，固定绝对容差 1.0 在本算子的量级（peak 1e2–1e6）下不可达，
  因此判据为 peak-scaled，详见下方测试一节。
- **host 开销**：每次调用有约 650 µs 的 host 侧固定开销（workspace 分配 + 框架下发），
  与 kernel 本身无关。评估 kernel 性能请用 msprof Task Duration，不要用 wall-clock。
- **`kal/` 不是运行时代码**：它依赖 `pypto-kernel-arch-lang` 插件（上游 PR !502，尚未合入
  master）提供的 `easyasc` 仿真器载荷，本仓库不提供。该目录是设计记录与结构参考。

## 测试

测试文件：[`tests/ops/pypto_pro/gated_delta_rule/test_gated_delta_rule_fwd.py`](../../../../../../tests/ops/pypto_pro/gated_delta_rule/test_gated_delta_rule_fwd.py)

```bash
pytest tests/ops/pypto_pro/gated_delta_rule -v        # 全部 8 个 case

# 或直接运行，可按名字挑子集
cd tests/ops/pypto_pro/gated_delta_rule
python3 test_gated_delta_rule_fwd.py
python3 test_gated_delta_rule_fwd.py mid wide_kv
```

8 个 case 覆盖对齐、T 尾块、V 尾块、多 tile K/V、batch>1、长序列和最宽 K/V。
判据是 bf16 容差行的 peak-scaled 形式，并额外交叉验证输出不差于 golden 的
理想 bf16 舍入——即除最后一次 cast 外没有任何计算误差。当前 8/8 全部
`matched_ratio = 1.000000`。

---

## 训练前向 + 反向算子组（8 个单 kernel 算子）

上面的 `gated_delta_rule_fwd` 是 **lean forward**：一路算到 `o`，M1–M4 的中间量被 M5 消费后即丢弃。
训练场景需要这些中间量，因此训练前向被拆成 **两个算子**（`gdr_fwd_train_wy` + `gdr_fwd_train_scan`），
把 lean forward 丢掉的 17 个边界张量 + `s_hist` + `v_new` 物化到 GM；反向在其上再分 **六个算子**。

这 8 个算子**各自是一个 `@pl.jit`、一次 launch**，在 Python 层按固定顺序组合：

```
gdr_fwd_train_wy → gdr_fwd_train_scan → gdr_bwd_dv_local → gdr_bwd_reverse_scan
                 → gdr_bwd_dqkw → gdr_bwd_wy_matmuls → gdr_bwd_wy_reduce → gdr_bwd_finalize
                 → (dq, dk, dv, dg, dbeta)
```

算子边界上**只有 GM 张量交接，没有任何 host 端算术**——wrapper 只做参数校验、纯 Python 形状运算、
`torch.empty` 分配和一次 kernel 启动。拆分的根因是 UB：把 M1–M5 融成一个 kernel 实测
`ub_bytes = 359,424 B`，是 253,952 B 预算的 141.5%（超 105,472 B），而 pypto 后端的 UB 是单调
bump 分配、无跨相生命周期复用，两个常驻相只能相加。mutex 不是原因（融合版 43 default / 29 merge，
在 32 以内）。拆开后每个 kernel 只持有一个常驻相，并且各自夺回了自己的 task grid。

| 算子 | 角色 / Module 范围 | 实现文件 |
|---|---|---|
| `gdr_fwd_train_wy` | 训练前向 OP-A：chunk 内 WY 构造（M1–M4），物化 17 个边界张量 | `gdr_fwd_train_wy_impl.py` |
| `gdr_fwd_train_scan` | 训练前向 OP-B：chunk 间串行 scan（M5），产 `o` + `s_hist` + `v_new` | `gdr_fwd_train_scan_impl.py` |
| `gdr_bwd_dv_local` | 反向 OP3：dv 的 chunk 内局部贡献（M6 / b1） | `gdr_bwd_dv_local_impl.py` |
| `gdr_bwd_reverse_scan` | 反向 OP4：逆向 chunk 间 dS scan（M7 / b2） | `gdr_bwd_reverse_scan_impl.py` |
| `gdr_bwd_dqkw` | 反向 OP5：chunk 并行 dqkwg（M8 / b3） | `gdr_bwd_dqkw_impl.py` |
| `gdr_bwd_wy_matmuls` | 反向 OP6a：WY 逆伴随的 cube 半（M9 cube / b4 matmuls） | `gdr_bwd_wy_matmuls_impl.py` |
| `gdr_bwd_wy_reduce` | 反向 OP6b：WY 逆伴随的纯 vector 半（M9 vec / b4 归约 + b4b_dk） | `gdr_bwd_wy_reduce_impl.py` |
| `gdr_bwd_finalize` | 反向 OP7：finalize + 去 padding + 转置 + cast，产 5 个公开梯度（M10） | `gdr_bwd_finalize_impl.py` |

`C = 64` 是全组共享的编译期 chunk 常量，`N = (T + 63) // 64`。内部布局统一为 `[B, H, N, C, *]`
（公开输入/输出才是 `[B, T, H, *]`），padding 行带 `g = 0` 故贡献 inert，终末 `[:T]` 切片只发生在
`gdr_bwd_finalize` 的 b7。

## gdr_fwd_train_wy

Host entry `gdr_fwd_train_wy_wrapper`，kernel `gdr_fwd_train_wy_module1234_kernel`。

### 数学契约

与 lean forward 的 M1–M4 **完全相同的数学**，只是把结果物化出来：

| 张量 | Shape | dtype |
|---|---|---|
| `q`, `k` | `[B, T, H, K]` | bfloat16 |
| `v` | `[B, T, H, V]` | bfloat16 |
| `g` | `[B, T, H]` | float32 |
| `beta` | `[B, T, H]` | bfloat16（logits，sigmoid 在 kernel 内） |

17 个输出全部 fp32、全部在内部 `[B, H, N, ...]` 布局，**不做 `[:T]` 切片**：

- `[B,H,N,64,K]`：`q_s`、`k_f32`、`dfac`、`q_de`、`w`
- `[B,H,N,64,V]`：`v_b`、`u`
- `[B,H,N,64,64]`：`attn`、`a_mat`
- `[B,H,N,K,64]`：`q_t`、`kbeta_t`、`w_t`（预转置，供下游非转置载入 L0A）
- `[B,H,N,64]`：`g_f32`、`beta_act`、`sig`、`decay`、`decay_exp`

四段数学：M1 preprocess（Vector：transpose + cast fp32 + pad 到 `Tp = N·C`、sigmoid、scale、`q_t`）→
M2 cumsum_qk（Cube：`decay = LTones @ g`、`P1 = k_beta @ k_f32^T`、`P2 = q_s @ k_f32^T`）→
M3 wy_build（Vector：`decay_exp`、`q_de`、`dfac = exp(decay_last − decay)`、
`attn = (I − tril_strict(P1·L))^{-1}` 经 C−1 步前向替换、`a_mat`、`kbeta_t`）→
M4 uw（Cube：`u = attn @ v_b`、`w = attn @ kbeta_de`、`w_t`）。
`k_beta`、`P1`、`P2`、`kbeta_de`、`L_mask` 是内部 scratch，不物化。

按下游消费方分三类：**13 个是反向残差**
（`q_s, v_b, g_f32, beta_act, sig, q_t, decay, attn, kbeta_t, w_t, k_f32, decay_exp, a_mat`）、
**7 个是交给 `gdr_fwd_train_scan` 的 handoff**（`u, w, q_de, dfac, k_f32, decay_exp, a_mat`），
两者交集 3 个。`s_hist` 与 `v_new` 不在此产出，属 OP-B 的责任。

容差：标称 `atol/rtol = 1e-2`；实际判据为混合式 `allclose(rtol=1e-2, atol=1e-2)` **或** peak-scaled 相对 L2。
已验证范围 `B ≤ 16`、`T ≤ 8192`、`H ≤ 128`、`K, V ≤ 128`。

### Grid / tile

- **BHN-parallel 扁平切**：work item = 一个 `(b,h,n)` chunk，共 `B*H*N` 个，无 carried state 故全独立；
  每核连续段 `per_core = ceil(B*H*N / num_cores)`，`block_dim = min(max_blocks, B*H*N)`，**launch 次数 1**。
- 常量：`C = 64`、`SUB = 64`（K/V free 维子块宽，即 TK = TV = 64）、`KMAX = 128`、`SCALE = 1/sqrt(K)`。
  K=64 → 1 个子块，K=128 → 2 个，V=96 → 2 个（末块 valid 32）。
- 归约轴全部单 tile：cumsum 的 LTones 收缩 C=64；`P1`/`P2` 收缩 K≤128 按 SUB=64 子块累加；`u`/`w` 收缩 C=64。
  free 维按子块并列 store，无跨子块状态。
- tile 全是 `[64,64]` 级（UB 侧 13 个，L1 为 `[64,128]`/`[128,64]`/`[16,64]`，L0C 为 `[16,64]`/`[64,64]`），
  depth=1 单缓冲 + `auto_mutex`。
- **实测片上预算**：UB `127,744 B = 50.3%`，mutex **28 default / 17 merge**，pipe-flag `0 / 0`
  （`merge_siblings=True` 对本算子强制）。

四项载重结构不得重新引入：mutex 链必须**扁平三段**（嵌套会死锁）；M4 必须持有**自己的** L1/L0C
（`l1_attn`/`l1_op`/`l0c_uw`，不复用 M2 的——这正是 mutex 为 28 而非 24 的根因）；`l1_op` 必须精确声明
`[C,C] = [64,64]`（声明成 `[C,KMAX]` 会截断收缩轴，max_abs 从 0 变 39）；`v_b` 必须在 M1 产出，
不能在 M3 经 GM round-trip 重建（跨 tile WAR 不被 `auto_sync` 追踪，首子块会损坏）。

### 实测结果

真机 msprof 单组 `PipeUtilization`，生产形状 `B=1, T=2048, H=8, K=V=128`：

| 口径 | Task Duration | 有界管道 | `cube_utilization` |
|---|---:|---:|---:|
| 单算子（当前 host，28 cube / 56 vector） | **199.407 µs** | aiv_vec 46.0% | **90.076%** |
| 组合流水线内（warm，同一 msprof pass） | **200.54 µs** | — | — |
| 旧 host（32 cube / 64 vector，已下线） | 160.048 µs | — | 98.577% |

按 cube_utilization 口径 **达到 80% 验收线**。

### 已接受的性能缺口

无。

## gdr_fwd_train_scan

Host entry `gdr_fwd_train_scan_wrapper`，kernel `gdr_fwd_train_scan_module1_kernel`。

### 数学契约

7 个 handoff 输入（全 fp32、全 chunked、已含 OP-A 的 padding 行）：
`u [B,H,N,64,V]`、`w`/`q_de`/`k_f32`/`dfac [B,H,N,64,K]`、`a_mat [B,H,N,64,64]`、`decay_exp [B,H,N,64]`。

三个输出：`o [B, 64N, H, V]` bf16、`s_hist [B,H,N,K,V]` fp32、`v_new [B,H,N,64,V]` fp32。
每 `(b,h)` 沿 chunk `i = 0..N-1` 串行，持久状态 `S[K,V]` 初值 0：

```
s_hist[:,:,i] = S                       # 进入态快照，必须在 s15 更新之前取；s_hist[:,:,0]=0
v_prime = w_i @ S ;  v_new_i = u_i − v_prime     # v_new 同样 store-side 提升为公开输出
o_inter = q_de_i @ S
o_pad[:,:,i] = o_inter + a_mat_i @ v_new_i
S = S * decay_exp_i[C-1] + (k_f32_i * dfac_i)^T @ v_new_i
```

`s16` 把 `o_pad` 拼回公开 `[B, 64N, H, V]`、fp32→bf16；真实公开输出是外层 `o[:, :T]` 视图
（`T` 不是本算子参数——输入已 padded）。`s_hist` 与 `v_new` 是相对 lean forward **新增**的两个输出，
零算术改动：前者是已驻留状态的一次快照，后者是 scan 内部本就算出的量。

`v_new` 是**载重的**：反向 OP5 必须吃它，不能用 primary `v` 重构——实测用 primary `v` 使
`dq` relL2 ≈ 0.999，改用 `v_new` 恢复到 ~1e-6。

精度：全程 fp32，唯一 narrow 在 `s16` 存 `o`。`o[:, :T]` 判据为混合 peak-scaled
`allclose(rtol=1e-2, atol=1e-2·max|o_g|)` **或** 相对 L2 `≤ 1e-2`；`s_hist`/`v_new` 以 fp32 直接比对
（误差量级 ~1e-5，fp32 matmul roundoff）。动态范围 `B ≤ 16`、`H ≤ 128`、`N ≤ 128`、`K, V ≤ 256`。

### Grid / tile

v1 是 KAL 转译产物的 re-home：硬件 7/7 正确但 `9712.93 µs`、只启动 8/32 核、有界管道 45.1%、`K,V ≤ 128`。
当前版本（v2）在**数学与输出契约完全不变**的前提下重做了 tile、地址、调度、分核与同步：

- **任务 = `(b, h, v_tile_pair)`**，`tasks = B*H*vpairs`，`vpairs = ceil(vtiles/STREAMS)`、`vtiles = ceil(V/VT)`。
  chunk 轴仍严格串行（loop-carried `S` 不跨核），**V 轴列切片数学独立**，三个输出在 V 轴写不相交列，
  故无跨核竞争也无跨核归约。每个 task 内 2 条 V 流交错、N 个 chunk 顺序 scan。**launch 次数 1**。
- `VT`/`STREAMS` 由 host 纯 Python 标量搜索选定（最大化已启动核 → 偏好 `S=2` → 偏好大 `VT`）。
  生产形状 `B=1,T=2048,H=8,K=V=128` 选到 `VT=32、vtiles=4、STREAMS=2、vpairs=2、tasks=16`，
  即 16 个 cube 核、32 条并发 V-scan（v1 是 8 / 8）。
- 常量：`C = 64`、`KT = 64`、`VTP = 64`（V 物理 tile 宽；`VT=32` 时靠 `set_validshape` 收窄）、
  `CH = 32`（= C/2，`DualModeSplitM` 下每个 AIV sub-block 的行带）、`NKOB_MAX = 4`、`STREAMS = 2`。
- 六类资源投影（地址 extent）：UB **83,200 B = 32.76%**、distinct mutex id **27 / 32 = 84.4%**、
  核内 pipe-flag **0**、cross-core event id **7 / 16 = 43.75%**、L0A **32,768 B = 50%**、L0B 同为 50%；
  附 L1 `180,224 B = 34.38%`、L0C `65,536 B = 25%`。对照 v1：UB 97.7%、L0A/L0B 均 100%。
- GM workspace 只剩一个 `vscaled_gm [B,H,64,V]`：`S` 的 GM 源直接用 `s_hist`（它本来就是进入态），
  `v_new` 的 GM 源直接用公开输出 `v_new`，`k^T` 改为 L1→L0A 的 `order=[1,0]` 转置填充。
- mutex id 共享（官方 `make_tile_group` 允许「不同 Tile 之间可以使用相同的 mutex ID」）用在四处，
  逐条论证过活跃区严格串行故零额外串行化：`obf_ub`↔`vs_ub`、`d_acc`↔`vp_acc`、
  `qde_l1`/`a_l1`/`kf_l1`（共 id 14）、`st_l1[s]`/`vn_l1[s]`（共 id 15/16）。

### 实测结果

| 口径 | 数值 |
|---|---:|
| 调优前 → 调优后（同一形状，warm，可复现 harness 内） | **380.0 µs → 178.9 µs** |
| 组合流水线内（warm，msprof op_summary 第 4 次出现） | **191.63 µs** |
| 有界管道 `aic_mac`（当前 host） | **74.7%** |
| 有界管道 `aic_mac`（旧 host） | 79.6% |

> PERF_NEWHOST 对照表里 `1149.873 µs` 那一行**不可信**：那是该 session 独立 driver 里第一次 JIT
> 编译后的近冷测量，没有记录 warm-up。同一份未修改文件在可复现 harness 里 warm 之后读 ~380 µs。

`cube_utilization = 18.965%` 不是本算子的验收口径，也不会被修好：`C = 64` 是 SPEC 冻结常量，
所有 matmul 都是 `[64,64]` 级，算术密度天生极低。它是**延迟受限的串行递推**，判据是管道被占满
（mac ratio），不是算力被喂满。

### 已接受的性能缺口

- 有界管道 `aic_mac` **74.7% < 80%** 验收线（旧 host 79.6%），已按 2026-10-05 的裁定接受。
- device-wide 占用在生产形状上限 **50%**：本算子的内在独立工作量是 `B·H·ceil(V/VT)`，
  `B=1,H=8,V=128` 在 `VT=32` 下是 32 条 V-scan，配 `STREAMS=2` 即 16 个 cube 核。
  要同时做到核全启动与每核双流需要 `VT=16`，那会把 matmul 的 N 维降到 16、向量寄存器有效率降到 25%，
  不采用。`B·H ≥ 16`（如 B=2）时自动达到 100%。

## gdr_bwd_dv_local

Host entry `gdr_bwd_dv_local_wrapper`，kernel `gdr_bwd_dv_local_module1_kernel`。

### 数学契约

dv 的 **chunk 内局部贡献**，是逆向工作里的非 scan 半（纯 dS 携带是 OP4）。每 `(b,h,n)` chunk 独立，
无跨 chunk 依赖。行列 `i, j ∈ [0, C)`：

```
A_l[i,j]   = scale * (k_i · q_j) * exp(g_col_j − g_row_i)   当 j >= i（上三角含对角），否则 0
dv_local   = A_l @ grad_o_chunk
```

`(k_i · q_j)` 由 `k_f32 [C,K] @ q_t` 得到，`q_t` 是上游已预转置的 `q_s^T`（`q_s = scale*q`），
故 `scale = 1/sqrt(K)` 已折进 `q_t`，**本 kernel 不再乘 scale**。

| 张量 | Shape | dtype | 来源 |
|---|---|---|---|
| `grad_o` | `[B, T, H, V]` | bfloat16 | primary（kernel 内切 chunk 行并 cast fp32） |
| `k_f32` | `[B, H, N, 64, K]` | float32 | 残差（train_wy M1） |
| `q_t` | `[B, H, N, K, 64]` | float32 | 残差（预转置） |
| `g_f32` | `[B, H, N, 64]` | float32 | 残差 |
| `dv_local`（输出） | `[B, H, N, 64, V]` | float32 | → OP4 |

容差 `atol/rtol = 1e-2`；`K, V ≤ 128`（继承父反向更紧的包络）。

### Grid / tile

- **BHN 展平分核**：`tasks = B*H*N`，`per = CeilDiv(tasks, num_cores)`、`begin = per*core_idx`、
  `end = min(begin+per, tasks)`。完全并行、零同步开销、均分即均衡。混合 kernel 用 `core_num`，
  `block_dim = min(max_cube_cores, B*H*N)`，**launch 次数 1**。
- 常量：`C = 64`、`KMAX = 128`、`VMAX = 128`，**无 SCALE 常量**。本算子**不做 TK/TV=64 子块**
  （父为 10 相共享池省 UB 才子块化）：直接用全宽 `[C,128]` / `[128,C]` 单 tile 单 matmul，无 K/V 子块循环。
- 两段 cube 各持独立 L1/L0C（M4 finding）：cube-A 算 `A_l`（`l1_k [64,128]`、`l1_qt [128,64]`、
  `l0c_al [64,64]`），vec 段做 cumsum/gdiff/exp/掩码，cube-B 算 `dv_local`（`l1_alB`、`l1_go`、
  `l0c_dv [64,128]`）。`ub_al` 跨 vec 常驻（depth-1 `current()` 不轮转）。load 三组 depth=2 双缓冲。
- 片上规划：L1 `180,224 B`、L0A `49,152 B`、L0B `65,536 B`、L0C `49,152 B`。
  **实测片上预算**：UB `111,104 B = 43.8%`，mutex **22 default / 22 merge**，pipe-flag `0 / 0`，
  不需要 `merge_siblings`。

本算子是全分解里第一个 probe-seeded 算子，因此它的 drift 是校准基准：实测 `111,104 B / 22 mutex`
对 probe 的 `116,992 B / 19`，即 **UB ×0.950、mutex ×1.158**——mutex 的上漂是 SPLITM DBuff-pair
这个 idiom 带来的结构性税，不是算子本身的性质。

### 实测结果

| 口径 | Task Duration | 有界管道 | `cube_utilization` |
|---|---:|---:|---:|
| 单算子（当前 host） | **49.271 µs** | aic_mac 54.1% | **87.120%** |
| 组合流水线内（warm） | **47.79 µs** | — | — |
| 旧 host | 42.779 µs | — | 90.32% |

按 cube_utilization 口径 **达到 80% 验收线**。换 host 后的首个 smoke 就是它
（`test_gdr_bwd_dv_local.py` → 12/12 PASS，误差与旧 host 逐项一致）。

### 已接受的性能缺口

无。

## gdr_bwd_reverse_scan

Host entry `gdr_bwd_reverse_scan_wrapper`，kernel `gdr_bwd_reverse_scan_module1_kernel`。

### 数学契约

逆向 chunk 间 dS scan，是前向 `gdr_fwd_train_scan` 的方向对偶。每 `(b,h)` 沿 `i = N-1 .. 0` 逆序，
携带**自己的** `dS[K,V]`（初值 0；它**不读** `s_hist`——那是 OP5 的输入）：

```
dh[.,i]  = dS                                   # 进入态快照，取在 dS 更新之前；dh[.,N-1] = 0
dvc      = k_f32_i @ dS                          # [C,V]，收缩 K
dv2[.,i] = dv_local_i + dvc * exp(g_last_i − g_i)
dS       = exp(g_last_i) * dS + scale * (q_t_i @ grad_o_i) − (w_t_i @ dv2_i)
```

输入：`q_t`/`w_t [B,H,N,K,64]`、`k_f32 [B,H,N,64,K]`、`grad_o [B,T,H,V]` bf16（primary）、
`decay_exp`/`g_f32 [B,H,N,64]`、`dv_local [B,H,N,64,V]`（OP3）。
输出：`dh [B,H,N,K,V]`、`dv2 [B,H,N,64,V]`，均 fp32。容差 `atol/rtol = 1e-2`，`K, V ≤ 128`。

`scale` 已由上游折进 `q_s`（`q_t = q_s^T`），本算子不再显式乘。

### Grid / tile

- **BH-serial 逆向**：`tasks = B*H`，`per = CeilDiv(B*H, GetCubeNum())`，每核处理连续 `per` 个 `(b,h)`。
  **loop-carried `dS` 的逆向递推不可跨核切分**，故每个 `(b,h)` 的 N-chunk 逆序列在单核内完整跑完
  （与前向 scan 同）。V-subblock 与 reverse chunk 是核内嵌套循环，不参与分核。
  `block_dim = min(max_cube_cores, B*H)`，**launch 次数 1**。
- 常量：`C = 64`、`TV = 64`（V 子块宽，使转置后的 L0B ≤ 8192 fp32 元素）、`KMAX = 128`、`TILE = 128`，
  **无 SCALE 常量**。V=64 → 1 子块；V=128 → 2；V=96 → 2（末块 `valid_v = 32`）。
- `ub_dS [128,64] = 32,768 B` 是 **persistent 单 buffer，不参与轮转**；其余 landing/staging 单缓冲
  （scan 串行无跨 chunk 重叠），L0A/L0B 为 SPLITM DBuff-pair。
- 单 buffer 基线规划：UB `141,568 B = 55.7%`、L1 `163,840 B = 31.3%`、L0A `65,536 B = 100%`、
  L0B `49,152 B = 75%`、L0C `49,152 B = 18.75%`。cube-1 的 `l0a_k [64,128]` 与 cube-2 的
  `l0a_qw [128,64]` 被 vec 边界隔开、非同时活跃，共用同一上界槽错峰复用。
- **实测片上预算**：UB `140,032 B = 55.1%`，mutex **22 default / 17 merge**，pipe-flag `0 / 0`，
  **L0A 恰在 64 KB 上限**。不需要 `merge_siblings`。

这是 drift 模型的第二个数据点：实测 `140,032 B / 22 mutex` 对 probe `162,048 / 22` →
**UB ×0.864、mutex ×1.000**。与 OP3 的 ×1.158 对比可知 mutex 上漂由 SPLITM idiom 而非算子决定；
本算子交付的是 SINGLE 实现（DESIGN §4 当初基于「SINGLE 在 A5 不正确」的错误前提要求 SPLITM，
该前提已撤回，SINGLE 实现经声明、批准并仿真验证）。

### 实测结果

| 口径 | Task Duration | 有界管道 | `cube_utilization` |
|---|---:|---:|---:|
| 调优前 → 调优后（B=4） | **856.6 µs → 643.0 µs** | — | — |
| 单算子 msprof（B=4，32 个 BH task） | 859.652 µs | aic_mac 57.1% | 57.127% |
| 组合流水线内（生产形状 B=1，8 个 BH task） | **217.62 µs** | — | — |
| 旧 host（B=4，32 核整除） | 433.43 µs | aic_mac 57.0% | 99.235% |

**B=4 与 B=1 那两个数字测的是不同的东西，不要混用。** 旧 host 的 `cube_utilization 99.235%` → 当前
`57.127%` **不是回归**：`aic_mac` 的管道比例完全没变（57.0% → 57.1%），掉下来的是原始
`cube_utilization` 列——`B*H = 32` 个 task 在 28 个 cube 核上不再整除，恰好是 **2 个 launch wave**
（旧 host 正好 32 核），同样的 mac-busy 时间被摊到更长的墙钟上。这是形状/核数量化的产物；
要看这个口径就得在 **task 数能被 28 整除**的形状上测。`859.652 / 4 = 214.9 µs` 与流水线内的
`217.62 µs` 在同一量级，但这只是观察，不是推导。

### 已接受的性能缺口

无「接受缺口」条目；上面 B=4 的 `cube_utilization` 落差是已归因的测量形状产物，不是待优化项。

## gdr_bwd_dqkw

Host entry `gdr_bwd_dqkw_wrapper`，kernel `GDR_BWD_DQKW`。

### 数学契约

chunk 并行的 dqkwg 反向，产 WY 伴随之前的 `dq_p`、`dk_p`、`dw` 与 finalize 之前的 `dg_p`。
每 `(b,h,i)` chunk 独立、无 carried state。记 `h_i = s_hist[.,i]`（进入态 `[K,V]`）、
`decay` 为 chunk 内累积对数门（`g` 沿 C 个 token 的 cumsum）、`g_last = decay[...,-1]`，
`decay_exp = exp(decay)` 由 kernel 内重算（不是 GM 输入）：

```
b3.1  ds   = tril_incl( (grad_o_i @ v_new_i^T) * exp(decay_row − decay_col) * scale )   # 同时留下 ds^T
b3.2  dq_p = (grad_o_i @ h_i) * decay_exp * scale  +  ds @ k_f32_i
b3.3  dk_p = (v_new_i @ dh_i) * exp(g_last − decay)  +  ds^T @ q_s_i
b3.4  dw   = −(dv2_i @ h_i)
b3.5  dg_p = sum_K(dq_p * q_s_i) − sum_K(dk_p * k_f32_i)
      再把 per-chunk 标量 dg_last 折到最后一行（token C−1）：
      dg_last = exp(g_last)*sum_{K,V}(h_i * dh_i) + sum_{K,V}(dk_inter * k_f32_i)
```

输入：`grad_o [B,T,H,V]` bf16（primary）、`v_new`/`dv2 [B,H,N,64,V]`、`k_f32`/`q_s [B,H,N,64,K]`、
`s_hist`/`dh [B,H,N,K,V]`、`g_f32`/`decay [B,H,N,64]`。
输出：`dq_p`/`dk_p`/`dw [B,H,N,64,K]`、`dg_p [B,H,N,64]`，均 fp32。
容差 `atol/rtol = 1e-2`，`K, V ≤ 128`。每 chunk 六个 fp32 matmul（即六个 LEFT/L0A 操作数）。

**`v_new` 不可用 primary `v` 替代**（2026-09-02 实测，seed 42、`[1,128,4,64,64]`）：
用 `v_new` 时对 CPU autograd VJP 的相对 L2 是 `dq 3.8e-7`、`dk 7.2e-7`、`dg 1.3e-6`；
换成 primary `v` 则 `dq 0.999`、`dk 0.851`、`dg 0.550`。

### Grid / tile

- **BHN 扁平**：`tasks = B*H*N`，`per = CeilDiv(tasks, GetCubeNum())`，每 task 等量 6 个 matmul
  故天然均衡。`block_dim = min(max_cores, B*H*N)`，**launch 次数 1**。
- idiom 选 **SINGLE**（cube 侧 unguarded `.subblk(0)` + vec 侧 sub-block-0 guard，TV=64）：
  本算子 BHN 并行、无状态，三处 cube↔vec 握手搬的都是 `[C,C]` 或 `[C,K]`，没有任何输出 tile
  必须跨两个 sub-block 同时物化，SPLITM 不被强制。
- 常量：`C = 64`、`TK = 64`、`TV = 64`、`KMAX = 128`、`VMAX = 128`；运行时标量
  `SCALE = 1/sqrt(Kd)`、`SQRT_K = sqrt(Kd)`（用于 `q_raw = q_s·SQRT_K` 恢复）。
  `[C,C] = [64,64]` 的 `ds`/`ds^T` **永不子块**；`[K,V]` 的 `h`/`dh` 按 `TV=64` 子块载入成 `[128,64]`。
- **L0A 是本算子的 gate**：6 个 LEFT 操作数里最大的是 `[C,V] = [64,128]` fp32 = 32,768 B
  （`grad_o`、`v_new`、`dv2`）；cube-1 与 cube-2 被 vec 的 ds-mask 相隔开可错峰复用同一 DBuff 对，
  cube-2 内 5 个 matmul 程序序串行、任一时刻至多 2 个 LEFT 在场 →
  峰值 `2 × 32,768 = 65,536 B = 64 KB`，**恰在上限、零余量**。
- **实测片上预算**：UB `238,432 B = 93.9%`，mutex **47 default / 22 merge**，pipe-flag `0 / 0`，
  **L0A 与 L0B 均 64 KB**。`merge_siblings=True` 由实测**确认为交付必需**（default 47 ≫ 32）。
  它超出了 drift 调整后的*舒适*门槛（>215 KB、>44 default），这是一条 **heaviness finding**，
  不是失败——每一项硬预算都满足（UB 238,432 < 253,952；merge 后 mutex 22 ≤ 32；L0A/L0B ≤ 64 KB；0 flag）。

这是 drift 模型的第三个数据点，并**推翻**了「SINGLE ⇒ mutex ×1.0、UB ×0.86」那条规则：
实测对 probe `206,848 B / 35` 是 **UB ×1.153、mutex ×1.343**。真正的驱动因素不是
SINGLE-vs-SPLITM，而是**输出 K 轴子块化**——`K/V = 128` 把输出 K 轴切成 2 个子块后，
正确性要求对每个子块的 tile 都做 DBuff（本算子：5 个 L0C + 5 个 UB 落点 + 2 个 L1），
单相算子没有这笔税。`q_raw` 被消去（把 `√K` 当标量提出——数学等价，省掉一个桥接并修掉一个 clobber bug）。

### 实测结果

| 口径 | Task Duration | 有界管道 | `cube_utilization` |
|---|---:|---:|---:|
| 单算子（当前 host） | **203.271 µs** | aic_mac **63.4%** | **88.821%** |
| 组合流水线内（warm） | **202.78 µs** | — | — |
| host 自带参考样例（旧 host 口径） | ~2406 µs | aic_mac 63.6% | — |

按 cube_utilization 口径达到 80% 线；相对参考样例约 **11.8×**，与已在案的 12× host 速度差一致。

### 已接受的性能缺口

**有，已接受**：有界管道 `aic_mac` **63.4%**，对照**实测的 ~79.7% fp32-MAC 上限**。
唯一的杠杆是把 cube 操作数改成 bf16，而那是一次 SPEC 变更（输入/中间量 dtype 契约），
故按 2026-10-05 的裁定接受现状。注意 `aic_mac` 在换 host 前后几乎没动（63.6% → 63.4%），
说明这是算子的算术结构属性，不是 host 差异。

## gdr_bwd_wy_matmuls

Host entry `gdr_bwd_wy_matmuls_wrapper`，kernel `gdr_bwd_wy_matmuls_module12345_kernel`。

### 数学契约

WY 逆伴随的**闭式 matmul 半**。每 `(b,h,i)` chunk 独立、无 carried state；**八个 fp32 matmul 分三段 cube，
中间被两段 vector glue 隔开**（这正是 OP6a/OP6b 必须分家的结构动因——`dA` 反复穿越 cube↔vector）：

```
[cube A] b4.1 dA   = dw @ kbeta_t                 b4.2 dkbg = attn^T @ dw          ← 输出
         b4.3 dA  += dv2 @ v_b^T                  b4.4 dvb  = attn^T @ dv2         ← 输出
[glue-1] dA = strict_low * dA                      # tril_strict，保留 row>col
[cube B] b4.5 dA  = dA @ attn^T                   b4.6 dA  = attn^T @ dA
[glue-2] decay_fac = exp(decay_row − decay_col) * strict_low ;  dA = −(dA * decay_fac)   ← 输出
[cube C] b4.7 dkb = dA @ k_f32                    b4.8 dkt = dA^T @ k_beta         ← 输出
```

`k_beta = k_f32 * beta_act` 在片上形成（便宜的 vec 乘），不是 GM 输入；`strict_low` 与 `decay_fac`
采用 **mask-before-exp**，上三角先归零故不会出 inf。

输入（全 fp32，**没有任何 primary `[B,T,H,*]` 公开输入**，`N` 是独立的动态 chunk 数轴）：
`attn [B,H,N,64,64]`、`k_f32`/`dw [B,H,N,64,K]`、`v_b`/`dv2 [B,H,N,64,V]`、`kbeta_t [B,H,N,K,64]`、
`beta_act`/`decay [B,H,N,64]`。
输出：`dkbg`/`dkb`/`dkt [B,H,N,64,K]`、`dvb [B,H,N,64,V]`、`d_a [B,H,N,64,64]`，均 fp32，
**只被 OP6b 消费**。容差 `atol/rtol = 1e-2`；`B ≤ 16`、`H ≤ 128`、`N ≤ 128`、`K, V ≤ 128`。

golden 里 `A_kk = k @ k^T` 的 `dg2` 项可以代数地化成 `dkb`/`dkt` 的向量归约
（`AdA.sum(-1) = beta_act*sum_K(dkb*k)`、`AdA.sum(-2) = sum_K(k*dkt)`），所以 OP6b **一个 matmul 都不需要**
——这就是为什么这一刀切得干净、八个 matmul 全留在 OP6a。

### Grid / tile

- **BHN 扁平**：`tasks = B*H*N`，`per = CeilDiv(tasks, GetCubeNum())`，chunk 内
  `b = tid//(H*N)`、`h = (tid%(H*N))//N`、`i = tid%N`。`block_dim = min(max_blocks, tasks)`，
  **launch 次数 1**。
- 常量：`C = 64`、`TK = 64`（K 输出子块）、`TV = 64`（V 输出/收缩子块，V=96 → 子块 1 partial 32）、
  `KL0 = 64`（K/V=128 时收缩轴 split 步长，2 段 `matmul_acc`）、`KMAX = 128`、`VMAX = 128`、
  `L0C_FRACTAL = 1024`（fp32 L0C 默认）。**无 SCALE**：`1/sqrt(K)` 已折进上游的 `attn` 构造。
- **L0A/L0B 是本算子的 gate**（全分解最 matmul-dense 的算子，8 个 matmul，规定任一投影 >64 KB
  就在 Stage 3 停）：采用 output-subblock + 收缩轴 split 的方案后，L0A 与 L0B 峰值各
  **32,768 B = 32 KB = 50%**；朴素全宽方案则是 65,536 B = 恰好 100%。选前者以留 50% 余量
  应对转译时的 NZ staging。三段 cube 的 L0A/L0B **共享基址**（off-peak reuse，depth=2）。
- **实测片上预算**：UB **67.7%**，mutex **32 default / 19 merge**，pipe-flag **intra 0 / cross 12**，
  **L0A 与 L0B 均在 64 KB 上限、零余量**。`merge_siblings` 为交付必需。

### 实测结果

| 口径 | Task Duration | 有界管道 | `cube_utilization` |
|---|---:|---:|---:|
| 单算子（当前 host） | **110.4 µs** | aic_mac **81.1%** | **89.541%** |
| 组合流水线内（warm） | **122.11 µs** | — | — |
| 旧 host | — | aic_mac 79.9% | — |

正确性 6/6。这是换 host 后**唯一一个把结论从「接受缺口」推到「直接达标」的算子**：
`aic_mac` 从 79.9% 升到 81.1%——管道*比例*本身也改善了，这不是单纯时钟差异能产生的。

### 已接受的性能缺口

曾按 79.9%（差 0.1 个点）记为接受缺口；当前 host 上 **81.1% 已直接越过 80% 线**，无遗留缺口。
已登记但未采纳的杠杆：把向量相软件流水提前一个 tid。

## gdr_bwd_wy_reduce

Host entry `gdr_bwd_wy_reduce_wrapper`，kernel `gdr_bwd_wy_reduce_kernel`。

### 数学契约

WY 逆伴随的**纯 vector 半**：无 cube、无 matmul。每 `(b,h,i)` chunk 独立。
先做 S0 前导（M1 式纯向量）把 primary `v` 自取成 `v_c`：
`v_c = reshape(pad_T(cast_fp32(transpose(v, [B,T,H,V]→[B,H,T,V])), Tp=N*C), [B,H,N,C,V])`。
随后逐 chunk（`[:,None]` 表示把 `[C]` 列沿 free 轴广播）：

```
kbeta_de  = k_f32 * beta_act[:,None] * decay_exp[:,None]
dk_term1  = dkbg * (decay_exp * beta_act)[:,None]
db1       = sum_K(dkbg * k_f32 * decay_exp[:,None])
dg2_loop1 = sum_K(dkbg * kbeta_de)
dv        = dvb * beta_act[:,None]                    ← 输出
db2       = sum_V(dvb * v_c)                           # 用的是 v_c（来自 primary v），不是 v_b
db3       = sum_K(dkb * k_f32)
dk2       = dkb * beta_act[:,None] + dkt + dk_term1
dg2       = dg2_loop1 + beta_act * db3 − sum_K(k_f32 * dkt)      ← 输出
dbeta_wy  = db1 + db2 + db3                                      ← 输出
dk        = dk_p + dk2                                           ← 输出（b4b_dk 跨阶段累加）
```

**Scale 台账**（本算子最容易错的地方）：`scale = 1/sqrt(K)` **不在这里**（已折进上游）；
成对的 decay 门 `exp(g_i − g_j)` **完全住在 OP6a 的 glue-2 里**（已烘进 `dkb`/`dkt`），
OP6b **不得**对 `dkb`/`dkt` 再施加任何成对或逐行 decay。`beta_act` 出现在 **5 条不同路径**上
（`dv`；经 `dk_term1` 的 `dk`；经 `dkb` 的 `dk`；经 `kbeta_de` 的 `dg2`；经 `db3` 的 `dg2`），
每条**恰好一次**；`decay_exp` 出现在 **3 条路径**上（`dk_term1`、`db1`、`kbeta_de`），各一次。
`db3` 被 `dbeta_wy` 与 `dg2` 共用。

输入：`dkbg`/`dkb`/`dkt`/`k_f32`/`dk_p [B,H,N,64,K]`、`dvb [B,H,N,64,V]`、
`beta_act`/`decay_exp [B,H,N,64]`（均 fp32）、`v [B,T,H,V]` bf16（唯一带原始 T 轴的输入）。
输出：`dk [B,H,N,64,K]`、`dv [B,H,N,64,V]`、`dbeta_wy`/`dg2 [B,H,N,64]`，均 fp32。
容差 `atol/rtol = 1e-2`；`K, V ≤ 128`。唯一的 cast 是 `v` 的 bf16→fp32 加宽。

### Grid / tile

- **扁平 strided 任务循环** `for task in pl.range(core_id, B*H*N, num_cores)`，
  解码 `i = task%N; h = (task//N)%H; b = task//(N*H)`。chunk 完全独立、GM 切片不相交、
  计算量相同故天然负载均衡。纯 Vector kernel → 用 `vector_core_num`，
  `num_cores = min(vector_core_num, B*H*N)`，**launch 次数 1**。
- 常量：`C = 64`（= fp32 寄存器 lane 数）、`LANES = 64`、`MAX_K = 128`、`MAX_V = 128`、
  `COL_PAD = 128`（per-token 列 tile 的过声明宽度，保 `load_align` 不越界）。**无 SCALE**。
- 布局裁定为 **C 作行、K/V 作 lane**（layernorm 范式）；free 轴 >64 时按「两个 64 宽半」跨寄存器归约。
- **只有 UB 一个空间**（无 L1/L0A/L0B/L0C）。采用 **two-pass 缓冲复用**降峰值：
  Pass-K（K 域归约 + `dk` 装配）与 Pass-V（V 域归约 + `dv`）复用同一批大缓冲。
  规划峰值 Pass-K `198,144 B = 78.0%`；若把 `dv` 缓冲推迟到 Pass-V 复用 `dkbg` 的地址，
  峰值降到 `165,376 B = 65.1%`。
- **实测片上预算**：UB **71.6%**，mutex **10 default / 5 merge**，pipe-flag `0 / 0`，
  L0A/L0B **N/A**，不需要 merge。两段式排序是必要的——融合成一遍会到 97.4%。

### 实测结果

| 口径 | Task Duration | 有界管道 |
|---|---:|---:|
| 单算子（当前 host） | **51.552 µs** | aiv_mte2 **63.1%** |
| 组合流水线内（warm） | **39.77 µs** | — |
| 旧 host | ~54 µs | 实测 3.25 TB/s 拷贝上限的 41% |

纯 vector kernel（`Mix Block Num = 0`）**没有** `cube_utilization` 列，故口径取占优的 `aiv_*_ratio`。
正确性 6/6（含 B=2）。

### 已接受的性能缺口

**有，已接受**：`aiv_mte2` **63.1%**，低于 80% 管道线。它是**带宽受限**的——历史上按
「实测 3.25 TB/s 本机拷贝上限」的口径接受（旧 host：~54 µs = 该上限的 41%），优化轮次已穷尽。
需要说明的是：这个算子从未按 80% 管道口径判过，它的 63.1% 是照实报告，并**不**声称达标或未达标；
重算拷贝上限需要单独的 load/store-only sweep，不是一次 kernel profile 能给出的。

## gdr_bwd_finalize

Host entry `gdr_bwd_finalize_wrapper`，kernel `gdr_bwd_finalize_kernel`。

### 数学契约

最后一个算子，它的公开输出**就是**整个反向的交付物。纯 Vector L0，无 cube、无 matmul：

```
b4b_dg  dg_sum[i] = dg_p[i] + dg2[i]
b5      dg_full[i] = triu_incl @ dg_sum[i]        # 每 chunk 沿 C token 轴的后缀累加：
                                                  # dg_full[i][r] = sum_{c>=r} dg_sum[i][c]
                                                  # 前向 s5 下三角 cumsum 的转置对偶；逐 chunk 独立（[C,C]，非跨 chunk）
b6      dbeta[i] = dbeta_wy[i] * sig[i] * (1 − sig[i])      # 过 sigmoid logit 的链式法则
b7      去 padding + 转置 + cast：
        reshape [B,H,N,C,*] → [B,H,N*C,*] → 切 [:, :, :T] → 转置到公开 [B,T,H,*]
        dq/dk/dv 由 fp32 narrow 到 bf16；dg/dbeta 保持 fp32
```

输入：`grad_o [B,T,H,V]` bf16、`dq_p`/`dk_in [B,H,N,64,K]`、`dv_in [B,H,N,64,V]`、
`dg_p`/`dg2`/`dbeta_wy`/`sig`/`beta_act [B,H,N,64]`（均 fp32）。
输出（公开、真实长度 T）：`dq`/`dk [B,T,H,K]` bf16、`dv [B,T,H,V]` bf16、`dg`/`dbeta [B,T,H]` fp32。
**M10 没有 scale 因子。** `grad_o` 只作为公开布局与序列长度 `T` 的锚点被串进来，
其**数值从不被读**；`beta_act` 被声明但不使用（它与 `sig` 恒等，都是 `sigmoid(beta_logit)`）。
容差 `atol/rtol = 1e-2`；`K` 与 `V` 相互独立（允许 V<K，如 V=96 < K=128）。

`dg` 的后缀累加在**向量里**用寄存器级 Hillis-Steele 扫描实现（步长 1,2,4,8,16,32），逐 head 独立、
数值精确；不需要 cube 三角 matmul。

### Grid / tile

- **纯 Vector，BLOCK（ceil-div 连续段）分核**，work item = 一个 `(b, n, hg)` **head-group**：
  `task = b*(N*HG) + n*HG + hg`，`tiles_per_core = ceil(total_tasks/core_num)`，
  `total_tasks = B*N*HG`。`block_dim = min(vector_core_num, B*N*HG)`，**launch 次数 1**。
- 分组数由 host 纯 Python 推出（同时照顾 UB 预算与核占用率）：
  ```python
  H_TILE_MAX = 16
  HG0    = min(H, max(ceil(H / H_TILE_MAX), ceil(vector_core_num / (B * N))))
  H_TILE = ceil(H / HG0)
  HG     = ceil(H / H_TILE)        # 用真实 H_TILE 反推，由构造保证末组 h_valid >= 1
  ```
  生产形状 `B=1, H=8, T=2048`（N=32）：`HG = 2`、`H_TILE = 4`、`total_tasks = 1*32*2 = 64`。
  `HG0` 必须封顶在 `H`：旧公式 `HG = max(1, ceil(H/16), ceil(core_num/(B*N)))` 在 `B*N` 很小时
  会把 `HG` 推到远超 `H`（如 B=1,N=2,H=4 → 32），使 `hg ≥ H` 的组 `h_valid ≤ 0` 流进 `set_validshape`。
- 常量：`C = 64`、`LANES = 64`、`TW = 128`（wide tile 声明宽，运行时 `set_validshape` 收窄到 K 或 V）、
  `H_TILE_MAX = 16`（head-group 的**编译期**静态声明宽度，取 8 的倍数以满足 `pl.transpose` 的 32B 对齐）。
  `H_TILE` 是运行时标量，只控制内层循环上界与最终 store 的裁剪宽度，**不参与任何 tile 的静态形状声明**。
- 六类资源投影（worst-case 静态占用，与运行时 H/HG 无关）：UB **133,120 B / 253,952 B = 52.4%**；
  mutex **20 个**（0–19，10 组 × 2）= 62.5%；手工核内 pipe-flag **0**（全部交 `auto_mutex`）；
  cross-core event id **0**（不涉及跨核依赖）；L0A / L0B **0（N/A，无 matmul）**。
- `dg`/`dbeta` 的写出走 `[H_TILE_MAX, C] → pl.transpose → [C, H_TILE_MAX]`，再
  `pl.store(..., [b, t0, hg*H_TILE], order=[1,2])`——partition 轴（C）映到 strided 的 T，
  free 轴映到 GM 里真正连续的 H。这是**必须这样写**的，见下。

### 实测结果

| 口径 | Task Duration | 有界管道 |
|---|---:|---:|
| 调优前 → 调优后 | **18.4 µs → 15.0 µs** | — |
| 单算子（当前 host） | 33.514 µs | aiv_mte2 **73.5%** |
| 组合流水线内（warm，1049 µs 那一趟） | **26.87 µs** | — |
| 旧 host（round_006 `prod_scale`） | 126.367 µs | aiv_mte3 71.6% |

纯 vector（`Mix Block Num = 0`），无 `cube_utilization` 列。相对旧 host 约 **3.8×**。

### 已接受的性能缺口

**有，已接受**：有界管道 **66%**，与 `gdr_bwd_wy_reduce` 同属**带宽受限**，贴近实测的
**3.25 TB/s 本机拷贝上限**。同样地，它从未按 80% 管道口径判过，而是按拷贝上限口径接受。
已识别但未采纳的候选缓解（避免在未经真机验证前引入第三个未证实假设）：把 `hc_dg` + `hc_dbeta`
（或 `ch_dg` + `ch_dbeta`）合进同一物理 group 共享 mutex。

### 硬件行为注意事项（store 映射，影响所有算子）

这一条是用**错误数值结果**（而非编译错误）换来的，且不止影响本算子：

> **`pl.store` 的 `order` 只正确控制 partition 轴到目标维度的（可 strided）映射；tile 的 free 轴
> 永远被当作目标里物理连续的一段字节写入，不论 `order` 怎么声明。**

因此目标里**不连续**的维度（如 `[B,T,H]` 的 T，stride = H）**只能**由 tile 的 partition 轴承担；
free 轴只能对应目标里真正 stride=1 的维度（`[B,T,H]` 是 H，`[B,T,H,W]` 是 W）。
违反它的指纹很好认：`H=2, T=64` 探测时 128 个输出元素里 **126 个错**，且每个错值都精确等于
**另一个 `(t,h)` 组合**的正确值——`[1,C]` tile 的 64 个 C 值被当成 64 个连续 GM 字节顺序写下去，
每写满 H 个就跨进下一个 t。`[C,1]`（partition=C→T、free=1→H）与 `[C,W]`（partition=C→T、free=W→W）
一直是对的，`[1,C]` + `order=[0,1]` 则在 `H > 1` 时必错。

同族的教训还有：`torch.npu.set_device()` 必须在 launch 前调用（device≠0 时否则每次 DMA 打到
未映射地址，报 507015/507035 "timeout or trap"，与 kernel 死锁无法区分）；每个 `TileType` 都要声明
`valid_shape=[-1,-1]`，否则 `set_validshape` 静默无效；`make_tile_group` 的
`addrs`/`mutex_ids`/`depth` 都是**编译期**量，运行时开关缩不了 id 占用。

## 组合与验证

8 个算子在 Python 层按固定顺序组合，边界上**只有 GM 张量交接，没有任何 host 端算术**。
组合 harness：[`tests/ops/pypto_pro/gated_delta_rule/test_composition.py`](../../../../../../tests/ops/pypto_pro/gated_delta_rule/test_composition.py)。

**正确性**：8 个算子全部在 **Ascend950PR（a5）** 上通过。组合在 **5/5 个 shape 上 PASS**——
4 个 DESIGN §8 形状（`aligned_b1_t128_h4_k64_v64` / `ttail_b1_t100_h4_k64_v64` /
`t3c_b1_t192_h2_k128_v128` / `tail_vlt_b1_t200_h3_k128_v96`）加上生产形状
`production_b1_t2048_h8_k128_v128`（**B=1, T=2048, H=8, K=V=128**，seed 2024），
**全部 30 个边界张量对 torch autograd 干净**，6 个公开量 `o/dq/dk/dv/dg/dbeta` 全部通过 VJP oracle。
生产 case 走的是与 4 个 §8 形状**完全相同**的正确性 + 边界二分路径，没有新代码路径、没有放宽容差。

```bash
# 正确性（默认，不带任何 flag；5 个 case）
python tests/ops/pypto_pro/gated_delta_rule/test_composition.py

# 流水线计时（opt-in，默认关闭）
python tests/ops/pypto_pro/gated_delta_rule/test_composition.py --timing --warmup 3 --repeats 1
```

**性能**：流水线 device 总计 **1049 µs**，可由 `python test_composition.py --timing` 复现。
`--timing` 先跑 3 趟不计时的 warm-up（吸收 JIT 编译——它只发生在每个 kernel 的第一次 launch），
再计时 1 趟；整个进程（3 warm-up + 1 计时 = 每 kernel 4 次 launch）在**一次**
`msprof --aic-metrics=PipeUtilization` pass 下抓取，取每个 kernel `Op Name` 的**最后一次（第 4 次，全热）**
出现作为其 device Task Duration：

| 算子（kernel） | Task Duration (µs) |
|---|---:|
| `gdr_fwd_train_wy` | 200.54 |
| `gdr_fwd_train_scan` | 191.63 |
| `gdr_bwd_dv_local` | 47.79 |
| `gdr_bwd_reverse_scan` | 217.62 |
| `gdr_bwd_dqkw` | 202.78 |
| `gdr_bwd_wy_matmuls` | 122.11 |
| `gdr_bwd_wy_reduce` | 39.77 |
| `gdr_bwd_finalize` | 26.87 |
| **合计（8 个 device Task Duration 之和）** | **1049.10** |

同一趟计时的 host 侧墙钟总计是 **1923.34 µs**（对 8 次 launch 逐个 sync 括起来）。
`1049.10` 与 `1923.34` 的差额是**每次 launch 的 host 往返开销**，不是 device 空闲——
逐个括起来等于把这笔开销付 8 遍。两个数字是两种**明确标注**的口径，不做平均也不做调和。

**调优结果**（数学与容差未动）：

| 算子 | 调优前 → 调优后 |
|---|---|
| `gdr_fwd_train_scan` | 380.0 µs → **178.9 µs** |
| `gdr_bwd_reverse_scan`（B=4） | 856.6 µs → **643.0 µs** |
| `gdr_bwd_finalize` | 18.4 µs → **15.0 µs** |

**已接受的性能缺口**（80% 有界管道线，79.x 可接受）：

| 算子 | 口径 | 判据 |
|---|---|---|
| `gdr_bwd_dqkw` | aic_mac **63.4%** | 对照**实测 ~79.7% 的 fp32-MAC 上限**；杠杆（bf16 cube 操作数）是 SPEC 变更 |
| `gdr_bwd_wy_reduce` | **63.1%** | **带宽受限**，贴近实测 **3.25 TB/s** 本机拷贝上限 |
| `gdr_bwd_finalize` | **66%** | 同上，带宽受限、贴近同一拷贝上限 |

**完成判据**：「所有组件都绿」**不等于**父算子完成。完成 = 8 个组件各自 PASS **加上**
组合数值验证 PASS——各组件自己对，不保证接起来仍等于原算子，后者必须由组合门禁证明。

**逐算子取证路径**：

```
kal/<op>/                  Stage 3.5 KAL 预构建产物（设计记录与结构参考，非运行时代码）
docs/perf/<op>/            真机 msprof 归档（round_001… / round_newhost 等）
tests/ops/pypto_pro/gated_delta_rule/
    test_<op>.py           单算子验收
    <op>_golden.py         NPU golden
    <op>_golden_cpu.py     CPU FP32 golden
    test_composition.py    端到端组合（含 --timing）
    precision_compare.py   统一精度判据（check_precision）
```

**平台注记**：当前 host 自报 **28 cube / 56 vector** 核（旧 host 是 32/64，已下线）。
**2026-10-07 之前的全部性能数字都已过时**，不要与上表混用。`kal/` 依赖
`pypto-kernel-arch-lang` 插件提供的仿真器载荷，本仓库不提供它，该目录是设计记录而非运行时代码。
