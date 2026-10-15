# GDA-05 GDN/PGDN 性能：先 profile 再定目标（阶段一：测量，不优化）

- **2026-09-20 用户放行（D-PM-43）。2026-10-08 PK-06 合入（PR #139 → `cf2475d`，issue #95 close）后，
  PM 按承诺（issue #96 NOTE6051129132）补全本阶段范围/写集/验收，不再是骨架。**

- 波次 / SoC：W0 / a5。**本阶段（测量）PM 可直接 ASSIGN，不需要额外用户授权**——D-PM-43 已放行、
  依赖已满足；后续若从"测量"进入"优化"，优化目标需 PM 看过 profile 数据后另开任务定，仍不豁免真机验证（D-PM-34）。
- 优先级：`-` · 时限：24h（首轮）· 依赖：`PK-06`（已 done，PR #139 已合入 `main` @ `cf2475d08f4c7cdec005dddeedf19b5411d5fcf9`）
- 写集（本阶段，只允许新增，不改任何既有文件）：
  - `benchmarks/profile_gda05_*.py`（新建一个或几个，命名 assignee 自定，职责：跑六个既有实现、拆分 device/host 耗时）
  - `docs/research/gda05_profile.md`（profile 报告）

## 这条任务是怎么来的

D-PM-22 排期的最后一个阶段。来自 #89（外部需求，分诊后采纳；用户批准前是 gated）。PK-06 合入后，PM 在
issue #96（comment 6051129132）把 profile 阶段的范围/写集/基线/交付判据发给了申领人，本文件是把那条
NOTE 的内容正式同步进规格。

## 范围（本阶段：测量，不优化）

六个**既有**实现，**只测不改**：

| 任务 | 实现 | 入口 |
|---|---|---|
| GDA-01/02 | GDN chunk 前向（含 GQA/GVA 分组） | `ascend_fla/ops/gdn_chunk_fwd.py` |
| GDA-03 | GDN chunk 反向（含分组） | `ascend_fla/ops/gdn_chunk_bwd.py` |
| PK-03 | PGDN chunk 前向（ATK 预条件） | `ascend_fla/ops/pgdn_chunk_fwd.py` |
| PK-05 | PGDN chunk 反向（ATK 求导） | `ascend_fla/ops/pgdn_chunk_bwd.py` |
| GDA-04 | GDN decode（`fused_recurrent_gdn`，短步递推 S=1..16）——**不是"短 conv"，那是 GDN-2 的
  `gdn2_short_conv_decode`，GDN 本身没有这个单元** | `ascend_fla/ops/gdn_fused_recurrent.py` |
| PK-06 | PGDN decode（融合 recurrent，带 ATK 状态） | `ascend_fla/ops/pgdn_fused_recurrent.py` |

> **更正**：本任务更早一版 PM NOTE（issue #96, 6051129132）把 GDA-04 误标成"短 conv"。PM 已直接读
> `kernels/projects/a5/` 与 `ascend_fla/ops/` 核实：GDN 系列只有 `gdn_fused_recurrent`（短步递推），
> `*_short_conv_decode` 只存在于 GDN-2（`gdn2_short_conv_decode`，不在本任务范围内）。按这张表为准，
> 不按旧 NOTE 的措辞。

- 优先 BF16 与前向测量，但六项的完整范围（含各任务已支持的 FP32）都要测——不只测子集。
- device 侧耗时与 host 侧开销必须拆开报（`AGENTS.md` §6 铁律一），缓存键开销要单独测一遍、确认是
  O(1) 字典查找（不要让它混进 device 数字，也不要在热路径上读源码/算 hash）。
- 不碰 `reserved_paths`、不碰 GDN-2、不碰 A2（这条任务是 a5）。

## 测量方法（基线，不是优化）

- 复用各任务 DONE 时已有的真实形状（`docs/matrix/models.json`），用已合入版本，不自己重写一遍实现。
- 同卡**同步三明治**，baseline-only（没有 candidate，这阶段是测量不是优化）。
- `bd`（block_dim）扫描跨进程/跨 build 各自隔离跑——不要在同一进程里换 `bd` 复用缓存（`AGENTS.md`
  §6 铁律二 的坑，已经在 KDA 的 profile 上踩过两次）。
- 保留原始样本（不只是均值），device 区间 / host 调用开销 / 同步总耗时分开报。
- 运行时生成输入与独立参考；完整既有 workload 先跑，保持原任务的数值预算与跨 `bd` 逐位判据（结果本身
  不应该因为只测不改而变化，用这个当健全性检查）。

## 验收

- [ ] 六个实现各自的 device/host 耗时拆分，带原始样本（不只是均值）。
- [ ] 缓存键开销单独测一遍，确认 O(1)、没有混进 device 数字。
- [ ] 报告**不下优化建议或速度目标的结论**——只给数字；目标由 PM 看数据后另开任务定。
- [ ] `git diff --stat` 对 `ascend_fla/`、`kernels/`、`tests/` 为空；只新增写集里列的文件。
- [ ] 真机验证（D-PM-34）：本阶段测的就是真机数字，没有豁免理由。

## 明确不在范围内

- 不做任何优化改动（哪怕看起来很小）——本任务只测量，优化是下一个任务。
- 不扩到 GDN-2、A2、PGDN/PKDA 之外的算子。
- 不下"应该优化哪里"之类的结论性建议——这是 PM 看数据后另立任务做的事，混在本任务报告里会把"测量"
  和"建议"的账记混。
