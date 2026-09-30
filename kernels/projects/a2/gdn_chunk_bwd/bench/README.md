# A2-K1 GDN chunk backward — latency / utilization benchmark vs fla Triton

Mirrors the gdn2 methodology (`kernels/projects/a2/gdn2_recurrent_train/BENCHMARK_TRITON.md`):
the fla Triton kernels run on the **same Ascend 910B3** via the integrated
triton-ascend backend (no GPU), so the a2 GDN chunk backward and fla's
`chunk_gated_delta_rule` backward are the same op on the same device, compared
like-for-like. `gated_delta_rule` (scalar per-head gate delta rule) is exactly what
`gdn_chunk` implements — the correct Triton baseline.

## Files

| file | what |
|---|---|
| `bench_a2_bwd.py`   | compiles the a2 backward chain (device `a2`, block_dim 40) and times it; the `msprof` target for a2 device time |
| `bench_triton_gdr.py` | fla `chunk_gated_delta_rule` fwd / fwd+bwd on the NPU; the Triton baseline (backward-only = fb − fwd) |
| `compare_bwd.py`    | runs both backwards on identical inputs — per-gradient rel-L2 correctness + host-wall speedup |
| `msprof_bwd.sh`     | msprof Task Duration → per-call **device µs** for a2 and Triton |
| `pipe_bwd.sh`       | msprof PipeUtilization → per-pipe ratios, to drive the bounding pipe past **>90%** |

## Environment (910B3 box, verified)

```
source /usr/local/Ascend/cann-9.2.0-beta.1/set_env.sh
export ASCRIPTOR_WORKSPACE=/workspace/ascriptor-ws          # library revision 90cfcdc (A2-01 pin)
export FLA_ROOT=/workspace/pypto-gym/references/flash-linear-attention
export PYTHONPATH="$ASCRIPTOR_WORKSPACE/library:<repo>:$FLA_ROOT"
export ASCEND_RT_VISIBLE_DEVICES=<a free card>              # pick one at <50% AICore via npu-smi
PY=/usr/local/python3.11.15/bin/python3                     # has triton 3.5 + triton_ascend
```

Import `torch, torch_npu` before `fla` (the drivers do). fla is pinned 0.6.0
(triton-ascend); correctness is judged by a2-vs-Triton grad rel-L2 in `compare_bwd.py`.

## Conventions (fairness)

- Raw (un-normalized) q/k on both sides; external `scale = 128**-0.5` (a2 applies it
  in-kernel via the `scale` param, fla via `scale=`); `use_qk_l2norm_in_kernel=False`.
- `beta` in (0,1) used directly (`use_beta_sigmoid_in_kernel=False`); `g` a log-decay ≤ 0.
- `HV == H` (fla has no grouped-value path); the a2 GVA ratio is 1 here, so
  `group_reduce` is a copy. GVA (HV > H) is an a2 extension, benchmarked separately.
- Eager vs eager. (NPU-Graph vs NPU-Graph can be added as gdn2 did, to strip host
  dispatch — see BENCHMARK_TRITON.md.)

## Run

```
python compare_bwd.py --shape s512            # correctness + wall speedup
./msprof_bwd.sh                               # device µs, a2 vs Triton (SHAPES/ITERS env)
SHAPE=s2048 ./pipe_bwd.sh                      # per-pipe utilization of the a2 backward
```

## Status / open items

- **Gated on a free 910B3 card.** The functional simulator gives no latency or
  utilization; these numbers need the device. Written and ready; not yet run on hardware.
- The measured backward here is the **current shipped 8-kernel chain**
  (`replay → reverse_rec → reverse_dq → reverse_dbeta → reverse_dg → reverse_dk_bz →
  reverse_dk_ddr → group_reduce`) — the pre-optimization baseline. That decomposition
  was forced by the *simulator's* sync-credit ceiling, not the hardware; it re-reads the
  two `[B,HV,T,128,128]` GM tapes (`tape_d`, `back_tape`, ~0.5 GB each at B1/HV4/T2048)
  across five stages, so it is expected to be MTE2-bound and well under the >90% bar. The
  optimization target is to collapse it toward the DESIGN.md lean structure (checkpoints →
  single reverse scan, two live `[128,128]` tiles) and eliminate the tape round-trips.
- Verified card-free: the `compile_kernel(device="a2")` CCE build of **all 8 stages**
  succeeds (replay 1.5s, reverse_rec 164s, others ~33s each) — the decomposition builds
  where the earlier monolithic reverse timed out. Compile is a one-time process-cache cost
  paid at bench startup, then every timed iteration reuses it.
- Remaining on-device checkpoints (need a card): the `scale` float scalar binding in the
  `reverse_rec` / `reverse_dq` scalar dict at execution; and that fla's `dg`/`dbeta` grad
  parametrization matches the a2 adjoint (resolve any convention gap when first run).
