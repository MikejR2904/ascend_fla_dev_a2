# KAL pre-build artifacts

These are **not** part of the runtime package and nothing in `pypto_gym` imports
them. They are the hardware-architecture-language (KAL) pre-build of
`gated_delta_rule_fwd`: the kernel written against the `easyasc` a5 facade,
checked in a Python simulator against a torch reference, then run through the
backend that transpiles it to PyPTO-Pro DSL. The shipped kernel
(`../gated_delta_rule_fwd_impl.py`) was written from these; they are kept as the
design record and as a structural reference for the backward and training-forward
operators, which reuse the scan's realisation.

These artifacts were produced with the **`pypto-kernel-arch-lang` plugin**
introduced in upstream PR **!502**
(`[WIP]feat(cannbot-skills): 新增官方插件 pypto-kernel-arch-lang`), which is still
open against `master`. Running them needs that plugin's `easyasc` simulator
payload on `sys.path`; it is not vendored here and is not on `master` yet, so
until !502 merges these files are read-only design records rather than anything
you can execute from this checkout.

## Files

| file | what it is | status |
|---|---|---|
| `gated_delta_rule_fwd_kal_chunk_prep.py` | KAL source for the chunk-prep phase — preprocess, gram, WY, u/w. Entry `run_chunk_prep`, kernel `gated_delta_rule_fwd_chunk_prep_kernel`. | **sim-verified** against `golden_chunk_prep.py` |
| `gated_delta_rule_fwd_kal_fused.py` | KAL source for the whole operator, chunk-prep plus the inter-chunk scan. Entry `run_fused`, kernel `gated_delta_rule_fwd_fused_kernel`. | scan sim-verified and transpile-clean; **full single-kernel fusion does not pass** — see below |
| `staged_chunk_prep.py` | backend output: chunk-prep transpiled to PyPTO-Pro DSL | transpile-clean |
| `staged_scan.py` | backend output: the scan transpiled to PyPTO-Pro DSL, kernel `scan_kernel` | sim-verified **and** transpile-clean (248064 B UB, 5888 B headroom; zero `Vec _nz` tiles) |
| `golden_chunk_prep.py` | torch reference the chunk-prep sim compares against | |
| `golden_fused.py` | torch reference for the full operator | |
| `NOTES.md` | the full pre-build findings, including both boundaries and their resolutions | |

## Why the fused KAL file does not pass

Two boundaries were hit while integrating the scan onto chunk-prep in one kernel.
`NOTES.md` has the detail; in short:

- **Resolved** — the scan's transpile gap and UB overflow. The clean simulator
  realisation bridged matmul operands UB→L1, which the transpiler rejects
  (a transposed L0A operand must have its layout chosen at the GM→L1 load) and
  which blew the UB budget at 403712 B. Routing the bridge differently and
  pre-transposing the operand fixed both: `staged_scan.py` now fits in 248064 B
  and transpiles clean.
- **Not resolved** — the UB budget of the *combined* kernel. Chunk-prep's WY
  tiles together with the scan's persistent `S[K,V]` and `Sdelta[K,V]` exceed
  253952 B. The pooling that does fit reuses chunk-prep tile memory for the scan
  and forces a nested loop that reads uninitialised memory on the first chunk.

So the full fusion was completed by hand in Stage 4 instead, and the shipped
kernel solves the same budget problem a different way: it carries `S` through a
GM workspace rather than keeping it resident on chip, which is also what lets
K and V reach 256.

**A simulator pass is not an acceptance.** These artifacts were never the
deliverable; the real-hardware result for the shipped kernel is in
`../docs/perf/` and in the acceptance tests under
`tests/ops/pypto_pro/gated_delta_rule/`.
