# A2 KDA forward qualification (A2-12)

This directory contains text evidence for the unchanged A2-03 stable forward unit,
run through the A2-10 runtime bridge. The public KDA entry remains unqualified;
its dispatch is outside this task. No kernel, backward, decode or A5 path changes
are part of this qualification. The PM explicitly confirmed this
[unit/public-entry boundary](https://github.com/ddddwee1/ascend_fla_dev/issues/39#issuecomment-5796851136).

## Input and comparison contract

The real Kimi-Linear dimensions come from `docs/matrix/models.json`:
B=1, H=HV=32, K=V=128, hidden size 2304, T in {64,128,256,512,4096}.
q/k are normalized on the CPU by the test generator, then supplied as BF16;
v is BF16; raw gate, beta and the nonzero initial state are FP32. Final state is
FP32 and output is BF16. FP32 q/k/v and block_dim outside {1,2} remain rejected.
This is runtime-generated input preparation for the existing unit ABI.

Default gates come from the local Kimi layer's randomly initialized A_log and dt
parameters, using eight seeds and CPU-generated hidden inputs with standard
deviation 0.5. The initialization matches the pinned FLA source. Calibrated gate
sweeps shift A_log by log(target_span/measured_span), preserving each seed's gate
distribution. The A_log and dt initialization distributions match FLA; the full
FLA layer was not instantiated, and matching a seed does not assert matching
projection weights. These are real **dimensions**, not trained model weights or
a model end-to-end evaluation. CUDA/Triton was not executed.

Outputs are compared with the CPU FP32 recurrence in `ascend_fla/reference/kda.py`.
That recurrence is checked against the literal pinned FLA `naive_recurrent_kda`,
and final qualification also checks the A2 unit's independent transposed-state
`bmm` recurrence against FLA (relative L2 <= 1e-5). The output and final-state
budgets are independently fixed at 0.05; finite status is reported separately.
Norms accumulate in FP64 after FP32 value conversion. A zero reference with a
nonzero residual fails rather than receiving an epsilon denominator.

C=1/HV32 is a positive case, per the PM correction on issue #39. The stable A2
implementation uses cube-local Aqk and does not inherit the earlier A5 upstream
fixed-slot lifetime defect. No rejection was added to manufacture that behavior.

## Range method

The protocol was frozen before the first scan: start at 8, double through 64,
then step by 32 until failure; refine the first failed interval to two span units.
For every measured condition retain the last good and first bad point. Choose
`floor(0.9 * min(first_bad) / 2) * 2`, capped by every observed last-good point,
then actually run the chosen guard across the declared lengths, seeds and both
block dimensions. The default-initialization upper bound must lie below this
guard. The A5 threshold is not an input to this procedure.

Nonfinite results above the eventual guard are retained as range observations.
The first affected exposed stage is recorded, along with all eleven tensor
hashes and stage finite flags. The CPU source-form exponent diagnostic is labeled
as such; it is neither a simulator run nor a captured internal device tensor.

## Execution and timing boundaries

The first native run used the complete T4096 workload through the original
aclnn harness before any simulation. It compiled and executed all five stages.
Later runs use the existing A2-10 in-process adapter; each block dimension has a
separate process and compiled signature. Every launch group acquires a shared
physical-device lock, rechecks health and occupation, isolates output, and holds
the lease until all children have drained. Machine identities and paths remain
in ignored external configuration.

Performance measures synchronized wall time from CPU-prepared inputs to device
outputs. Both candidate and torch_npu `kda_chunk_vectorized` baseline include
allocation and H2D; input generation, CPU references, D2H comparisons and initial
compilation are excluded. The unit adapter also copies NaN-seeded output buffers;
this is a measurement of the delivered validation adapter, not a production
public-op or kernel-only latency. Three baseline-before/candidate/baseline-after
sandwiches each use three warmups and twenty timed repetitions. Raw samples and
correctness checks are retained. No speedup claim follows from compilation or
whole-suite wall time.

## Source and toolchain receipts

The selected sources are library `90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5`,
kernels `b3b3f9c16df7c4626ed3c081032a1be5a753d0b1`, and FLA
`e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2`. They were clean at launch and the actual
library import and literal FLA file were checked against those selected trees.
The owner runner hash is checked against the unit's `runner-source.json`.

Executed environment: Ascend910B3, Python 3.11.15, torch 2.10.0+cpu with torch_npu
2.10.0, CANN 9.0.0, compiler and OPP timestamp `20260428_134817545`, built-in
packages `ascend910b` and `ascend910_93`. Environment receipts include version-file
hashes, source hashes, launch signatures and the repository revision.

`build/` preserves vendor/harness logs with mechanical path/device redactions.
Terminal control bytes and trailing whitespace are visibly escaped as `\xNN`;
no log lines, warning messages or numerical observations are removed.
`redaction-manifest.json` records original and published hashes. All fifteen
vendor builds completed. Their only warning is the unused
`CMAKE_CROSS_PLATFORM_COMPILER` preset while `ENABLE_CROSS_COMPILE=False`;
native GNU 11.4 and the ascend910b target were verified. No correctness,
synchronization or support warning was suppressed. No sim or pipesim acceptance
is claimed for this task.

During the torch_npu timing baseline, `zeros_like` emitted its one-time warning
that internal-format allocation was disabled and a base format would be used.
The matching version's [tensor factory source](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/aten/common/TensorFactories.cpp)
selects the base format and calls the NPU allocator on this branch. The local
package's `npu_config.py` maps `allow_internal_format` to that option. No format
option or warning filter was changed. Both baseline outputs passed the original
numeric checks before timing; the raw warning remains in the control-suite log.
Timing conclusions apply to this actual allocation configuration, without a
claim about alternative layouts.

## Reproduction

Use the accepted Python environment and pinned `ASCRIPTOR_WORKSPACE`; set
`FLA_KDA_NAIVE` to that checkout's KDA naive source. Select machine configuration
outside version control with `ASCRIPTOR_MACHINE_SPECS` and `ASCRIPTOR_BOARDS`.
Run the following under the shared device lease after a fresh health/occupation
check, with separate processes for each block dimension and isolated private
output directories:

```sh
python benchmarks/verify_real_shapes.py --check a2-unit --block-dim 1 \
  --a2-launcher bridge --a2-suite real --a2-lengths 4096 128 256 512 64 \
  --a2-seeds 2026 --a2-spans 8 --a2-output "$PRIVATE_OUTPUT/real-bd1"
python benchmarks/verify_real_shapes.py --check a2-unit --block-dim 2 \
  --a2-launcher bridge --a2-suite defaults --a2-lengths 4096 \
  --a2-seeds 0 1 2 3 4 5 6 7 --a2-output "$PRIVATE_OUTPUT/defaults-bd2"
python benchmarks/verify_real_shapes.py --check a2-unit --block-dim 1 \
  --a2-launcher bridge --a2-suite performance --a2-lengths 4096 128 \
  --a2-seeds 2026 --a2-spans 8 --a2-output "$PRIVATE_OUTPUT/performance-bd1"
```

Use `--a2-suite range` to retain expected out-of-domain failures. The individual
case receipts and boundary histories give every measured seed/length/span;
`RESULTS.md` reports the final guard and its validation matrix. The files in
`sources/` archive the exact scripts used, including preliminary runner versions;
the supported executable entry is the repository benchmark at its original
location, not those relocated provenance copies.

Host replay checks require no NPU and enforce complete case sets, separate
numeric/finite budgets, boundary history, all eleven cross-bd tensor hashes,
source identities, lease closeout and the untouched public-entry gate:

```sh
python -m pytest tests/test_kda_fwd_a2.py tests/test_platform.py -q
```

Final host regression: **1228 passed, 10 skipped, 5 warnings** in 159.97 seconds.
All five warnings are the existing pytest 8.3.2 `importorskip` deprecation for the
CPU-only torch_npu import stub used by the host suite; no warning filter was
applied. Hardware-only tests are skipped in that host environment. Native A2
validation above uses the real torch_npu package separately under device locks.
The matrix generator and PM board checks pass. AST scope checks confirm changes
are confined to the A2 capability value, the approved platform-test function
body, the additional A2 benchmark branch/tests, and this evidence directory.

The retained T4096/seed0/bd2 probe distinguishes the failure modes: at span
176 the exposed nonfinite tensors are `Aqk` and `o`, while `final_state` can still be
finite and accurate; at span 192, `strict`, `Akk`, `w`, `u` and final state are
also affected. Both output checks therefore remain mandatory. The measured
boundary does not imply that a finite final state is sufficient acceptance.
