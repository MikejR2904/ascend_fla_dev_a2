# A5K-03 scan diagnostics

Experimental investigation of the upstream KDA backward scan's `dh0` race.
These entry points are invoked explicitly; none is registered in public dispatch.
The continuous-slot copies are causal interventions, not a delivered production fix.
See the [research report](../../../../../docs/research/kda_scan_dh0_nondeterminism.md).

## Reproduce

Use the accepted library and kernels revisions in `derivation.json` and
`evidence/sources/accepted-runtime.json`. The native tools intentionally require
an immutable deployment: `repo/` beside `source-manifest.json`, and a verified
`BF08_NATIVE_ROOT` containing `library/`, `workspace/` and
`accepted-source-manifest.json`. The environment/import checks are shared with
`kda_prep/verify_backward_native.py`. Source manifests map relative file names to
SHA256; generate them from the selected checkout, including the diagnostic tools.

Machine paths, Docker configuration, accepted Python/Torch versions, cache and
temporary directories, device mapping and the task lock belong in ignored
configuration. Set `ASCRIPTOR_MACHINE_SPECS` and `ASCRIPTOR_BOARDS` from that
configuration. Take the task lock and check device health before setting
`BF08_EXTERNAL_DEVICE_LOCK=1`; that variable asserts a lock already held and does
not acquire one. Shared healthy devices are allowed for these runs.
Never reset a shared card or terminate another workload. Use a new output and
independent process for every `block_dim`.

Inside the assigned Docker environment, let `PYTHON` be its accepted interpreter,
`DIAG` this directory and `RESULTS` a fresh private output directory:

```bash
"$PYTHON" "$DIAG/replay_native.py" --output "$RESULTS/original-bd4" \
  --device-label control-c --block-dim 4 --trials 50
"$PYTHON" "$DIAG/intervene_native.py" --output "$RESULTS/intervention-bd4" \
  --device-label control-c --block-dim 4 --trials 50 --poison-outputs
"$PYTHON" "$DIAG/stress_native.py" --output "$RESULTS/stress-bd4" \
  --device-label control-c --block-dim 4 --trials 50
```

Repeat the intervention command in separate processes for bd1–3 and the other
healthy card, changing labels and outputs. Use `cross-env` for the independently
verified second CANN environment. The first intervention sweep omitted
`--poison-outputs`; the subsequent sweeps included it to match historical output
initialization. Each command compiles all 50 production vendors (nine backward)
and any three experimental vendors before the first custom launch. It then runs
the complete Kimi T4096 training verifier before diagnostic replays. Stress also
checks all three experimental scans at B1/HV32/C64/T4096 against the unchanged
CPU FP32 physical-stage budgets before reducing the workload.

The six unstressed families are B2/HV4/C3, B1/HV2/C3, B1/HV2/C2,
B1/HV1/C3, B2/HV4/C1 and B2/HV4/C2. Each has three variants, one anchor
and 50 measured calls. Stress uses B2/HV4/C3 and SPINS 0, 64, 256, 1024,
4096. Anchors are excluded from rates. Always inspect both repeatability and
differences from the unperturbed output: a repeated incorrect result is still
incorrect. All incorrect stress captures, including stable incorrect results,
are retained. The unmodified replay is not a substitute for the full verifier.

After full hardware execution, use `inspect_events.py --help` for bounded model
replays. Supply the captured scan-input tensor and its exact file SHA256, the
accepted runtime root/manifest, B/HV/C/bd and a fresh output. The reduced model
B1/HV2/C3/bd1 retains complete tiles, both AIVs and the failing BHV boundary.
`--variant ring` selects the slot intervention; `--stress --spins 64` adds the
common timing perturbation. Model results do not qualify hardware execution.

`counterfactual.py` analyzes the original FMT-02 captures on Torch CPU FP32;
`counterfactual_stress.py` analyzes the new stress captures. Both verify file
identities before loading. The latter accepts `--labels cross-env` for the second
environment. Neither runs an NPU kernel or introduces a numerical tolerance.

## Evidence

Start with `evidence/manifest.json`, then the research report's tables. Each
native job has its raw environment version lines/hashes, production and
experimental compilation receipts, full-workload checks and per-family results.
`source_manifest` references deduplicate repeated manifests; the canonical JSON
hash is explicitly labeled and is distinct from the formatted file hash in the
top-level manifest. All numeric values and output hashes are retained.

`evidence/provenance/preregistered-predictions.json` preserves the original H1/H2
text, including the approximate source line later corrected by the first model.
Private tensor/trace archives are addressed by hashes and verified by restoration;
machine configuration and raw tensors are not committed. Historical evidence,
new device results, CPU counterfactuals, emitted CCE and event models remain
separate. No performance or production-repair qualification is claimed.
