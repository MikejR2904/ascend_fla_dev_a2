# A2 (Ascend 910B3) bring-up

A2-10. Brings the repository up on the A2 SoC and proves the in-process **runtime bridge**
(`AGENTS.md` §4) matches the aclnn harness bit-for-bit on real hardware. Everything below is
measured on the 910B3; per `AGENTS.md` §2 / D-PM-30 these A2 numbers are **observation only**
until A2-11, and no A2 operator conclusion is drawn here. Receipts: `benchmarks/a2/evidence/`
(`platform/` from A2-02, `bringup/` here); each carries the SoC + CANN identity and the pinned
ascriptor revisions (library `90cfcdc`, kernels `b3b3f9c`). Nothing is inherited from an A5
machine (`AGENTS.md` §6: conclusions do not cross SoCs).

## 1. Environment and built-in op-package coverage

Measured by `benchmarks/a2/bringup.py` (`probe_environment`, `probe_torch_npu_coverage`):

- SoC: `torch_npu` device name `Ascend910B3`; `platform.resolve_soc()` → `a2`.
- CANN (paths relative to `ASCEND_HOME`, full sha256 in the receipts): driver `version.info`
  `25.5.1`, compiler `9.2.0-beta.1`, opp `9.2.0-beta.1`.
- Built-in op-package `…/tbe/kernel/ascend910b`: `ops_cv`, `ops_legacy`, `ops_math`, `ops_nn`,
  `ops_oam`, `ops_transformer`.
- torch_npu compute-op coverage (each tried on the device): **12/12 pass** — `randn`, `zeros`,
  `empty`, `cast` (bf16 and fp16), `contiguous`, `matmul`, `einsum`, `cumsum`, host-to-device,
  device-to-host, and strided device-to-host (`Slice`). So the host preparation and layout paths
  the bridge relies on are all available on this A2 machine, without falling back to an A5 result.

## 2. Runtime bridge vs aclnn harness (bit-for-bit)

The a2 units already run through the ascriptor **aclnn harness** (`run.py … --launcher aclnn`,
out-of-line build + separate process). This repository's deliverable is the **runtime bridge**:
`compile_kernel(device="a2", backend="cce")` builds each kernel into a resident CANN custom op
and calls it in-process through `torch.library` on NPU device tensors, zero-copy. The bridge had
never run on A2 before this task.

`bringup.py`'s `_bridge_launch_kernel` is a drop-in for the unit harness's `launch_kernel` that
routes each kernel through `compile_kernel`, and `probe_bridge` replays a unit's contract cases
through both paths and compares every output byte-for-byte (same kernel, same inputs → identical
bytes). Three A2 realities had to be handled, exactly as the task's known-traps note warns:

- **`ASCEND_CUSTOM_OPP_PATH` is read once** (`gaps.json` `opp-path-read-once`): every vendor tree
  must register before the first execution, and the bridge and harness cannot share a process. So
  each `(path, block_dim)` runs in its own subprocess; the bridge pre-registers all kernels first.
- **The aclnn HostSpec lists every GM symbolic dim as a scalar.** The kernel's explicit scalar
  parameters map positionally to the leading `scalar_names`; the compiler-appended symbolic dims
  (`BT`, `HVK`, …) are derived from the tensor shapes via `spec[...]["dims"]`.
- **bf16/fp16 outputs** are hashed through a `uint8` view, since numpy cannot represent them.

Result on the 910B3, across the full contract matrix of both a2 KDA units — the five-kernel
`kda_fwd_stable` chain (20 cases) and the single-kernel `kda_fused_recurrent` decode unit (8
cases), each at `block_dim ∈ {1, 2}` — **every output is bit-for-bit identical** between the
in-process bridge and the aclnn harness (matching sha256 per output). That is **152/152**
comparisons: forward 20 cases × 3 outputs (`o`, `final_state`, `g_cumsum`) × 2 block_dims = 120,
and decode 8 cases × 2 outputs (`o`, `final_state`) × 2 block_dims = 32. Per-case verdicts and
both hashes are in `evidence/bringup/{fwd,decode}_bd{1,2}.json`, with the scrubbed console in the
matching `.log`. Each receipt also carries a **bridge trace** (`probe_bridge`'s `bridge_trace`):
per case, the number of in-process kernel launches (forward 5, decode 1) and one op signature per
launch — positive proof the bridge path actually executed, not a silent fall-back to the harness.
`tests/test_a2_bringup.py` guards that every recorded comparison is bitwise, that the matrix covers
every contract case × output × block_dim, that the trace shows the expected launch counts, and that
no receipt leaks a machine path. So the runtime bridge — never run on A2 before — drives both a2
KDA units in-process, zero-copy, indistinguishable from the harness on this hardware.

## 3. Device facts: profile vs measured (observation only)

The ascriptor `b3` profile (`family: a2`, `arch: c220`) declares **20 cube / 40 vector cores**,
UB 192 KB, L0C 128 KB, L1 512 KB, L0A/L0B 64 KB. `probe_device_facts` reads back the runtime's own
view via `torch.npu.get_device_properties(0)` and compares: measured `cube_core_num = 20` and
`vector_core_num = 40` **match** the profile's cube/vector counts (`evidence/bringup/*.json`
`device_facts`). The runtime view carries no identity fields — the machine `uuid` is deliberately
excluded. This is a consistency check between the compiler profile and the driver, not an A2
capability conclusion (D-PM-30); it holds on this machine and is not inherited from A5.

`probe_block_dim` (`evidence/bringup/block_dim_probe.json`) then asks a separate question: how far
does `block_dim` physically reach. Both a2 KDA units *declare* `block_dim ∈ {1, 2}` and reject
higher values cleanly at the unit boundary before any launch — so the bridge matrix above verifies
correctness only at 1 and 2. To probe the hardware past that declared domain, the probe widens
`unit.BLOCK_DIMS` and runs `kda_fwd_stable` at `block_dim ∈ {1, 2, 4, 8, 20, 40}`, each in its own
subprocess. On this 910B3 **all six run to completion (return code 0), none deadlocks** — including
40, the full vector-core count. That is the honest observation: this A2 does *not* reproduce the
A5 deadlock-past-cores behaviour. It is a **liveness** result only — the probe records that each
launch returned, not that its outputs are correct; correctness beyond the declared `{1, 2}` is
unverified and is A2-11's territory. None of these numbers is an A2 block_dim or capability
conclusion.

## 4. One build per op name per process

The bridge builds each kernel into a resident CANN custom op keyed by op name, and CANN reads
`ASCEND_CUSTOM_OPP_PATH` once per process (§2). `runtime.binding._claim_op_name` enforces the
matching invariant: the **first** build of an op name owns it for the process, and a **second**
build of the same name from a different vendor tree raises `AclError` rather than silently reusing
the first. `probe_claim_op_name` exercises this on the real gate kernel (`evidence/bringup/
claim_op_name.json`): building `KdaSub1GateA2Kernel` at `block_dim 1` then again at `block_dim 2`
in one process raises `AclError` with a first-line-only, path-scrubbed message. `test_a2_bringup.py`
additionally asserts the host-level guard directly (`_claim_op_name` on two vendor trees). This is
why every bridge sweep above runs one `(path, block_dim)` per subprocess.
