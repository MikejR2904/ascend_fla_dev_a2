"""A2 (Ascend 910B3) bring-up: environment, built-in op-package coverage, device facts.

A2-10 brings the repository up on the A2 SoC and proves the runtime bridge matches the
aclnn harness bit-for-bit. This module carries the host-measurable parts:

* ``environment``  — SoC name, CANN identity (driver/compiler/opp ``version.info`` raw text,
  inner/compiler timestamps and sha256), and the built-in op-package kernel directory listing.
* ``coverage``     — each torch_npu compute op this repository relies on, tried on the device,
  recorded pass/fail with the raw error text. Nothing is inherited from an A5 machine
  (AGENTS.md §6: conclusions do not cross SoCs).
* ``device``       — cube/vector core counts and UB/L1/L0C capacities the ascriptor profile
  declares for the resolved SoC. Observation only until A2-11 (AGENTS.md §2).

Every run writes a JSON receipt whose first line carries the SoC + CANN identity, so a
number can never be read without the environment that produced it.

    ASCEND_RT_VISIBLE_DEVICES=<free card> PYTHONPATH=<ascriptor>/library:<repo> \\
        python benchmarks/a2/bringup.py --out benchmarks/a2/evidence/bringup/runs
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import platform as _py_platform
import traceback

import torch

REPO = pathlib.Path(__file__).resolve().parents[2]

#: Ascend install root — from the environment so no machine path is written into the repository;
#: only paths relative to it are recorded (AGENTS.md §5 / §10).
ASCEND_HOME = pathlib.Path(os.environ.get("ASCEND_HOME", "/usr/local/Ascend"))

#: torch_npu compute ops the KDA/GDN runtime bridge and host preparation depend on. Each is
#: tried on the device; ``strided_d2h`` is the ``Slice`` path AGENTS.md §5 says to re-measure.
_COVERAGE_OPS = (
    "randn", "zeros", "empty", "cast_bf16", "cast_fp16", "contiguous",
    "matmul", "einsum", "cumsum", "h2d", "d2h", "strided_d2h",
)


def _sha256(path: pathlib.Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _relative(path: pathlib.Path) -> str:
    """Path relative to ASCEND_HOME, so no machine-absolute path is recorded."""
    try:
        return str(path.relative_to(ASCEND_HOME))
    except ValueError:
        return path.name


def _scrub_paths(text: str) -> str:
    """Replace machine-absolute path prefixes in captured error text (AGENTS.md §5 / §10)."""
    import re
    return re.sub(r"(/tmp|/usr/local|/home|/workspace|/root)[^\s'\"]*", "<path>", text)


def _version_info(path: pathlib.Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    return {"path": _relative(path), "sha256": _sha256(path), "raw": text}


def _cann_root() -> pathlib.Path | None:
    for candidate in sorted(ASCEND_HOME.glob("cann-*")):
        if candidate.is_dir():
            return candidate
    fallback = ASCEND_HOME / "ascend-toolkit/latest"
    return fallback if fallback.exists() else None


def _opp_kernel_dir() -> pathlib.Path | None:
    env = os.environ.get("ASCEND_OPP_PATH")
    roots = [pathlib.Path(env)] if env else []
    cann = _cann_root()
    if cann:
        roots.append(cann / "opp")
    for root in roots:
        kernel = root / "built-in/op_impl/ai_core/tbe/kernel"
        if kernel.is_dir():
            return kernel
    return None


def probe_environment() -> dict:
    from ascend_fla import platform

    soc = None
    device_name = None
    try:
        device_name = torch.npu.get_device_name(0)
        soc = platform.resolve_soc()
    except Exception as error:  # noqa: BLE001 - report, do not assume
        soc = f"<unresolved: {error}>"

    cann = _cann_root()
    versions = {}
    if cann:
        for component in ("compiler", "opp", "runtime"):
            info = cann / component / "version.info"
            if info.is_file():
                versions[component] = _version_info(info)
    driver_info = ASCEND_HOME / "driver/version.info"
    if driver_info.is_file():
        versions["driver"] = _version_info(driver_info)

    kernel_dir = _opp_kernel_dir()
    op_package = {}
    if kernel_dir:
        op_package = {
            "path": _relative(kernel_dir),
            "socs": sorted(p.name for p in kernel_dir.iterdir() if p.is_dir()),
        }
        soc_dir = kernel_dir / "ascend910b"
        if soc_dir.is_dir():
            op_package["ascend910b"] = sorted(p.name for p in soc_dir.iterdir())

    return {
        "soc": soc,
        "device_name": device_name,
        "cann_versions": versions,
        "op_package": op_package,
        "ascriptor_revisions": _ascriptor_revisions(),
        "python": _py_platform.python_version(),
        "torch": torch.__version__,
    }


def _ascriptor_revisions() -> dict:
    """Full 40-char git revisions of the ascriptor library and kernels (A2 numbers are revision
    sensitive; A2-01 pinned library 90cfcdc / kernels b3b3f9c)."""
    import subprocess

    workspace = os.environ.get("ASCRIPTOR_WORKSPACE")
    revisions = {}
    for component in ("library", "kernels"):
        path = pathlib.Path(workspace) / component if workspace else None
        try:
            revisions[component] = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        except Exception:  # noqa: BLE001 - report absence, do not guess
            revisions[component] = None
    return revisions


def probe_torch_npu_coverage() -> dict:
    results = {}
    for op in _COVERAGE_OPS:
        try:
            _run_coverage_op(op)
            results[op] = {"ok": True}
        except Exception:  # noqa: BLE001 - the error text is the deliverable
            results[op] = {"ok": False, "error": traceback.format_exc(limit=2).strip()}
    return results


def _run_coverage_op(op: str) -> None:
    npu = torch.device("npu:0")
    if op == "randn":
        torch.randn(64, 128, device=npu)
    elif op == "zeros":
        torch.zeros(64, 128, device=npu)
    elif op == "empty":
        torch.empty(64, 128, device=npu)
    elif op == "cast_bf16":
        torch.randn(64, 128, device=npu).to(torch.bfloat16)
    elif op == "cast_fp16":
        torch.randn(64, 128, device=npu).to(torch.float16)
    elif op == "contiguous":
        torch.randn(64, 128, device=npu).transpose(0, 1).contiguous()
    elif op == "matmul":
        a = torch.randn(64, 128, device=npu)
        (a @ a.t()).cpu()
    elif op == "einsum":
        a = torch.randn(8, 64, 128, device=npu)
        torch.einsum("bik,bjk->bij", a, a).cpu()
    elif op == "cumsum":
        torch.randn(64, 128, device=npu).cumsum(dim=-1).cpu()
    elif op == "h2d":
        torch.randn(64, 128).to(npu)
    elif op == "d2h":
        torch.randn(64, 128, device=npu).cpu()
    elif op == "strided_d2h":
        torch.randn(64, 128, device=npu)[:, ::2].cpu()
    else:
        raise ValueError(f"unknown coverage op {op!r}")


def _profile_device_facts(soc: str) -> dict:
    """Core counts and capacities the ascriptor profile *declares* for ``soc``."""
    try:
        import ascriptor.devices.profiles as _profiles
        root = pathlib.Path(next(iter(_profiles.__path__)))
    except Exception as error:  # noqa: BLE001
        return {"error": f"ascriptor profiles unavailable: {error}"}
    device = {"a5": "950", "a2": "b3", "a3": "a3"}.get(soc, soc)
    profile_path = root / f"{device}.json"
    if not profile_path.is_file():
        return {"error": f"no profile for soc={soc} (device={device})"}
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    return {"profile_device": device, "cube_cores": profile.get("cube_cores"),
            "vec_cores": profile.get("vec_cores"), "capacities_kb": profile.get("capacities_kb")}


def _measured_device_facts() -> dict:
    """Core counts read from torch_npu at runtime (the physical chip, not the profile). Excludes
    the uuid deliberately — it identifies the machine (AGENTS.md §5)."""
    try:
        properties = torch.npu.get_device_properties(0)
    except Exception as error:  # noqa: BLE001
        return {"error": str(error)}
    return {name: getattr(properties, name, None) for name in
            ("cube_core_num", "vector_core_num", "multi_processor_count",
             "L2_cache_size", "total_memory")}


def probe_device_facts(soc: str) -> dict:
    """Compare the ascriptor profile's declared cores against the runtime-measured cores. The A5
    lesson is that they can disagree (profile 32/64 vs physical 28/56, a hardware-barrier deadlock
    past the physical count), so both are recorded and matched per field. Observation only (D-PM-30).
    """
    profile = _profile_device_facts(soc)
    measured = _measured_device_facts()
    comparison = {}
    if "error" not in profile and "error" not in measured:
        for profile_key, measured_key in (("cube_cores", "cube_core_num"),
                                          ("vec_cores", "vector_core_num")):
            comparison[profile_key] = {"profile": profile.get(profile_key),
                                       "measured": measured.get(measured_key),
                                       "match": profile.get(profile_key) == measured.get(measured_key)}
    return {"profile": profile, "measured": measured, "comparison": comparison,
            "note": "profile (ascriptor) vs measured (torch_npu runtime); physical block_dim "
                    "reach is the block_dim probe; observation only until A2-11"}


def _hash_tensor(tensor: torch.Tensor) -> str:
    # View as uint8 so bf16/fp16 (which numpy cannot represent) still hash by raw bytes.
    raw = tensor.contiguous().cpu().flatten().view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def _to_npu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.device.type == "npu" else tensor.to(torch.device("npu"))


#: Per-case trace of the bridge launches (kernel name + CompiledKernel.signature), so a receipt
#: can prove the bridge — not a silent fall-back to the harness — produced the outputs.
_BRIDGE_TRACE: list = []


def _bridge_launch_kernel(kernel, args: tuple, options: dict):
    """A ``launch_kernel``-compatible launcher that runs ``kernel`` through this repository's
    runtime bridge (``compile_kernel`` → in-process ``AclnnOp`` on NPU tensors) instead of the
    ascriptor aclnn harness. The aclnn HostSpec lists every GM symbolic dim as a scalar; those
    are derived from the tensor shapes (via ``spec[...]["dims"]``) and value scalars come from the
    float arguments. Outputs stay on the NPU so a chain of kernels runs zero-copy.
    """
    from ascend_fla.runtime.compile import compile_kernel

    compiled = compile_kernel(kernel, device=options["device"], block_dim=options["block_dim"],
                              backend=options["backend"])
    input_count, output_count = len(compiled.input_names), len(compiled.output_names)
    input_tensors = [_to_npu(t) for t in args[:input_count]]
    output_tensors = [_to_npu(t) for t in args[input_count:input_count + output_count]]

    # The kernel's explicit scalar parameters come first in scalar_names, in signature order, so
    # they map positionally to the scalar arguments; the compiler appends the remaining GM
    # symbolic dims, which are derived from the tensor shapes (via spec[...]["dims"]).
    dims: dict[str, int] = {}
    for entry, tensor in zip(compiled.spec.inputs + compiled.spec.outputs,
                             input_tensors + output_tensors):
        for symbol, extent in zip(entry["dims"], tensor.shape):
            if isinstance(symbol, str):
                dims[symbol] = int(extent)
    scalar_args = args[input_count + output_count:]
    scalars = dict(zip(compiled.scalar_names[:len(scalar_args)], scalar_args))
    for name in compiled.scalar_names[len(scalar_args):]:
        scalars[name] = dims[name]

    _BRIDGE_TRACE.append({"kernel": compiled.op.op_name, "signature": compiled.signature})
    compiled(dict(zip(compiled.input_names, input_tensors)), scalars,
             dict(zip(compiled.output_names, output_tensors)))
    return output_tensors[0] if output_count == 1 else tuple(output_tensors)


def emit_path_hashes(unit_dir: pathlib.Path, use_bridge: bool, block_dim: int, case_ids,
                     force_block_dim: bool = False) -> dict:
    """Run a unit's cases through one path — the runtime bridge or the aclnn harness — at one
    ``block_dim`` in the current process, returning ``{"hashes": {case:bd:output: sha256}, "trace":
    {case:bd: {launches, signatures}}}`` (the trace is filled for the bridge path only).

    One process per (path, block_dim) on purpose: CANN reads ``ASCEND_CUSTOM_OPP_PATH`` once, so
    the bridge and harness cannot share a process, and ``_claim_op_name`` allows one ``block_dim``
    per op name per process. For the bridge every vendor tree is registered up front, before the
    first execution (AGENTS.md §5). ``force_block_dim`` (block_dim probe only) widens the unit's
    declared domain in this process so a block_dim past {1,2} reaches the hardware instead of the
    unit's own ``ValueError``; the unit source is not changed.
    """
    import importlib
    import sys
    import tempfile

    for path in (str(unit_dir), str(REPO)):
        if path not in sys.path:
            sys.path.insert(0, path)
    unit = importlib.import_module("unit")
    runner = importlib.import_module("_unit_runner")
    contract = json.loads((unit_dir / "contract.json").read_text(encoding="utf-8"))
    if force_block_dim and hasattr(unit, "BLOCK_DIMS"):
        unit.BLOCK_DIMS = tuple(sorted(set(unit.BLOCK_DIMS) | {block_dim}))
    # Per-process scratch under the OS temp dir (not the repo tree): the aclnn harness builds into
    # out_dir, so parallel runs (different block_dim or device) must not share it or gmake corrupts.
    scratch = tempfile.mkdtemp(prefix=f"a2_bringup_bd{block_dim}_")
    options = dict(device="a2", backend="cce", block_dim=block_dim, launcher="aclnn",
                   board=None, out_dir=scratch, timeout=1800)
    if use_bridge:
        from ascend_fla.runtime.compile import compile_kernel
        # A unit exposes either _kernels() (a dict, e.g. the 5-kernel forward chain) or
        # _kernel() (one kernel, e.g. decode). Register every vendor tree before executing.
        kernels = list(unit._kernels().values()) if hasattr(unit, "_kernels") else [unit._kernel()]
        for kernel in kernels:
            compile_kernel(kernel, device="a2", block_dim=block_dim, backend="cce")
        runner.launch_kernel = _bridge_launch_kernel

    hashes = {}
    trace = {}
    for case in contract["cases"]:
        if case_ids and case["id"] not in case_ids:
            continue
        _BRIDGE_TRACE.clear()
        outputs = unit.execute(unit.make_inputs(case), dict(options))
        key = f"{case['id']}:bd{block_dim}"
        for name, value in outputs.items():
            hashes[f"{key}:{name}"] = _hash_tensor(value.cpu())
        if use_bridge:
            trace[key] = {"launches": len(_BRIDGE_TRACE),
                          "signatures": [entry["signature"] for entry in _BRIDGE_TRACE]}
    return {"hashes": hashes, "trace": trace}


def _spawn_path_hashes(unit_dir: pathlib.Path, path: str, block_dim: int, case_ids) -> dict:
    import subprocess
    import sys

    command = [sys.executable, str(pathlib.Path(__file__).resolve()), "--emit-hashes", path,
               "--bridge-unit", str(unit_dir), "--block-dims", str(block_dim)]
    if case_ids:
        command += ["--bridge-cases", ",".join(case_ids)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=3600,
                               env=os.environ.copy())
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("HASHES "):
            return json.loads(line[len("HASHES "):])
    raise RuntimeError(f"{path} subprocess emitted no hashes (rc={completed.returncode}):\n"
                       f"{completed.stdout[-1500:]}\n{completed.stderr[-1500:]}")


def probe_bridge(unit_dir: pathlib.Path, *, block_dims=(1, 2), case_ids=None) -> dict:
    """Compare the runtime bridge against the aclnn harness bit-for-bit over a unit's contract
    cases — same kernel, same inputs, so the bytes must match. Each (path, block_dim) runs in its
    own subprocess (see :func:`emit_path_hashes`). Returns ``comparisons`` (per ``case:bd:output``
    the bitwise verdict and both hashes) and ``bridge_trace`` (per ``case:bd`` the bridge launch
    count and kernel signatures, so the receipt proves the bridge ran, not a harness fall-back).
    """
    comparisons = {}
    bridge_trace = {}
    for block_dim in block_dims:
        harness = _spawn_path_hashes(unit_dir, "harness", block_dim, case_ids)["hashes"]
        bridge = _spawn_path_hashes(unit_dir, "bridge", block_dim, case_ids)
        bridge_trace.update(bridge["trace"])
        for key in harness:
            comparisons[key] = {
                "bitwise": harness[key] == bridge["hashes"].get(key),
                "harness_sha256": harness[key],
                "bridge_sha256": bridge["hashes"].get(key),
            }
    return {"comparisons": comparisons, "bridge_trace": bridge_trace}


def probe_block_dim(unit_dir: pathlib.Path, case_id: str, block_dims, timeout_s: int = 300) -> dict:
    """Try one case at each block_dim in its own subprocess, small first, widening the unit's
    declared domain so the launch reaches the hardware. A block_dim past the physical core count
    deadlocks on a hardware barrier, so on the first timeout record it and stop — do not retry and
    do not try larger values (AGENTS.md §5). Observation only until A2-11.
    """
    import subprocess
    import sys

    results = {}
    for block_dim in sorted(block_dims):
        command = [sys.executable, str(pathlib.Path(__file__).resolve()), "--emit-hashes", "bridge",
                   "--bridge-unit", str(unit_dir), "--block-dims", str(block_dim),
                   "--bridge-cases", case_id, "--force-block-dim"]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s,
                                       env=os.environ.copy())
        except subprocess.TimeoutExpired:
            results[block_dim] = {"ran": False, "timeout_s": timeout_s,
                                  "note": "timed out (likely hardware-barrier deadlock); stopped, not retried"}
            break
        ran = any(line.startswith("HASHES ") for line in completed.stdout.splitlines())
        tail = completed.stderr.strip().splitlines()[-1][:200] if completed.stderr.strip() else ""
        results[block_dim] = {"ran": ran, "returncode": completed.returncode,
                              "note": None if ran else _scrub_paths(tail or "no hashes emitted")}
    return results


def probe_claim_op_name(unit_dir: pathlib.Path) -> dict:
    """`_claim_op_name` guarantees one build per op name per process — building the same kernel at a
    second block_dim in one process must **raise**, not silently reuse the first. Compile at
    block_dim 1 then 2 and record that the second raises (the A2 behaviour the spec wants checked,
    not bypassed)."""
    import importlib
    import sys

    for path in (str(unit_dir), str(REPO)):
        if path not in sys.path:
            sys.path.insert(0, path)
    unit = importlib.import_module("unit")
    from ascend_fla.runtime.compile import compile_kernel

    kernel = list(unit._kernels().values())[0] if hasattr(unit, "_kernels") else unit._kernel()
    compile_kernel(kernel, device="a2", block_dim=1, backend="cce")
    try:
        compile_kernel(kernel, device="a2", block_dim=2, backend="cce")
    except Exception as error:  # noqa: BLE001 - raising is the expected, correct behaviour
        return {"op": getattr(kernel, "name", None), "second_build_raised": True,
                "error_type": type(error).__name__, "message": _scrub_paths(str(error).splitlines()[0])}
    return {"op": getattr(kernel, "name", None), "second_build_raised": False,
            "note": "second build did NOT raise [FAIL: _claim_op_name not enforced on a2]"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A2 bring-up: environment, coverage, device facts")
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "benchmarks/a2/evidence/bringup/runs")
    parser.add_argument("--bridge-unit", type=pathlib.Path,
                        help="run the runtime bridge vs aclnn harness bitwise comparison for this unit")
    parser.add_argument("--bridge-cases", help="comma-separated case ids (default: all in the contract)")
    parser.add_argument("--block-dims", default="1,2", help="comma-separated block_dim values to sweep")
    parser.add_argument("--emit-hashes", choices=("bridge", "harness"),
                        help="internal: run one path in this process and print HASHES <json> (see probe_bridge)")
    parser.add_argument("--block-dim-probe", metavar="CASE",
                        help="probe which block_dim values run for this case (needs --bridge-unit)")
    parser.add_argument("--block-dim-values", default="1,2,4,8,20,40",
                        help="comma-separated block_dim values for --block-dim-probe (small first)")
    parser.add_argument("--force-block-dim", action="store_true",
                        help="internal: widen the unit's declared block_dim domain (block_dim probe)")
    parser.add_argument("--claim-check", action="store_true",
                        help="check _claim_op_name: one build per op name per process (needs --bridge-unit)")
    args = parser.parse_args(argv)

    if args.emit_hashes:
        case_ids = args.bridge_cases.split(",") if args.bridge_cases else None
        emitted = emit_path_hashes(args.bridge_unit, args.emit_hashes == "bridge",
                                   int(args.block_dims), case_ids, force_block_dim=args.force_block_dim)
        print("HASHES " + json.dumps(emitted))
        return 0

    environment = probe_environment()
    receipt = {
        "receipt": _receipt_line(environment),
        "environment": environment,
        "coverage": probe_torch_npu_coverage(),
        "device_facts": probe_device_facts(environment["soc"]),
    }
    if args.claim_check:
        receipt["claim_op_name"] = probe_claim_op_name(args.bridge_unit)
    if args.bridge_unit and not args.block_dim_probe and not args.claim_check:
        receipt["bridge"] = probe_bridge(
            args.bridge_unit,
            block_dims=tuple(int(x) for x in args.block_dims.split(",")),
            case_ids=args.bridge_cases.split(",") if args.bridge_cases else None,
        )
    if args.block_dim_probe:
        receipt["block_dim_probe"] = probe_block_dim(
            args.bridge_unit, args.block_dim_probe,
            [int(x) for x in args.block_dim_values.split(",")],
        )

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.out / f"bringup_{stamp}.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")

    print(receipt["receipt"])
    failed = [op for op, r in receipt["coverage"].items() if not r["ok"]]
    print(f"coverage: {len(receipt['coverage']) - len(failed)}/{len(receipt['coverage'])} ok"
          + (f"; failed: {failed}" if failed else ""))
    print(f"device_facts: {receipt['device_facts']}")
    if "bridge" in receipt:
        comparisons = receipt["bridge"]["comparisons"]
        mismatches = [k for k, v in comparisons.items() if not v["bitwise"]]
        print(f"bridge: {len(comparisons) - len(mismatches)}/{len(comparisons)} bitwise-identical"
              + (f"; mismatches: {mismatches}" if mismatches else ""))
    if "block_dim_probe" in receipt:
        print(f"block_dim_probe: {receipt['block_dim_probe']}")
    if "claim_op_name" in receipt:
        print(f"claim_op_name: {receipt['claim_op_name']}")
    print(f"receipt written: {path.name}")
    return 0


def _receipt_line(environment: dict) -> str:
    compiler = environment.get("cann_versions", {}).get("compiler", {})
    first_line = compiler.get("raw", "").splitlines()[0] if compiler.get("raw") else "?"
    return (f"SoC={environment['soc']} device_name={environment['device_name']} "
            f"cann_compiler={first_line} sha256={compiler.get('sha256', '?')[:16]}")


if __name__ == "__main__":
    raise SystemExit(main())
