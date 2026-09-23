"""A2-01: minimal reproducer for ascriptor's A2-family split-K FP32 cube defect (library defect M10-081).

What the defect is (ascriptor library ``docs/defects/M10-081-a2-family-fp32-mmad-settle.md``): on the c220
A2 family (910B, 910_93) two short FP32 ``cube.mmad`` that update the same L0C are not interlocked by the
hardware. The second, ``is_init=False`` MMAD can read the accumulator before the first one has settled, and
the result is silently wrong. The functional model and the pipe model execute it exactly, so only the device
shows it. ``matmul(..., splitk=S)`` expands into one MMAD per K fragment, which is where it was found
(M16/N64/K128, split-K16). The library repair (in the pinned revision) inserts ``barrier(Pipe.M)`` after each
MMAD of the A2-family **FP32** split-K expansion only; hand-written chains of separate ``matmul``/``mmad`` calls
into one L0C get a lowered-IR lint (``trap`` channel, printed to stderr) and no automatic barrier.

This script builds tiny cube kernels for both shapes of the construct and runs them against an independent
CPU float64 reference:

* ``splitk_*``  one ``matmul(..., splitk=S)``: the original M10-081 case, its isolation shape, the KDA
  ``intra.py`` shape (M64/N64/K128, split-K64), and BF16/FP16 operand variants the settle rule does not cover.
* ``chain_*``   T separate ``matmul`` calls into one L0C (first ``is_init=True``, then ``is_init=False``),
  with (``bar``) or without (``nobar``) ``barrier(Pipe.M)`` between them. ``m16_n16_k16_t2/t3`` in FP32 is the
  exact construct of the KDA/GDN triangular inverse (``BLOCK=16``). ``t1`` is the single-MMAD control.

Every operand is a bounded dyadic value (eighths in [-1, 1], bias in quarters), so the product and every
partial sum are exact in FP32, BF16 and FP16 inputs alike. The comparison is therefore **bitwise**; any
deviation is a wrong result, not rounding. Outputs are pre-filled with -777 so an unwritten tile cannot pass.

    PYTHONPATH=<ascriptor>/library python benchmarks/a2/repro_splitk_fp32.py --profile a2 --launcher reference
    PYTHONPATH=<ascriptor>/library python benchmarks/a2/repro_splitk_fp32.py --profile a2 --launcher sim
    PYTHONPATH=<ascriptor>/library python benchmarks/a2/repro_splitk_fp32.py --profile a5 --launcher pipesim
    ASCEND_RT_VISIBLE_DEVICES=<card> PYTHONPATH=<ascriptor>/library \\
        python benchmarks/a2/repro_splitk_fp32.py --profile a2 --launcher aclnn --repeat 5

``--case`` takes ids or glob patterns (``--list`` prints them). ``--launcher aclnn`` builds each kernel as an
aclnn custom operator with the local CANN (ascriptor ``OpExec``) and runs it on logical device 0; it is only
accepted for the ``a2``/``a3`` profiles. Receipts (one JSON per case plus ``summary.json``) go to ``--out``
(default ``tmp/A2-01/runs``); they hold case data, hashes and versions, never host names or absolute paths.

Exit status: 0 when every ``bitwise``-expectation case matched bitwise in every run; 4 otherwise. ``probe``
cases (unsettled chains, BF16/FP16 split-K) have no expectation: they are what this script measures.
"""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import hashlib
import importlib.util
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
import traceback

import torch

REPO = pathlib.Path(__file__).resolve().parents[2]

#: operand dtype -> (GM alias, DT name, torch dtype)
DTYPES = {
    "f32": ("f32", "float", torch.float32),
    "bf16": ("bf16", "bfloat16", torch.bfloat16),
    "f16": ("f16", "float16", torch.float16),
}
SEED = 103609  # the M10-081 seed; each case derives its own stream from it
POISON = -777.0


@dataclasses.dataclass(frozen=True)
class Case:
    id: str
    kind: str  # "splitk" | "chain"
    dtype: str
    M: int
    N: int
    K: int
    splitk: int = 0
    bias: bool = False
    terms: int = 1
    barrier: bool = False
    expect: str = "bitwise"  # "bitwise" | "probe"
    why: str = ""
    jitter: int = 0  # A2-11 timing perturbation: independent MMADs into a scratch L0C between chain terms


def build_cases() -> list[Case]:
    cases = [
        Case("splitk_f32_m16_n64_k128_s16_bias", "splitk", "f32", 16, 64, 128, splitk=16, bias=True,
             why="original M10-081 case (8 K fragments + bias); a2: settled by the pinned desugar rule"),
        Case("splitk_f32_m16_n64_k128_s16", "splitk", "f32", 16, 64, 128, splitk=16,
             why="original shape without bias (also runs on the a5 profile)"),
        Case("splitk_f32_m16_n64_k32_s16", "splitk", "f32", 16, 64, 32, splitk=16,
             why="M10-081 isolation shape: two K fragments"),
        Case("splitk_f32_m64_n64_k128_s64", "splitk", "f32", 64, 64, 128, splitk=64,
             why="KDA kda_fwd(_stable)/intra.py shape: FP32 [64,128]x[64,128]^T, split-K 64"),
        Case("splitk_bf16_m16_n64_k128_s16", "splitk", "bf16", 16, 64, 128, splitk=16, expect="probe",
             why="BF16 operands: the settle rule is FP32-only, so no barrier is inserted"),
        Case("splitk_f16_m16_n64_k128_s16", "splitk", "f16", 16, 64, 128, splitk=16, expect="probe",
             why="FP16 operands: the settle rule is FP32-only, so no barrier is inserted"),
    ]
    # BF16/FP16 split-K boundary sweep: 2 fragments (no L0 slot reuse in the two-slot ring) .. 8 fragments,
    # at the failing M16, at M32 (M10-081's passing FP32 control) and at M64 (the KDA/GDN chunk size).
    for dt in ("bf16", "f16"):
        for m in (16, 32, 64):
            for k in (32, 48, 64, 128):
                cid = f"splitk_{dt}_m{m}_n64_k{k}_s16"
                if any(c.id == cid for c in cases):
                    continue
                cases.append(Case(cid, "splitk", dt, m, 64, k, splitk=16, expect="probe",
                                  why=f"{dt} split-K sweep: {k // 16} K fragments, M{m}"))
    for dt in ("f32", "bf16", "f16"):
        # hand-written twin of splitk_<dt>_m16_n64_k128_s16: eight K16 MMADs into one [16,64] L0C. With the barrier
        # it is exactly what the FP32 settle rule emits; without it, what the split-K expansion emits for BF16/FP16.
        for bar in (False, True):
            cases.append(Case(f"chain_{dt}_m16_n64_k16_t8_{'bar' if bar else 'nobar'}", "chain", dt, 16, 64, 16,
                              terms=8, barrier=bar, expect="bitwise" if bar else "probe",
                              why=f"split-K twin: 8 K16 matmuls into one L0C, "
                                  f"{'barrier(Pipe.M) between' if bar else 'no barrier'}"))
    for dt in ("f32", "bf16", "f16"):
        cases.append(Case(f"chain_{dt}_m16_n16_k16_t1", "chain", dt, 16, 16, 16, terms=1,
                          why="single MMAD control (no accumulate)"))
        for m in (16, 32, 64):
            for t in (2, 3):
                for bar in (False, True):
                    what = "KDA/GDN triangular-inverse construct" if (dt == "f32" and m == 16) else "accumulate chain"
                    cases.append(Case(
                        f"chain_{dt}_m{m}_n16_k16_t{t}_{'bar' if bar else 'nobar'}", "chain", dt, m, 16, 16,
                        terms=t, barrier=bar, expect="bitwise" if bar else "probe",
                        why=f"{what}; {t} matmuls into one L0C, {'barrier(Pipe.M) between' if bar else 'no barrier'}"))
    return cases


# --------------------------------------------------------------------------------------------- kernels


def kernel_name(case: Case, block_dim: int = 1) -> str:
    return "a201_" + case.id + (f"_bd{block_dim}" if block_dim != 1 else "")


def kernel_source(case: Case, profile: str, block_dim: int = 1) -> str:
    gm, dt, _ = DTYPES[case.dtype]
    M, N, K = case.M, case.N, case.K
    out = [f"import ascriptor.{profile} as api", "", "", f"@api.kernel(mode='cube', block_dim={block_dim})"]
    if case.kind == "splitk":
        params = [f"x: api.GM[api.{gm}, ({M}, {K})]", f"y: api.GM[api.{gm}, ({N}, {K})]"]
        if case.bias:
            params.append(f"bias: api.GM[api.f32, (1, {N})]")
        params.append(f"z: api.GM[api.f32, ({M}, {N})]")
        out.append(f"def {kernel_name(case, block_dim)}({', '.join(params)}):")
        body = [f"l1x = api.Tensor(api.DT.{dt}, [{M}, {K}], api.Position.L1)",
                f"l1y = api.Tensor(api.DT.{dt}, [{N}, {K}], api.Position.L1)"]
        if case.bias:
            body.append(f"l1b = api.Tensor(api.DT.float, [1, {N}], api.Position.L1, layout=api.Layout.ND)")
        body += [f"l0c = api.Tensor(api.DT.float, [{M}, {N}], api.Position.L0C)", "with api.auto_sync():",
                 "    l1x <<= x[:, :]", "    l1y <<= y[:, :]"]
        if case.bias:
            body.append("    l1b <<= bias[:, :]")
        extra = ", bias=l1b" if case.bias else ""
        body += [f"    api.matmul(l0c, l1x, l1y, splitk={case.splitk}, m={M}, n={N}, k={K}, is_init=True{extra})",
                 "    z[:, :] <<= l0c", "return z"]
    else:
        params = []
        for i in range(case.terms):
            params += [f"a{i}: api.GM[api.{gm}, ({M}, {K})]", f"b{i}: api.GM[api.{gm}, ({N}, {K})]"]
        params.append(f"z: api.GM[api.f32, ({M}, {N})]")
        out.append(f"def {kernel_name(case, block_dim)}({', '.join(params)}):")
        body = []
        for i in range(case.terms):
            body += [f"l1a{i} = api.Tensor(api.DT.{dt}, [{M}, {K}], api.Position.L1)",
                     f"l1b{i} = api.Tensor(api.DT.{dt}, [{N}, {K}], api.Position.L1)"]
        body.append(f"l0c = api.Tensor(api.DT.float, [{M}, {N}], api.Position.L0C)")
        if case.jitter:
            # scratch accumulator for the A2-11 timing perturbation: independent MMADs issued between the two
            # target MMADs delay the second one's read of l0c by a tunable, pipeline-realistic amount, without
            # changing z (l0c2 is never stored). Each is is_init=True (an overwrite, no accumulate hazard of its own).
            body.append(f"l0c2 = api.Tensor(api.DT.float, [{M}, {N}], api.Position.L0C)")
        body.append("with api.auto_sync():")
        for i in range(case.terms):
            body += [f"    l1a{i} <<= a{i}[:, :]", f"    l1b{i} <<= b{i}[:, :]"]
        for i in range(case.terms):
            body.append(f"    api.matmul(l0c, l1a{i}, l1b{i}, m={M}, n={N}, k={K}, is_init={i == 0})")
            if i + 1 < case.terms:
                for _ in range(case.jitter):
                    body.append(f"    api.matmul(l0c2, l1a0, l1b0, m={M}, n={N}, k={K}, is_init=True)")
                if case.barrier:
                    body.append("    api.barrier(api.Pipe.M)")
        body += ["    z[:, :] <<= l0c", "return z"]
    out += ["    " + line for line in body]
    return "\n".join(out) + "\n"


def load_kernel(case: Case, profile: str, src_dir: pathlib.Path, block_dim: int = 1):
    src = kernel_source(case, profile, block_dim)
    src_dir.mkdir(parents=True, exist_ok=True)
    path = src_dir / f"{kernel_name(case, block_dim)}.py"
    if not path.exists() or path.read_text() != src:
        path.write_text(src)
    spec = importlib.util.spec_from_file_location(f"a201_{profile}_{kernel_name(case, block_dim)}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, kernel_name(case, block_dim)), hashlib.sha256(src.encode()).hexdigest()


# ------------------------------------------------------------------------------------ inputs / reference


def make_inputs(case: Case) -> tuple[list[torch.Tensor], torch.Tensor | None]:
    """Bounded dyadic operands: eighths in [-1, 1] (exact in FP32/BF16/FP16); bias in quarters."""
    seed = SEED + int(hashlib.sha256(case.id.encode()).hexdigest()[:8], 16) % 1_000_000
    g = torch.Generator().manual_seed(seed)
    tdt = DTYPES[case.dtype][2]
    pairs = []
    n_terms = 1 if case.kind == "splitk" else case.terms
    for _ in range(n_terms):
        a = (torch.randint(-8, 9, (case.M, case.K), generator=g).to(torch.float64) / 8).to(tdt)
        b = (torch.randint(-8, 9, (case.N, case.K), generator=g).to(torch.float64) / 8).to(tdt)
        pairs += [a.contiguous(), b.contiguous()]
    bias = None
    if case.bias:
        bias = (torch.randint(-16, 17, (1, case.N), generator=g).to(torch.float64) / 4).to(torch.float32)
    return pairs, bias


def reference(case: Case, pairs: list[torch.Tensor], bias: torch.Tensor | None) -> torch.Tensor:
    """Independent CPU float64 reference (no DSL, no simulator), rounded once to FP32."""
    acc = torch.zeros(case.M, case.N, dtype=torch.float64)
    for i in range(0, len(pairs), 2):
        acc += pairs[i].to(torch.float64) @ pairs[i + 1].to(torch.float64).T
    if bias is not None:
        acc += bias.to(torch.float64)
    ref = acc.to(torch.float32)
    if not torch.equal(ref.to(torch.float64), acc):  # the domain promise: every value is exact in FP32
        raise AssertionError(f"{case.id}: reference is not exactly representable in FP32")
    return ref


def compare(case: Case, out: torch.Tensor, ref: torch.Tensor) -> dict:
    out = out.detach().to("cpu").reshape(ref.shape).to(torch.float32).contiguous()
    bitwise = torch.equal(out.view(torch.int32), ref.view(torch.int32))
    finite = torch.isfinite(out)
    diff = (out.to(torch.float64) - ref.to(torch.float64))
    diff = torch.where(finite, diff, torch.full_like(diff, float("inf")))
    ref_norm = ref.to(torch.float64).norm().item()
    return {
        "bitwise": bool(bitwise),
        "max_abs_diff": float(diff.abs().max().item()) if diff.numel() else 0.0,
        "rel_l2": float(diff.norm().item() / ref_norm) if ref_norm else float(diff.norm().item()),
        "mismatched": int((out.view(torch.int32) != ref.view(torch.int32)).sum().item()),
        "elements": int(ref.numel()),
        "non_finite": int((~finite).sum().item()),
        "poison_left": int((out == POISON).sum().item()),
        "output_sha256": hashlib.sha256(out.numpy().tobytes()).hexdigest(),
    }


def fragments(case: Case, pairs: list[torch.Tensor]) -> list[torch.Tensor]:
    """The per-MMAD partial products the kernel accumulates: K fragments for split-K, terms for a chain."""
    if case.kind == "splitk":
        a, b = pairs[0].to(torch.float64), pairs[1].to(torch.float64)
        return [a[:, k:k + case.splitk] @ b[:, k:k + case.splitk].T for k in range(0, case.K, case.splitk)]
    return [pairs[i].to(torch.float64) @ pairs[i + 1].to(torch.float64).T for i in range(0, len(pairs), 2)]


def decompose(case: Case, out: torch.Tensor, pairs: list[torch.Tensor], bias: torch.Tensor | None) -> dict:
    """Explain a wrong output as ``bias + sum_f c_f * P_f`` (least squares over the whole tile, then per row).

    Integer coefficients with ~0 residual mean whole fragments were dropped / repeated, i.e. an accumulate
    read a stale accumulator (the M10-081 signature); a large residual means the error is not
    fragment-shaped (e.g. addressing or layout)."""
    frags = fragments(case, pairs)
    y = out.detach().to("cpu").reshape(case.M, case.N).to(torch.float64)
    if bias is not None:
        y = y - bias.to(torch.float64)
    X = torch.stack([f.reshape(-1) for f in frags], dim=1)
    sol = torch.linalg.lstsq(X, y.reshape(-1, 1)).solution.reshape(-1)
    resid = (X @ sol - y.reshape(-1)).norm().item() / max(y.norm().item(), 1e-30)
    per_row = []
    for r in range(case.M):
        Xr = torch.stack([f[r] for f in frags], dim=1)
        cr = torch.linalg.lstsq(Xr, y[r].reshape(-1, 1)).solution.reshape(-1)
        rr = (Xr @ cr - y[r]).norm().item() / max(y[r].norm().item(), 1e-30)
        per_row.append({"row": r, "coef": [round(v, 4) for v in cr.tolist()], "rel_resid": rr})
    return {"fragments": len(frags), "tile_coef": [round(v, 4) for v in sol.tolist()], "tile_rel_resid": resid,
            "rows": per_row}


# ---------------------------------------------------------------------------------------- IR facts


def ir_facts(kernel) -> dict:
    """Lowered-IR facts that differ between profiles: MMADs, M barriers, and the A2-family trap lint."""
    from ascriptor.ir.lint import lint_lowered
    from ascriptor.passes import PIPELINE, PassManager

    lowered = PassManager(PIPELINE).run(kernel.ir())
    facts = {"mmad": 0, "mmad_accumulate": 0, "barrier_M": 0, "barrier_M_settle": 0}
    for f in lowered.functions:
        for op in f.walk():
            if op.opcode == "cube.mmad":
                facts["mmad"] += 1
                if op.attrs.get("is_init", True) is False:
                    facts["mmad_accumulate"] += 1
            elif op.opcode == "sync.barrier" and str(op.attrs.get("pipe")) in ("M", "Pipe.M"):
                facts["barrier_M"] += 1
                if any("settle" in (o.note or "") for o in op.origin):  # the desugar rule's provenance note
                    facts["barrier_M_settle"] += 1
    diags = [d for d in lint_lowered(lowered) if d.kind == "trap"]
    facts["trap_lints"] = sorted({(d.rule or "?") + ": " + d.message.split(":")[0] for d in diags})
    return facts


# ------------------------------------------------------------------------------------------ launchers


def patch_generated(files: dict, variant: str) -> int:
    """Diagnostic only (``--patch-generated``): edit this run's *generated* CCE cube source in memory before it
    is built; the ascriptor checkout is never touched. ``settle`` inserts ``PipeBarrier<PIPE_M>();`` after each
    split-K MMAD branch (``if (_subk_is0) { mmad(..., true) } else { mmad(..., false) }``) and the M-side event
    ``.set()`` that follows it -- the position the pinned desugar rule gives the barrier for FP32 ("after the whole
    MMAD branch and before counter advance", M10-081). ``sham`` puts a comment there instead, so the
    patch-and-rebuild path itself is controlled. Returns the number of insertions."""
    n = 0
    for name in list(files):
        if not name.endswith("_cube.h"):
            continue
        lines = files[name].decode().split("\n")
        if variant == "settle" and any("PipeBarrier<PIPE_M>" in ln for ln in lines):
            continue  # already settled (the FP32 rule fired): nothing to add
        out, i = [], 0
        while i < len(lines):
            out.append(lines[i])
            if "mmad(" in lines[i] and ", false);" in lines[i] and i + 1 < len(lines) and lines[i + 1].strip() == "}":
                out.append(lines[i + 1])
                i += 2
                if i < len(lines) and lines[i].strip().endswith(".set();") or (i < len(lines) and ".set();  //" in lines[i]):
                    out.append(lines[i])
                    i += 1
                indent = lines[i - 1][: len(lines[i - 1]) - len(lines[i - 1].lstrip())]
                out.append(indent + ("PipeBarrier<PIPE_M>();  // A2-01 diagnostic: settle after the split-K MMAD"
                                     if variant == "settle" else "// A2-01 diagnostic: sham edit (no instruction)"))
                n += 1
                continue
            i += 1
        files[name] = "\n".join(out).encode()
    return n


class Runner:
    def __init__(self, launcher: str, profile: str, out: pathlib.Path, timeout: float, patch: str = "none",
                 block_dim: int = 1):
        self.launcher, self.profile, self.out, self.timeout, self.patch = launcher, profile, out, timeout, patch
        self.block_dim = block_dim
        self._exec = {}
        self.patched = {}

    def __call__(self, case: Case, kernel, args: tuple) -> tuple[torch.Tensor, dict]:
        if self.launcher == "sim":
            from ascriptor.backends.sim.launch import run_kernel

            return run_kernel(kernel, *args, block_dim=1, timeout=self.timeout, seed_outputs=True), {}
        if self.launcher == "pipesim":
            from ascriptor.backends.sim.pipesim import simulate
            from ascriptor.passes import PIPELINE, PassManager
            from ascriptor.passes.autosync import check_balance

            lowered = PassManager(PIPELINE).run(kernel.ir())
            balance = check_balance(lowered)
            sim = simulate(lowered, args, block_dim=1, timeout=self.timeout, seed_outputs=True, check_gm=True)
            out = sim.outputs[0] if isinstance(sim.outputs, (list, tuple)) else sim.outputs
            return out, {"event_balance": [str(b) for b in balance] if balance else [],
                         "hazards": len(sim.hazards), "hazard_samples": [str(h)[:300] for h in sim.hazards[:3]],
                         "deadlock": bool(sim.report.get("deadlock")), "model_cycles": sim.cycles}
        if self.launcher == "aclnn":
            from ascriptor.runtime import OpExec

            if case.id not in self._exec:
                sub = kernel_name(case, self.block_dim) + ("" if self.patch == "none" else f".{self.patch}")
                ex = OpExec(kernel, launcher="aclnn", backend="cce", device=self.profile,
                            out_dir=self.out / "build" / self.profile / sub,
                            block_dim=self.block_dim, timeout=self.timeout, seed_outputs=True)
                if self.patch != "none":
                    self.patched[case.id] = patch_generated(ex.artifacts.files, self.patch)
                self._exec[case.id] = ex
            extra = {"patch_generated": self.patch, "patch_insertions": self.patched[case.id]} if self.patch != "none" else {}
            return self._exec[case.id](*args), extra
        raise ValueError(self.launcher)


def versions() -> dict:
    import ascriptor

    lib = pathlib.Path(ascriptor.__file__).resolve().parent
    v = {"python": platform.python_version(), "torch": torch.__version__,
         "ascriptor_version": getattr(ascriptor, "__version__", None),
         # identities of the two files that carry the repair, so the receipt names the source without a path
         "ascriptor_desugar_sha256": hashlib.sha256((lib / "passes/desugar.py").read_bytes()).hexdigest(),
         "ascriptor_lint_sha256": hashlib.sha256((lib / "ir/lint.py").read_bytes()).hexdigest()}
    try:
        v["ascriptor_commit"] = subprocess.run(["git", "-C", str(lib), "rev-parse", "HEAD"], capture_output=True,
                                               text=True, timeout=10).stdout.strip() or None
    except Exception:  # pragma: no cover - not a git checkout
        v["ascriptor_commit"] = None
    try:
        import numpy

        v["numpy"] = numpy.__version__
    except ImportError:
        v["numpy"] = None
    return v


def device_facts() -> dict:
    """SoC / CANN / built-in op package names for the receipt (names only, never paths)."""
    facts: dict = {}
    home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    for rel in ("opp/version.info", "version.cfg"):
        p = pathlib.Path(home or "/nonexistent") / rel
        if p.is_file():
            for line in p.read_text(errors="replace").splitlines():
                if line.startswith("Version="):
                    facts["cann_version"] = line.split("=", 1)[1].strip()
            if "cann_version" in facts:
                break
    opp = os.environ.get("ASCEND_OPP_PATH")
    kdir = pathlib.Path(opp or "/nonexistent") / "built-in/op_impl/ai_core/tbe/kernel"
    facts["builtin_op_packages"] = sorted(p.name for p in kdir.iterdir() if p.is_dir()) if kdir.is_dir() else None
    try:
        import torch_npu  # noqa: F401

        p = torch.npu.get_device_properties(0)
        facts.update(soc=p.name, cube_core_num=getattr(p, "cube_core_num", None),
                     vector_core_num=getattr(p, "vector_core_num", None), torch_npu=torch_npu.__version__)
    except Exception as exc:  # torch_npu absent or no device: record why, do not guess
        facts["soc"] = None
        facts["soc_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return facts


def receipt_line(device: dict | None) -> str:
    """One-line SoC / CANN / op-package identity with a UTC timestamp, so no run's numbers are ever read
    without their environment (A2-11 acceptance; same intent as bringup.py's ``_receipt_line``)."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not device:
        return f"SoC=model-only cann=- op_packages=0 at={ts}"
    pkgs = device.get("builtin_op_packages") or []
    return f"SoC={device.get('soc')} cann={device.get('cann_version')} op_packages={len(pkgs)} at={ts}"


# --------------------------------------------------------------------------------------------- hit table

#: (group, stage, tree, unit, module, function, role). ``tree`` is "upstream" (ascriptor kernels, read-only) or
#: "repo" (this repository's derived units). "dispatched" rows are what ascend_fla's public ops load today
#: (ops/kda/chunk.py, chunk_bwd.py, fused_recurrent.py; GDN per the upstream units' execute.py);
#: "replaced" rows are the upstream originals of derived kernels, scanned so A2-03 sees both.
HIT_KERNELS = [
    ("KDA fwd", "gate", "repo", "kda_fwd_stable", "gate", "kda_sub1_gate_stable_kernel", "dispatched"),
    ("KDA fwd", "scores", "repo", "kda_fwd_stable", "intra", "kda_sub2_score_stable_kernel", "dispatched"),
    ("KDA fwd", "inverse", "upstream", "kda_fwd", "triangular_inverse", "tril_inverse64_v2_strict_bf16_kernel", "dispatched"),
    ("KDA fwd", "wy", "repo", "kda_fwd_stable", "wy", "kda_sub3_wy_stable_kernel", "dispatched"),
    ("KDA fwd", "recurrent", "repo", "kda_fwd_stable", "recurrent", "kda_sub45_aqk_repaired_kernel", "dispatched"),
    ("KDA bwd", "scan_fused", "upstream", "kda_bwd", "scan_fused", "scan_fused_kernel", "dispatched"),
    ("KDA bwd", "inverse_mm", "repo", "kda_bwd_stable", "inverse_mm", "inverse_mm_bounded_kernel", "dispatched"),
    ("KDA bwd", "inverse_epilogue", "upstream", "kda_bwd", "inverse_epilogue", "inverse_epilogue_kernel", "dispatched"),
    ("KDA bwd", "inverse_dainv", "upstream", "kda_bwd", "inverse_dainv", "inverse_dainv_kernel", "dispatched"),
    ("KDA bwd", "inverse_dakk_fused", "upstream", "kda_bwd", "inverse_dakk_fused", "inverse_dakk_fused_kernel", "dispatched"),
    ("KDA bwd", "finalize_pre", "repo", "kda_bwd_stable", "finalize_pre", "finalize_pre_stable_kernel", "dispatched"),
    ("KDA bwd", "finalize_pair", "upstream", "kda_bwd", "finalize_pair", "finalize_pair_kernel", "dispatched"),
    ("KDA bwd", "finalize_post", "repo", "kda_bwd_stable", "finalize_post", "finalize_post_stable_kernel", "dispatched"),
    ("KDA bwd", "finalize_reduce", "upstream", "kda_bwd", "finalize_reduce", "finalize_reduce_kernel", "dispatched"),
    ("KDA decode", "step", "repo", "kda_fused_recurrent", "step", "kda_fused_recurrent_kernel", "dispatched"),
    ("GDN fwd", "preprocess", "upstream", "gdn_fwd", "preprocess", "gdn_preprocess_v2_kernel", "dispatched"),
    ("GDN fwd", "inverse", "upstream", "gdn_fwd", "inverse", "tril_inverse64_v2_strict_bf16_kernel", "dispatched"),
    ("GDN fwd", "recompute", "upstream", "gdn_fwd", "recompute", "gdn_recompute_wu_v2_kernel", "dispatched"),
    ("GDN fwd", "scores", "upstream", "gdn_fwd", "scores", "sub1_kernel", "dispatched"),
    ("GDN fwd", "recurrent", "upstream", "gdn_fwd", "recurrent", "gdn_recurrent_plain", "dispatched"),
    ("GDN bwd", "scan_local", "upstream", "gdn_bwd", "scan_local", "scan_local_bwd_kernel", "dispatched"),
    ("GDN bwd", "scan_state", "upstream", "gdn_bwd", "scan_state", "scan_state_bwd_kernel", "dispatched"),
    ("GDN bwd", "wu", "upstream", "gdn_bwd", "wu", "wu_bwd_kernel", "dispatched"),
    ("GDN bwd", "inverse_preprocess", "upstream", "gdn_bwd", "inverse_preprocess", "inverse_preprocess_bwd_kernel", "dispatched"),
    ("GDN bwd", "finalize", "upstream", "gdn_bwd", "finalize", "finalize_bwd_kernel", "dispatched"),
    ("GDN fwd", "recurrent_saved", "upstream", "gdn_fwd", "recurrent_saved", "gdn_recurrent_saved", "alternative"),
    ("KDA fwd", "gate", "upstream", "kda_fwd", "gate", "kda_sub1_gate_kernel", "replaced"),
    ("KDA fwd", "scores", "upstream", "kda_fwd", "intra", "kda_sub2_score_kernel", "replaced"),
    ("KDA fwd", "wy", "upstream", "kda_fwd", "wy", "kda_sub3_wy_kernel", "replaced"),
    ("KDA fwd", "recurrent", "upstream", "kda_fwd", "recurrent", "kda_sub45_fused_kernel", "replaced"),
    ("KDA bwd", "inverse_mm", "upstream", "kda_bwd", "inverse_mm", "inverse_mm_kernel", "replaced"),
    ("KDA bwd", "finalize_pre", "upstream", "kda_bwd", "finalize_pre", "finalize_pre_kernel", "replaced"),
    ("KDA bwd", "finalize_post", "upstream", "kda_bwd", "finalize_post", "finalize_post_kernel", "replaced"),
]
_MM_NAMES = ("matmul", "mmad", "matmul_mx", "mmad_mx")
_DT_SHORT = {"float": "f32", "float32": "f32", "bfloat16": "bf16", "half": "f16", "float16": "f16", "f32": "f32",
             "bf16": "bf16", "f16": "f16"}


def _ast_calls(path: pathlib.Path, fn_name: str) -> list[dict]:
    """matmul/mmad calls reachable from ``fn_name`` in its file (helpers followed by name), with operand dtypes
    resolved from each buffer's declaration ``name = XBuff(DT.<dtype>, ..., Position.<pos>)``."""
    import ast

    tree = ast.parse(path.read_text())
    decl, funcs = {}, {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Call):
            for a in n.value.args[:1]:
                if isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name) and a.value.id == "DT":
                    decl[n.targets[0].id] = _DT_SHORT.get(a.attr, a.attr)
    seen, todo, calls = set(), [fn_name], []
    while todo:
        name = todo.pop()
        if name in seen or name not in funcs:
            continue
        seen.add(name)
        for n in ast.walk(funcs[name]):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)):
                continue
            if n.func.id in funcs:
                todo.append(n.func.id)
            if n.func.id not in _MM_NAMES:
                continue

            def base(e):
                while isinstance(e, (ast.Subscript, ast.Attribute)):
                    e = e.value
                return e.id if isinstance(e, ast.Name) else None

            names = [base(a) for a in n.args[:3]]
            kw = {k.arg: ast.unparse(k.value) for k in n.keywords}
            if n.func.id.startswith("mmad") and len(n.args) > 6:
                kw.setdefault("is_init", ast.unparse(n.args[6]))
            calls.append({"line": n.lineno, "in": name, "op": n.func.id,
                          "dtypes": [decl.get(x, "?") for x in names],
                          "splitk": kw.get("splitk"), "splitn": kw.get("splitn"),
                          "is_init": kw.get("is_init", "True"),
                          "mnk": [kw.get(x) for x in ("m", "n", "k")] if n.func.id.startswith("matmul")
                          else [kw.get(x) for x in ("M", "N", "K")]})
    return sorted(calls, key=lambda c: c["line"])


def _ir_sites(kernel) -> tuple[int, int, list[dict]]:
    """Accumulating MMADs (``is_init=False``) reachable after an MMAD to the same L0C root with no M/ALL
    barrier, on the kernel's own lowered IR (a5 profile, as authored). ``library_fp32_rule`` is the pinned
    library's own A2-family check applied to that op; ``any_dtype`` is the same path analysis with the FP32
    condition dropped (the library helpers are reused read-only)."""
    from ascriptor.ir import lint as L
    from ascriptor.passes import PIPELINE, PassManager

    lowered = PassManager(PIPELINE).run(kernel.ir())
    n_mmad, n_acc, sites = 0, 0, []
    for f in lowered.functions:
        defs = {r.name: op for op in f.walk() for r in op.results}

        def pending(target, root):
            hazard = False

            def to_root(op):
                return op.opcode == "cube.mmad" and len(op.operands) >= 3 and L._root_name(op.operands[0], defs) == root

            def block_state(block, incoming):
                nonlocal hazard
                state = set(incoming)
                for cur in block.ops:
                    if cur is target and True in state:
                        hazard = True
                    if cur.opcode == "sync.barrier" and L._ident(cur, "pipe", "ALL") in ("M", "ALL"):
                        state = {False}
                    elif to_root(cur):
                        state = {True}
                    if not cur.regions:
                        continue
                    if cur.opcode == "cf.for":
                        entry = set(state)
                        while True:
                            widened = entry | block_state(cur.regions[0], entry)
                            if widened == entry:
                                break
                            entry = widened
                        state = entry
                    else:
                        exits = [block_state(r, state) for r in cur.regions]
                        state = set().union(*exits) if exits else state
                return state

            block_state(f.body, {False})
            return hazard

        for op in f.walk():
            if op.opcode != "cube.mmad":
                continue
            n_mmad += 1
            if op.attrs.get("is_init", True) is not False:
                continue
            n_acc += 1
            root = L._root_name(op.operands[0], defs)
            if root is None or not pending(op, root):
                continue
            loc = op.loc.chain[0] if op.loc else ""
            m = __import__("re").search(r"([^/\\]+\.py):(\d+)", str(loc))
            dts = [_DT_SHORT.get(v.type.dtype.name, v.type.dtype.name) for v in op.operands[1:3]
                   if hasattr(getattr(v, "type", None), "dtype")]
            sites.append({"file": m.group(1) if m else None, "line": int(m.group(2)) if m else None, "root": root,
                          "operand_dtypes": dts, "library_fp32_rule": bool(L._a2_family_unsettled_fp32_mmad(f, op, defs)),
                          "expansion": sorted({o.note for o in op.origin if o.note})})
    return n_mmad, n_acc, sites


def hit_table(upstream_root: pathlib.Path, repo_root: pathlib.Path, out: pathlib.Path) -> int:
    rows = []
    for group, stage, tree, unit, module, fn, role in HIT_KERNELS:
        root = upstream_root if tree == "upstream" else repo_root
        path = root / unit / "kernels" / f"{module}.py"
        row = {"group": group, "stage": stage, "tree": tree, "file": f"{unit}/kernels/{module}.py", "kernel": fn, "role": role}
        try:
            row["calls"] = _ast_calls(path, fn)
            spec = importlib.util.spec_from_file_location(f"a201_hit_{tree}_{unit}_{module}", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            row["ir_mmad_ops"], row["ir_accumulate_ops"], row["ir_sites"] = _ir_sites(getattr(mod, fn))
        except FileNotFoundError as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        except Exception as exc:  # e.g. a kernel that does not lower at this pin: keep the AST verdict
            import re

            msg = re.sub(r"(?:[A-Za-z]:)?[/\\][^\s\"']*[/\\]([^/\\\s\"']+\.py)", r"\1", str(exc).splitlines()[0])  # no paths
            row["ir_error"] = f"{type(exc).__name__}: {msg}"[:200]
        calls, sites = row.get("calls", []), row.get("ir_sites", [])
        splitk_lines = {c["line"] for c in calls if c["splitk"]}
        verdict = []
        if any(c["splitk"] and c["dtypes"][1:] == ["f32", "f32"] for c in calls):
            verdict.append("HIT-A fp32 split-K (a2: settled by the pinned desugar rule)")
        if any(s["library_fp32_rule"] and s["line"] not in splitk_lines for s in sites):
            verdict.append("HIT-B fp32 accumulate chain, NOT settled (a2 lint trap)")
        low = [s for s in sites if not set(s["operand_dtypes"]) <= {"f32"}]
        if low:
            verdict.append("EXPOSED-C " + "/".join(sorted({d for s in low for d in s["operand_dtypes"]}))
                           + " accumulate, outside the fp32 rule (device probe)")
        if "ir_error" in row:  # AST only: an is_init=False call is a candidate chain (settling not analysable)
            for c in calls:
                if c["is_init"] not in ("True", "1") and not c["splitk"]:
                    kind = "HIT-B?" if c["dtypes"][1:] == ["f32", "f32"] else "EXPOSED-C?"
                    verdict.append(f"{kind} line {c['line']} accumulate (AST only; IR did not lower)")
        if not verdict:
            if not calls and not row.get("ir_mmad_ops"):
                verdict.append("MISS no cube matmul")
            elif row.get("ir_accumulate_ops"):
                verdict.append("MISS every accumulate MMAD follows an explicit M/ALL barrier")
            else:
                verdict.append("MISS every MMAD initialises its L0C (no accumulate)")
        if "ir_error" in row:
            verdict.append("IR: " + row["ir_error"])
        if "error" in row:
            verdict = ["ERROR " + row["error"]]
        row["verdict"] = verdict
        rows.append(row)
    out.mkdir(parents=True, exist_ok=True)
    (out / "hit_table.json").write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n")
    print("| # | group | stage | kernel | file | role | matmul calls (line: a,b dtype; m,n,k; split; is_init) | IR accumulate sites (line: dtypes, lib fp32 rule) | verdict |")
    print("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        calls = "<br>".join(f"{c['line']}: {','.join(c['dtypes'][1:])}; {','.join(str(x) for x in c['mnk'])}; "
                            f"{('splitk=' + c['splitk']) if c['splitk'] else ('splitn=' + c['splitn']) if c['splitn'] else '-'}; "
                            f"{c['is_init']}" for c in r.get("calls", [])) or "none"
        sites = "<br>".join(f"{s['line']}: {','.join(s['operand_dtypes'])}, {'yes' if s['library_fp32_rule'] else 'no'}"
                            for s in r.get("ir_sites", [])) or "none"
        print(f"| {i} | {r['group']} | {r['stage']} | `{r['kernel']}` | `{r['file']}` | {r['role']} | {calls} | {sites} | "
              f"{'; '.join(r['verdict'])} |")
    return 1 if any("error" in r for r in rows) else 0  # a lowering failure is reported, not an error


# ------------------------------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=("a2", "a3", "a5"), default="a2")
    ap.add_argument("--launcher", choices=("reference", "sim", "pipesim", "aclnn"), default="sim")
    ap.add_argument("--case", nargs="*", default=["*"], help="case ids or glob patterns (default: all)")
    ap.add_argument("--repeat", type=int, default=1, help="device runs per case (aclnn only)")
    ap.add_argument("--block-dim", type=int, default=1,
                    help="cube block_dim for the aclnn build (A2-11 multi-bd sweep; one build per process)")
    ap.add_argument("--jitter", type=int, default=0,
                    help="A2-11 timing perturbation: independent scratch MMADs between chain terms (chain cases only)")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "tmp" / "A2-01" / "runs")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--save-outputs", action="store_true", help="save wrong outputs as .pt next to the receipts")
    ap.add_argument("--patch-generated", choices=("none", "sham", "settle"), default="none",
                    help="aclnn diagnostic: edit the generated CCE source before building (see patch_generated)")
    ap.add_argument("--hit-table", action="store_true",
                    help="scan the 25 KDA/GDN kernels (+ alternatives) for the construct; no launch")
    ap.add_argument("--upstream-kernels", type=pathlib.Path, default=None,
                    help="ascriptor kernels/projects/a5 (default: $ASCRIPTOR_WORKSPACE/kernels/projects/a5)")
    ns = ap.parse_args(argv)
    if ns.hit_table:
        ws = pathlib.Path(os.environ.get("ASCRIPTOR_WORKSPACE", REPO.parent / "ascriptor"))
        up = ns.upstream_kernels or ws / "kernels" / "projects" / "a5"
        return hit_table(up, REPO / "kernels" / "projects" / "a5", ns.out / "hit_table")

    cases = [c for c in build_cases() if any(fnmatch.fnmatchcase(c.id, p) for p in ns.case)]
    if ns.jitter:  # A2-11: perturb the chain cases' inter-MMAD timing (split-K cases are one MMAD, unperturbed)
        cases = [dataclasses.replace(c, jitter=ns.jitter, id=f"{c.id}_j{ns.jitter}") if c.kind == "chain" else c
                 for c in cases]
    if ns.list:
        for c in cases:
            print(f"{c.id:40s} expect={c.expect:7s} {c.why}")
        return 0
    if not cases:
        print(f"no case matches {ns.case}", file=sys.stderr)
        return 2
    if ns.launcher == "aclnn" and ns.profile not in ("a2", "a3"):
        print(f"--launcher aclnn needs an a2/a3 device; profile {ns.profile} is a model-only control here "
              f"(this reproducer targets the c220 family)", file=sys.stderr)
        return 2
    if ns.case and any(c.bias for c in cases) and ns.profile == "a5":
        cases = [c for c in cases if not c.bias]  # the bias-table form is the A2 example's; a5 control drops it
        print("note: a5 profile skips the *_bias case (A2 BT-row form); the no-bias twin runs instead")
    repeat = ns.repeat if ns.launcher == "aclnn" else 1
    card = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0")  # multi-card sweeps keep receipts distinct (D-PM-28)

    env = {"versions": versions(), "device": device_facts() if ns.launcher == "aclnn" else None}
    print("ENV", json.dumps(env, sort_keys=True), flush=True)
    if ns.patch_generated != "none" and ns.launcher != "aclnn":
        print("--patch-generated applies to --launcher aclnn only", file=sys.stderr)
        return 2
    runner = Runner(ns.launcher, ns.profile, ns.out, ns.timeout, ns.patch_generated, ns.block_dim)
    tag = ns.launcher + (f"_bd{ns.block_dim}_card{card}" if ns.launcher == "aclnn" else "")
    if ns.patch_generated != "none":
        tag = f"{tag}.{ns.patch_generated}"
    run_dir = ns.out / ns.profile / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "a2-11.repro/1", "profile": ns.profile, "launcher": ns.launcher, "repeat": repeat,
               "block_dim": ns.block_dim, "jitter": ns.jitter, "card": card, "receipt": receipt_line(env["device"]),
               "comparison": "bitwise vs CPU float64 reference rounded once to FP32 (exact domain)", **env,
               "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cases": []}
    failed_expectation = []
    for case in cases:
        rec = {"case": dataclasses.asdict(case), "runs": [], "receipt": receipt_line(env["device"]),
               "block_dim": ns.block_dim, "card": card}
        try:
            pairs, bias = make_inputs(case)
            ref = reference(case, pairs, bias)
            rec["reference_sha256"] = hashlib.sha256(ref.numpy().tobytes()).hexdigest()
            if ns.launcher != "reference":
                kernel, src_sha = load_kernel(case, ns.profile, ns.out / "kernels" / ns.profile, ns.block_dim)
                rec["kernel_source_sha256"] = src_sha
                rec["ir"] = ir_facts(kernel)
                args = tuple(pairs) + ((bias,) if bias is not None else ())
                for r in range(repeat):
                    z = torch.full((case.M, case.N), POISON, dtype=torch.float32)
                    t0 = time.time()
                    try:
                        out, extra = runner(case, kernel, args + (z,))
                    except Exception as exc:  # a device fault (e.g. AI Core 507015) records the run and continues the sweep
                        res = {"run": r + 1, "bitwise": False, "device_error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}",
                               "wall_s": round(time.time() - t0, 3)}
                        rec["runs"].append(res)
                        if (r + 1) % 25 == 0 or r + 1 == repeat:
                            (run_dir / f"{case.id}.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
                        print(f"CASE {case.id} profile={ns.profile} launcher={ns.launcher} run={r + 1}/{repeat} "
                              f"expect={case.expect} DEVICE_ERROR {res['device_error'][:90]}", flush=True)
                        continue
                    res = {"run": r + 1, **compare(case, out, ref), **extra, "wall_s": round(time.time() - t0, 3)}
                    if not res["bitwise"] and res["non_finite"] == 0:
                        res["decomposition"] = decompose(case, out, pairs, bias)
                        rows = res["decomposition"]["rows"]
                        bad = [x["row"] for x in rows if any(abs(c - 1) > 1e-6 for c in x["coef"])]
                        print(f"DECOMP {case.id} run={r + 1} fragments={res['decomposition']['fragments']} "
                              f"tile_coef={res['decomposition']['tile_coef']} "
                              f"tile_rel_resid={res['decomposition']['tile_rel_resid']:.3g} "
                              f"max_row_resid={max(x['rel_resid'] for x in rows):.3g} rows_not_all_ones={bad}",
                              flush=True)
                        if ns.save_outputs:
                            torch.save(out.detach().cpu(), run_dir / f"{case.id}.run{r + 1}.out.pt")
                    rec["runs"].append(res)
                    if (r + 1) % 25 == 0 or r + 1 == repeat:  # A2-11: persist partial receipt so a timeout keeps the runs
                        (run_dir / f"{case.id}.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
                    print(f"CASE {case.id} profile={ns.profile} launcher={ns.launcher} run={r + 1}/{repeat} "
                          f"expect={case.expect} bitwise={res['bitwise']} max_abs_diff={res['max_abs_diff']:.6g} "
                          f"rel_l2={res['rel_l2']:.6g} mismatched={res['mismatched']}/{res['elements']} "
                          f"non_finite={res['non_finite']} poison_left={res['poison_left']} "
                          f"sha={res['output_sha256'][:16]} ir_settle={rec['ir']['barrier_M_settle']} "
                          f"ir_barrier_M={rec['ir']['barrier_M']} traps={len(rec['ir']['trap_lints'])}"
                          + (f" hazards={extra['hazards']} deadlock={extra['deadlock']}" if "hazards" in extra else ""),
                          flush=True)
            else:
                print(f"CASE {case.id} profile={ns.profile} launcher=reference exact_fp32=True "
                      f"ref_sha={rec['reference_sha256'][:16]}", flush=True)
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["traceback_tail"] = traceback.format_exc().splitlines()[-6:]
            print(f"CASE {case.id} profile={ns.profile} launcher={ns.launcher} ERROR {rec['error'][:400]}", flush=True)
        device_errors = [r for r in rec["runs"] if "device_error" in r]
        genuine_wrong = [r for r in rec["runs"] if not r.get("bitwise") and "device_error" not in r]
        # a transient device fault (AI Core 507015) is an environmental event, not a wrong result: only a
        # genuine wrong output fails a bitwise-expect case. Device faults are counted and reported separately.
        ok = "error" not in rec and not genuine_wrong and any(r.get("bitwise") for r in rec["runs"])
        rec["all_bitwise"] = ok
        rec["device_error_runs"] = len(device_errors)
        if case.expect == "bitwise" and not ok:
            failed_expectation.append(case.id)
        (run_dir / f"{case.id}.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
        summary["cases"].append({"id": case.id, "expect": case.expect, "all_bitwise": ok, "error": rec.get("error"),
                                 "runs": len(rec["runs"]),
                                 "bitwise_runs": sum(1 for r in rec["runs"] if r.get("bitwise")),
                                 "wrong_runs": len(genuine_wrong), "device_error_runs": len(device_errors),
                                 "max_abs_diff": max((r["max_abs_diff"] for r in rec["runs"] if "max_abs_diff" in r), default=None)})
    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary["failed_expectation"] = failed_expectation
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(f"SUMMARY profile={ns.profile} launcher={ns.launcher} cases={len(cases)} "
          f"all_bitwise={sum(c['all_bitwise'] for c in summary['cases'])} "
          f"errors={sum(1 for c in summary['cases'] if c['error'])} failed_expectation={failed_expectation}", flush=True)
    return 4 if failed_expectation else 0


if __name__ == "__main__":
    sys.exit(main())
