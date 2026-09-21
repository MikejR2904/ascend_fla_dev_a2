"""A2-02 real-machine probe: SoC resolution, CANN identity, op-entry gating (Ascend 910B3).

Prints, for a D-PM-34 receipt that a reviewer can reproduce and verify:

* the torch_npu device name and ``platform.resolve_soc()`` result;
* the driver/compiler/opp ``version.info`` first line and **full** sha256;
* the built-in op-package ``ascend910b`` kernel directory listing;
* the **full** exception each gated op entry (``prepare`` / ``chunk_kda`` /
  ``fused_recurrent_kda``) raises before any compile.

Nothing here is filtered or truncated. Redirect the whole run to a file for the receipt:

    ASCEND_RT_VISIBLE_DEVICES=<free card> PYTHONPATH=<ascriptor>/library:<repo> \\
        python benchmarks/a2/probe_platform.py > evidence.log 2>&1
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import traceback

import torch

#: Ascend install root — taken from the environment so no machine path is written into the
#: repository; only paths relative to it are printed (AGENTS.md §5 / §10).
ASCEND_HOME = pathlib.Path(os.environ.get("ASCEND_HOME", "/usr/local/Ascend"))


def _print_environment() -> None:
    from ascend_fla import platform

    print("== SoC ==")
    print("device_name:", torch.npu.get_device_name(0))
    print("resolve_soc:", platform.resolve_soc())

    print("== CANN version.info (full sha256; paths relative to ASCEND_HOME) ==")
    cann = sorted(ASCEND_HOME.glob("cann-*"))
    infos = [ASCEND_HOME / "driver/version.info"]
    if cann:
        infos += [cann[-1] / "compiler/version.info", cann[-1] / "opp/version.info"]
    for info in infos:
        if info.is_file():
            raw = info.read_bytes()
            first = raw.decode("utf-8", "replace").splitlines()[0]
            print(f"{info.relative_to(ASCEND_HOME)}: {first} sha256={hashlib.sha256(raw).hexdigest()}")

    print("== built-in op-package ascend910b (relative to ASCEND_HOME) ==")
    for root in cann:
        soc_dir = root / "opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b"
        if soc_dir.is_dir():
            print(f"{soc_dir.relative_to(ASCEND_HOME)}:", sorted(p.name for p in soc_dir.iterdir()))


def _print_gating() -> None:
    from ascend_fla.ops.kda import prepare
    from ascend_fla.ops.kda.autograd import chunk_kda
    from ascend_fla.ops.kda.fused_recurrent import fused_recurrent_kda

    zeros = lambda *shape: torch.zeros(*shape)
    entries = {
        "prepare": lambda: prepare(),
        "chunk_kda": lambda: chunk_kda(
            zeros(1, 64, 1, 128), zeros(1, 64, 1, 128), zeros(1, 64, 1, 128),
            zeros(1, 64, 1, 128), zeros(1, 64, 1)),
        "fused_recurrent_kda": lambda: fused_recurrent_kda(
            zeros(1, 1, 1, 128), zeros(1, 1, 1, 128), zeros(1, 1, 1, 128),
            zeros(1, 1, 1, 128), zeros(1, 1, 1)),
    }
    print("== op-entry gating (must raise before compile) ==")
    for name, call in entries.items():
        try:
            call()
            print(f"{name}: NO ERROR [FAIL: entry did not gate]")
        except Exception:  # noqa: BLE001 - the full text is the receipt
            print(f"{name} raised:")
            print(traceback.format_exc())


def main() -> int:
    _print_environment()
    _print_gating()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
