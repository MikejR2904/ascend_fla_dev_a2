"""Host-side guards over the A2 bring-up evidence (A2-10).

The bring-up runs on the 910B3 (``benchmarks/a2/bringup.py``); these tests read the committed
receipts and fix their conclusions so a regenerated receipt that regresses — a dropped case, a
mislabelled SoC, a missing revision — turns the suite red. They need no NPU. A2 numbers are
observation-only until A2-11 (D-PM-30); what is guarded is that the recorded bridge outputs
matched the harness bit-for-bit, that the receipts cover every contract case, and that the
environment and hygiene hold — not any operator conclusion.
"""
from __future__ import annotations

import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
EVIDENCE = REPO / "benchmarks/a2/evidence/bringup"
FWD_CONTRACT = REPO / "kernels/projects/a2/kda_fwd_stable/contract.json"
DECODE_CONTRACT = REPO / "kernels/projects/a2/kda_fused_recurrent/contract.json"
FWD_OUTPUTS = {"o", "final_state", "g_cumsum"}
DECODE_OUTPUTS = {"o", "final_state"}


def _receipts() -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(EVIDENCE.glob("*.json"))]


def _contract_case_ids(path: pathlib.Path) -> set[str]:
    return {case["id"] for case in json.loads(path.read_text(encoding="utf-8"))["cases"]}


def test_bringup_evidence_present():
    assert EVIDENCE.is_dir() and list(EVIDENCE.glob("*.json")), f"no bring-up receipts under {EVIDENCE}"


def test_bridge_outputs_are_bitwise_identical_to_harness():
    compared, violations = 0, []
    for receipt in _receipts():
        for key, result in receipt.get("bridge", {}).get("comparisons", {}).items():
            compared += 1
            if not result["bitwise"] or result["harness_sha256"] != result["bridge_sha256"]:
                violations.append(f"{key}: {result}")
    assert compared, "no bridge comparisons recorded"
    assert not violations, "\n".join(violations)


def test_bridge_matrix_covers_every_contract_case_and_output():
    """The recorded comparisons cover exactly the two units' contract cases, at both block_dims,
    with every output present — so deleting comparisons (the PM's negative control) fails here."""
    fwd_ids, decode_ids = _contract_case_ids(FWD_CONTRACT), _contract_case_ids(DECODE_CONTRACT)
    seen: dict[str, set[str]] = {}
    for receipt in _receipts():
        for key in receipt.get("bridge", {}).get("comparisons", {}):
            case, block_dim, output = key.split(":")
            seen.setdefault(f"{case}:{block_dim}", set()).add(output)
    for case_ids, outputs, unit in ((fwd_ids, FWD_OUTPUTS, "fwd"), (decode_ids, DECODE_OUTPUTS, "decode")):
        for case in case_ids:
            for block_dim in ("bd1", "bd2"):
                got = seen.get(f"{case}:{block_dim}")
                assert got == outputs, f"{unit} {case} {block_dim}: outputs {got} != {outputs}"


def test_bridge_trace_proves_the_bridge_ran():
    """Each case's bridge trace shows exactly the launch count its unit requires — the five-kernel
    forward chain launches 5 kernels with 5 pairwise-distinct signatures, the decode unit launches
    1 — judged per unit, not merely ``launches in (1, 5)``. So trimming a forward case to one launch
    or reusing a signature (the PM's negative control) fails here, not just a silent harness
    fall-back."""
    fwd_ids, decode_ids = _contract_case_ids(FWD_CONTRACT), _contract_case_ids(DECODE_CONTRACT)
    traced = 0
    for receipt in _receipts():
        for key, entry in receipt.get("bridge", {}).get("bridge_trace", {}).items():
            traced += 1
            case = key.split(":")[0]
            if case in fwd_ids:
                expected = 5
            elif case in decode_ids:
                expected = 1
            else:
                raise AssertionError(f"{key}: case is in neither contract")
            launches, signatures = entry["launches"], entry["signatures"]
            assert launches == expected, f"{key}: {launches} launches, expected {expected}"
            assert len(signatures) == expected, f"{key}: {len(signatures)} signatures, expected {expected}"
            assert len(set(signatures)) == expected, f"{key}: signatures not pairwise distinct: {signatures}"
    assert traced, "no bridge trace recorded"


def test_claim_op_name_receipt_records_the_raise():
    """The `claim_op_name` receipt records that a second in-process build of the same op name
    actually raised, and with the expected error type — so flipping `second_build_raised` to false
    (the PM's negative control) fails here. This guards the recorded A2 observation; the host guard
    that the behaviour holds live is `test_claim_op_name_enforces_one_build_per_process`."""
    claim = json.loads((EVIDENCE / "claim_op_name.json").read_text(encoding="utf-8"))["claim_op_name"]
    assert claim["second_build_raised"] is True, claim
    assert claim["error_type"] == "AclError", claim


def test_block_dim_probe_records_the_reach():
    """The block_dim probe recorded a reach that spans the declared domain and past it — 1, 2 and at
    least one value > 2 — with a `ran` field on every entry. Whether a bd > 2 actually ran is
    observation only (a false `ran` is a valid observation, D-PM-30), so this guards that the record
    is present and complete, not the outcome; dropping the probe or a probed value (the PM's negative
    control) fails here."""
    probe = json.loads((EVIDENCE / "block_dim_probe.json").read_text(encoding="utf-8"))["block_dim_probe"]
    values = {int(k) for k in probe}
    assert {1, 2} <= values, f"probe must cover block_dim 1 and 2: {sorted(values)}"
    assert any(v > 2 for v in values), f"probe must reach past 2: {sorted(values)}"
    for block_dim, entry in probe.items():
        assert "ran" in entry, f"block_dim {block_dim} entry has no 'ran' field: {entry}"


def test_environment_is_a2_with_full_ascriptor_revisions():
    for receipt in _receipts():
        env = receipt.get("environment")
        if env is None:
            continue
        assert env["soc"] == "a2", f"receipt soc={env['soc']!r}, expected a2"
        revisions = env.get("ascriptor_revisions", {})
        for component in ("library", "kernels"):
            revision = revisions.get(component)
            assert isinstance(revision, str) and len(revision) == 40, \
                f"{component} revision not a 40-char sha: {revision!r}"


def test_device_facts_compare_profile_and_measured():
    facts = next((r["device_facts"] for r in _receipts()
                  if r.get("device_facts", {}).get("comparison")), None)
    assert facts is not None, "no device_facts with a profile/measured comparison"
    assert facts["profile"]["profile_device"] == "b3"
    for field in ("cube_cores", "vec_cores"):
        assert facts["comparison"][field]["measured"] is not None, f"{field} not measured"


def test_torch_npu_coverage_all_passed():
    checked, failures = 0, []
    for receipt in _receipts():
        for op, result in receipt.get("coverage", {}).items():
            checked += 1
            if not result.get("ok"):
                failures.append(f"{op}: {result.get('error', 'not ok')}")
    assert checked, "no coverage recorded"
    assert not failures, "\n".join(failures)


def test_claim_op_name_enforces_one_build_per_process():
    """A host-side guard that ``_claim_op_name`` raises (not silently reuses) on a second build of
    the same op name in one process — the behaviour the A2 bring-up relies on. Needs no NPU."""
    from ascend_fla.runtime import binding

    binding._op_name_owner.pop("PmClaimProbe", None)
    binding._claim_op_name("PmClaimProbe", "vendor-tree-a")
    with pytest.raises(binding.AclError):
        binding._claim_op_name("PmClaimProbe", "vendor-tree-b")


def test_evidence_carries_no_machine_paths():
    leaks = []
    for path in sorted(EVIDENCE.glob("*")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for prefix in ("/usr/local/", "/tmp/", "/home/", "/workspace/", "/root/"):
            if prefix in text:
                leaks.append(f"{path.name}: {prefix}")
    assert not leaks, leaks
