"""Static guard for the A2 (910B3) accumulate discipline behind M10-081.

On the A2 cube two short MMADs that write the same L0C block do not interlock, so the
second read can see an unsettled accumulator (ascriptor M10-081; characterised in A2-01).
The pinned library auto-inserts a PIPE_M settle for FP32 split-K, but the hand-written FP32
accumulate chains and the BF16/FP16 split-K form get no such fix. The a2 units therefore
follow two rules, which A2-03 review checked by hand and this test fixes in place:

1. every ``matmul(..., is_init=False)`` accumulate is immediately preceded by ``barrier(Pipe.M)``;
2. no ``matmul`` uses ``splitk`` with ``m < 64`` (the M10-081 danger band). The check is
   conservative: the operand dtype is not visible in the call, so it flags every sub-64
   split-K rather than only the BF16/FP16 ones the discipline forbids.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

A2_KERNELS_ROOT = pathlib.Path(__file__).resolve().parent.parent / "kernels" / "projects" / "a2"
MIN_SPLITK_M = 64


def _module_int_constants(tree: ast.Module) -> dict[str, int]:
    constants: dict[str, int] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Constant) \
                and isinstance(statement.value.value, int):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = statement.value.value
    return constants


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _keyword_value(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _is_accumulate_matmul(node: ast.AST) -> bool:
    if _call_name(node) != "matmul":
        return False
    is_init = _keyword_value(node, "is_init")
    return isinstance(is_init, ast.Constant) and is_init.value is False


def _is_settle_barrier(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Expr) or _call_name(statement.value) != "barrier":
        return False
    args = statement.value.args
    return (
        len(args) == 1
        and isinstance(args[0], ast.Attribute)
        and isinstance(args[0].value, ast.Name)
        and args[0].value.id == "Pipe"
        and args[0].attr == "M"
    )


def _statement_matmul(statement: ast.stmt) -> ast.Call | None:
    """The matmul ``Call`` a statement is, whether written bare or bound to a name."""
    if isinstance(statement, ast.Expr) and _call_name(statement.value) == "matmul":
        return statement.value
    if isinstance(statement, ast.Assign) and _call_name(statement.value) == "matmul":
        return statement.value
    return None


def _statement_blocks(tree: ast.AST):
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if isinstance(block, list) and any(isinstance(item, ast.stmt) for item in block):
                yield block


def find_barrier_violations(source: str, filename: str) -> list[str]:
    tree = ast.parse(source, filename=filename)
    violations: list[str] = []
    checked_at_statement_level: set[int] = set()

    for block in _statement_blocks(tree):
        for index, statement in enumerate(block):
            matmul = _statement_matmul(statement)
            if matmul is None or not _is_accumulate_matmul(matmul):
                continue
            checked_at_statement_level.add(id(matmul))
            if index == 0 or not _is_settle_barrier(block[index - 1]):
                violations.append(
                    f"{filename}:{matmul.lineno}: is_init=False matmul is not immediately "
                    f"preceded by barrier(Pipe.M)"
                )

    for node in ast.walk(tree):
        if _is_accumulate_matmul(node) and id(node) not in checked_at_statement_level:
            violations.append(
                f"{filename}:{node.lineno}: is_init=False matmul is nested inside another "
                f"expression, so its settle barrier cannot be verified"
            )
    return violations


def find_splitk_violations(source: str, filename: str) -> list[str]:
    tree = ast.parse(source, filename=filename)
    constants = _module_int_constants(tree)
    violations: list[str] = []
    for node in ast.walk(tree):
        if _call_name(node) != "matmul" or _keyword_value(node, "splitk") is None:
            continue
        m_node = _keyword_value(node, "m")
        m_value = None
        if isinstance(m_node, ast.Constant) and isinstance(m_node.value, int):
            m_value = m_node.value
        elif isinstance(m_node, ast.Name):
            m_value = constants.get(m_node.id)
        if m_value is None:
            violations.append(
                f"{filename}:{node.lineno}: splitk matmul has an unresolved m; make it a "
                f"module-level integer so the M>={MIN_SPLITK_M} floor is checkable"
            )
        elif m_value < MIN_SPLITK_M:
            violations.append(
                f"{filename}:{node.lineno}: splitk matmul with m={m_value} < {MIN_SPLITK_M} "
                f"(M10-081 danger band)"
            )
    return violations


def _a2_kernel_sources() -> list[pathlib.Path]:
    return sorted(A2_KERNELS_ROOT.rglob("*.py"))


def test_a2_kernels_present():
    assert _a2_kernel_sources(), f"no a2 kernel sources under {A2_KERNELS_ROOT}"


def test_every_accumulate_has_settle_barrier():
    violations = [
        message
        for path in _a2_kernel_sources()
        for message in find_barrier_violations(path.read_text(encoding="utf-8"), str(path))
    ]
    assert not violations, "\n".join(violations)


def test_no_splitk_below_m64():
    violations = [
        message
        for path in _a2_kernel_sources()
        for message in find_splitk_violations(path.read_text(encoding="utf-8"), str(path))
    ]
    assert not violations, "\n".join(violations)


_MISSING_BARRIER_SOURCE = """
def kernel():
    for step in range(4):
        matmul(l0c_out, l1_qg, l1_h.T, m=L, n=V, k=K)
        matmul(l0c_out, l1_aqk, l1_v.T, m=L, n=V, k=L, is_init=False)
"""

_NESTED_ACCUMULATE_SOURCE = """
def kernel():
    result = wrap(matmul(l0c_out, l1_aqk, l1_v.T, m=L, n=V, k=L, is_init=False))
"""

_SMALL_M_SPLITK_SOURCE = """
BLOCK = 32
def kernel():
    matmul(l0c, l1_a, l1_b, splitk=BLOCK, m=BLOCK, n=L, k=K)
"""


def test_checker_flags_missing_barrier():
    violations = find_barrier_violations(_MISSING_BARRIER_SOURCE, "missing.py")
    assert len(violations) == 1 and "barrier(Pipe.M)" in violations[0]


def test_checker_flags_nested_accumulate():
    violations = find_barrier_violations(_NESTED_ACCUMULATE_SOURCE, "nested.py")
    assert len(violations) == 1 and "nested" in violations[0]


def test_checker_flags_small_m_splitk():
    violations = find_splitk_violations(_SMALL_M_SPLITK_SOURCE, "small.py")
    assert len(violations) == 1 and "danger band" in violations[0]
