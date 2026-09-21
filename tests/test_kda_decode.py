"""decode 路径的主机侧检查（不需要 NPU）。

真机精度与 state 串接由 `benchmarks/verify_decode.py` 验（要 NPU）；这里盯的是
**两边会漂移的常量与门控** —— 那些东西一旦不一致，真机上表现为越界读而不是报错。
"""
from __future__ import annotations

import ast
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
UNIT = ROOT / "kernels/projects/a5/kda_fused_recurrent"


def _kernel_source() -> str:
    return (UNIT / "kernels/step.py").read_text(encoding="utf-8")


def _module_consts(path: pathlib.Path) -> dict:
    """用 ast 读模块级常量赋值 —— **不 import**。

    `fused_recurrent.py` 用相对 import（`from .chunk import ...`），单文件加载不了；
    而 import 整个包会把 torch_npu 拖进来，主机上没有。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    try:
                        out[t.id] = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        pass
    return out


def _op_consts() -> dict:
    # SUPPORTED_BLOCK_DIM is no longer a module-level literal (它从 platform 能力表读，A2-02)，
    # 所以走 import 取值；T_MAX 等仍是字面量，继续走 ast。import 在无 torch_npu 的主机上可用。
    consts = _module_consts(ROOT / "ascend_fla/ops/kda/fused_recurrent.py")
    import ascend_fla.ops.kda.fused_recurrent as fused_recurrent
    consts["SUPPORTED_BLOCK_DIM"] = fused_recurrent.SUPPORTED_BLOCK_DIM
    return consts


def _kernel_calls() -> set[str]:
    """kernel 源码里**实际调用**的方法名（不含注释与 docstring 里的文字）。"""
    names = set()
    for node in ast.walk(ast.parse(_kernel_source())):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute):
                names.add(f.attr)
            elif isinstance(f, ast.Name):
                names.add(f.id)
    return names


def test_t_max_matches_between_kernel_and_wrapper():
    """``T_MAX`` 在 kernel 与算子入口里各写了一遍 —— 必须相等。

    不等的后果是**越界读**而不是报错：入口放过 T=24，kernel 的流式缓冲只有 16 行。
    """
    kernel_t_max = _module_consts(UNIT / "kernels/step.py").get("T_MAX")
    assert kernel_t_max is not None, "kernel 里找不到 T_MAX 的定义"
    assert kernel_t_max == _op_consts()["T_MAX"]
    # 缓冲行数必须用这个常量，不能再写字面量
    assert "DBuff(DT.float, [16," not in _kernel_source(), \
        "kernel 里还有写死的 16，应当用 T_MAX"


def test_block_dim_domain_matches_contract():
    """声明的 block_dim 必须与契约一致，而且每个值都得是实测过的。"""
    contract = json.loads((UNIT / "contract.json").read_text(encoding="utf-8"))
    assert list(_op_consts()["SUPPORTED_BLOCK_DIM"]) == contract["domain"]["block_dim"]
    # 28 是物理上限（56 个向量核），不能再往上声明
    assert max(contract["domain"]["block_dim"]) == 28


def test_no_cross_core_sync_in_kernel():
    """本 kernel 刻意不手写跨核同步 —— 手写同步正是 c1-multihead-o-corrupt 的成因。

    盯两件事：① 用 ``auto_sync()``；② 没有手写的 DEvent / Mutex。
    """
    src = _kernel_source()
    assert "auto_sync()" in src, "没有用 auto_sync()"
    for bad in ("DEvent(", "CvMutex(", "VcMutex("):
        assert bad not in src, (
            f"kernel 里出现了手写同步 {bad} —— 本单元的设计前提是每头一核、核间无同步，"
            f"若确实需要手写同步，要先解释为什么不会重犯 c1-multihead-o-corrupt"
        )


def test_cadd_is_not_used_as_a_broadcast_multiplier():
    """``cadd()`` 的结果只落在 lane 0，不是广播 —— 当乘数用会只有 1/128 个 lane 正确。

    这是实测踩过的（o 相对 L2 8.5e-02 而 final_state 3.2e-08）。现在的实现根本不用
    ``cadd``；这个测试防止有人"优化"时把它加回来而不经过 UB 往返广播。
    """
    assert "cadd" not in _kernel_calls(), (
        "kernel 里出现了 cadd() —— 它的结果只在 lane 0。要当乘数用必须先经 UB 存一次、"
        "再用 .single() 读回广播（见 ascriptor gdn_bwd 的 broadcast_scalar_vf）；"
        "本单元的两趟扫法本来就不需要它，见 README"
    )


def test_decode_gap_records_the_call_overhead_finding():
    gaps = json.loads((ROOT / "docs/matrix/gaps.json").read_text(encoding="utf-8"))
    by = {g["id"]: g for g in gaps["gaps"]}
    assert "decode-call-overhead" in by, "gaps.json 里没有 decode-call-overhead"
    assert by["decode-call-overhead"]["severity"] == "P1"
    # decode 缺口不能再说"算子完全缺失" —— KDA 那半已经做了
    assert "完全缺失" not in by["fused-recurrent-missing"]["title"]


def test_contract_declares_no_gate_span_limit():
    """recurrent 形式结构性地没有门控跨度上限 —— 这一点必须写在契约里。

    chunk 路径有前向 155 / 反向 105 的闸；decode 没有，因为它只用 exp(g_i)。
    调用方据此选路，所以不能只写在 README 里。
    """
    contract = json.loads((UNIT / "contract.json").read_text(encoding="utf-8"))
    span = contract["domain"]["gate_span"]
    assert "无上限" in span and "exp(g_i)" in span
