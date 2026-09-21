"""ascend_fla.platform 的主机侧测试：SoC 解析、能力表、入口检查。

不需要 NPU。a5 的实测值在这里**写死**（A2-02 只搬家不改值，所以这些常量改了就算回归）。
"""
from __future__ import annotations

import pytest

from ascend_fla import platform


@pytest.fixture(autouse=True)
def _clear_soc_cache(monkeypatch):
    """每个用例前清进程内备忘并抹掉环境变量，用例自己按需设。"""
    monkeypatch.delenv(platform.SOC_ENV, raising=False)
    platform._reset_cache()
    yield
    platform._reset_cache()


# ---- a5 实测值写死（改前的值；变了就是回归） ----
A5_BLOCK_DIM_CHUNK = (1, 2, 3, 4)
A5_BLOCK_DIM_DECODE = (1, 2, 4, 8, 16, 28)
A5_GATE_SPAN = {
    "upstream": {"forward": 80.0, "backward": 80.0},
    "stable": {"forward": 155.0, "backward": 105.0},
}


def test_a5_capability_values_are_frozen():
    cap = platform.capability("a5")
    assert cap["qualified"] is True
    assert cap["supported_block_dim"]["chunk"] == A5_BLOCK_DIM_CHUNK
    assert cap["supported_block_dim"]["decode"] == A5_BLOCK_DIM_DECODE
    assert cap["max_gate_span"] == A5_GATE_SPAN
    assert cap["unit_root"] == "kernels/projects/a5"


def test_a5_accessors_match_ops_kda_constants():
    # 与 ops/kda 的现有常量必须逐字一致 —— platform 是它们的唯一事实源之后的别名来源。
    from ascend_fla.ops.kda.chunk import MAX_GATE_SPAN, SUPPORTED_BLOCK_DIM
    from ascend_fla.ops.kda.fused_recurrent import (
        SUPPORTED_BLOCK_DIM as DECODE_BLOCK_DIM,
    )

    assert platform.supported_block_dim("a5", "chunk") == SUPPORTED_BLOCK_DIM
    assert platform.supported_block_dim("a5", "decode") == DECODE_BLOCK_DIM
    assert platform.max_gate_span("a5") == MAX_GATE_SPAN


@pytest.mark.parametrize("soc", ["a2", "a3"])
def test_unqualified_socs_raise_on_use(soc):
    cap = platform.capability(soc)
    assert cap["qualified"] is False
    assert cap["max_gate_span"] == {}
    with pytest.raises(RuntimeError, match="未验收"):
        platform.require_qualified(soc)
    with pytest.raises(RuntimeError, match="未验收"):
        platform.supported_block_dim(soc, "chunk")
    with pytest.raises(RuntimeError, match="未验收"):
        platform.max_gate_span(soc)


def test_require_qualified_passes_for_a5():
    platform.require_qualified("a5")  # 不抛


def test_resolve_from_env(monkeypatch):
    monkeypatch.setenv(platform.SOC_ENV, "a2")
    assert platform.resolve_soc() == "a2"


def test_explicit_wins_and_is_not_cached(monkeypatch):
    monkeypatch.setenv(platform.SOC_ENV, "a5")
    assert platform.resolve_soc("a2") == "a2"      # 显式实参优先
    assert platform.resolve_soc() == "a5"          # 且不污染缓存


def test_resolution_is_memoized(monkeypatch):
    monkeypatch.setenv(platform.SOC_ENV, "a5")
    assert platform.resolve_soc() == "a5"
    monkeypatch.setenv(platform.SOC_ENV, "a2")     # 缓存后改环境变量不再生效
    assert platform.resolve_soc() == "a5"
    platform._reset_cache()
    assert platform.resolve_soc() == "a2"          # 清缓存后重读


def test_unknown_soc_rejected(monkeypatch):
    monkeypatch.setenv(platform.SOC_ENV, "a9")
    with pytest.raises(ValueError, match="未知 SoC"):
        platform.resolve_soc()


def test_unresolvable_raises_with_how_to_set(monkeypatch):
    # 无环境变量、无 torch_npu：报错里必须带 ASCEND_FLA_SOC 的设法。
    monkeypatch.setattr(platform, "_probe_from_device_name", lambda: None)
    with pytest.raises(RuntimeError) as ei:
        platform.resolve_soc()
    assert platform.SOC_ENV in str(ei.value)


def test_unit_root_points_under_soc(monkeypatch):
    root = platform.unit_root("a2")
    assert root.name == "a2"
    assert root.parent.name == "projects"
