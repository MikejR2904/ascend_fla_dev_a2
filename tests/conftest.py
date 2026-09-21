"""全套测试共用的 session 级准备。

**为什么必须在这里做**：CANN 只在首次算子解析时读 ``ASCEND_CUSTOM_OPP_PATH``，之后注册的
vendor 树它看不见（``gaps.json`` 的 ``opp-path-read-once``）。pytest 默认把所有测试跑在
**同一个进程**里，所以只要有任何一个测试先执行了 chunk 的算子，后面再编译 decode 的
kernel 就会失败 —— 报 "已经执行过 aclnn 算子，不能再注册新的 vendor 树"。
在测试函数内部调 ``prepare(decode=True)`` **来不及**，实测踩过。

所以这里用一个 session 级、autouse 的 fixture，在任何测试执行之前把
**本仓全部 kernel**（chunk 前向 + 反向 + decode）一次编完。

这不是测试环境的特殊处理 —— **任何宿主程序都有同样的约束**：一个既要 prefill 又要
decode 的服务，必须在启动时调一次 ``prepare(decode=True)``。
"""
from __future__ import annotations

import pathlib
import os
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# 主机侧没有 NPU，SoC 解析探不到设备名。显式声明本仓测试跑在 a5（当前唯一已验收的
# SoC），不靠 platform 的设备探测或悄悄默认（A2-02 / AGENTS.md §7）。setdefault —— 真机上
# 调用方若已设别的 SoC 就尊重它，不覆盖。
os.environ.setdefault("ASCEND_FLA_SOC", "a5")


@pytest.fixture(scope="session", autouse=True)
def _compile_all_kernels_first(request):
    """KDA 真机测试开始前编译全部 kernel；纯模型/CPU 测试不触碰 ascriptor。"""
    kda_device_files = {
        "test_kda_bwd_deep_npu.py",
        "test_kda_bwd_npu.py",
        "test_kda_caches_npu.py",
        "test_kda_decode.py",
        "test_kda_fwd_npu.py",
        "test_kda_layer_npu.py",
    }
    selected_files = {item.path.name for item in request.session.items}
    if not selected_files.intersection(kda_device_files):
        return
    try:
        import torch
        import torch_npu  # noqa: F401
    except ImportError:
        return
    if not torch.npu.is_available():  # pragma: no cover
        return
    from ascend_fla.ops.kda import prepare

    # block_dim 取 1：测试里大多用 1，而一个算子名一个进程只能有一份 build。
    # 需要别的 block_dim 的测试要自己分进程（AGENTS.md §6 铁律二）。
    prepare(block_dim=1, backward=True, decode=True)
