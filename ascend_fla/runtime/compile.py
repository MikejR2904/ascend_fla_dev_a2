"""编译层：ascriptor kernel → 可在进程内调用的 CANN 自定义算子。

用 ascriptor 的 ``OpExec`` 完成"生成源码 → CMake/AscendC 编译 → 安装 vendor 树"
这一段（这条路已在 Ascend950PR / CANN 9.1.0 上实测走通），但**不**使用它的 harness
执行路径；产物交给 :class:`ascend_fla.runtime.binding.AclnnOp` 在本进程内直接调用。

两层缓存：

* **磁盘**：``out_dir`` 按 (kernel, device, block_dim, 标量绑定) 的签名分目录，
  ascriptor 自己的 ``.source_hash`` 让源码未变时跳过重编译。
* **进程内**：同一签名只构造一个 ``AclnnOp``，避免重复 ``dlopen`` 同一个 .so。
  签名本身也备忘（``_sig_memo``）—— 它要读源码算 hash，不备忘的话光算缓存键就
  吃掉端到端耗时的 95%。

标量绑定（``bindings``）会进签名：ascriptor 的 PTO 后端会把整型标量特化进生成的
代码，不同形状可能对应不同产物。
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import pathlib
import threading
from typing import Any

from .binding import AclnnOp

__all__ = ["CompiledKernel", "compile_kernel", "default_cache_root"]

_process_cache: dict[str, "CompiledKernel"] = {}
# 签名备忘。算签名要 inspect.getsource + json + sha256，实测每 kernel ~2ms；
# kda_fwd 一次前向要查 5 个 kernel，于是"算缓存键"比缓存省下的还贵 —— 实测占
# 端到端耗时的 95%（见 benchmarks/profile_bridge_overhead.py）。key 里用 id(kernel)
# 是安全的：value 里一并持有 kernel 的强引用，对象不会被回收、id 不会被复用。
_sig_memo: dict[tuple, tuple[Any, str]] = {}
_lock = threading.Lock()


def default_cache_root() -> pathlib.Path:
    """编译产物根目录。

    优先 ``ASCEND_FLA_CACHE``；否则用 ``$TMPDIR/ascend_fla_cache``。远程机器上
    务必让它落在分配给本任务的工作区内 —— 环境脚本应当设好 ``TMPDIR``。
    """
    env = os.environ.get("ASCEND_FLA_CACHE")
    if env:
        return pathlib.Path(env)
    return pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / "ascend_fla_cache"


def _kernel_source(kernel: Any) -> str:
    """kernel 所在模块的源码文本，用于算签名。

    不能只取 decorated entry 的函数体：CCE kernel 常把实际算术放在同文件的 ``@vf`` /
    ``@func`` helper 中。只 hash entry 会在 helper 改动后误命中旧二进制。这个读取只发生在
    每个 kernel 对象第一次编译前，热路径仍由 ``_sig_memo`` 做 O(1) 查询。
    """
    target = getattr(kernel, "fn", kernel)
    try:
        source_path = inspect.getsourcefile(target)
        if source_path:
            return pathlib.Path(source_path).read_text(encoding="utf-8")
    except (OSError, TypeError, UnicodeError):
        pass
    try:
        return inspect.getsource(target)
    except (OSError, TypeError):
        return f"{getattr(target, '__module__', '?')}.{getattr(target, '__qualname__', target)}"


def _signature(kernel: Any, device: str, block_dim: int | None, bindings: dict[str, int], backend: str) -> str:
    payload = json.dumps(
        {
            "kernel": getattr(kernel, "name", getattr(kernel, "__name__", str(kernel))),
            "source": _kernel_source(kernel),
            "device": device,
            "block_dim": block_dim,
            "bindings": dict(sorted(bindings.items())),
            "backend": backend,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _signature_memoized(kernel: Any, device: str, block_dim: int | None,
                        bindings: dict[str, int], backend: str) -> str:
    """:func:`_signature` 的进程内备忘版。语义完全相同，只是不重复读源码。

    进程内源码不会变，所以同一 kernel 对象的签名是常量。这个函数存在的唯一理由是
    让"查缓存"这件事便宜到可以放进每次前向的热路径。
    """
    key = (id(kernel), device, block_dim, tuple(sorted(bindings.items())), backend)
    hit = _sig_memo.get(key)
    if hit is not None:
        return hit[1]
    sig = _signature(kernel, device, block_dim, bindings, backend)
    _sig_memo[key] = (kernel, sig)   # 持有 kernel 强引用，保证 id 不被复用
    return sig


class CompiledKernel:
    """一个已编译的 kernel：持有 aclnn 可调用对象与产物位置。"""

    def __init__(self, op: AclnnOp, vendor_dir: pathlib.Path, out_dir: pathlib.Path,
                 signature: str, spec: Any) -> None:
        self.op = op
        self.vendor_dir = vendor_dir
        self.out_dir = out_dir
        self.signature = signature
        self.spec = spec

    @property
    def input_names(self) -> list[str]:
        return self.op.input_names

    @property
    def scalar_names(self) -> list[str]:
        return self.op.scalar_names

    @property
    def output_names(self) -> list[str]:
        return self.op.output_names

    def __call__(self, inputs: dict, scalars: dict, outputs: dict) -> dict:
        return self.op(inputs, scalars, outputs)

    def __repr__(self) -> str:  # pragma: no cover
        return f"CompiledKernel({self.op.op_name}, sig={self.signature}, out_dir={self.out_dir})"


def compile_kernel(
    kernel: Any,
    *,
    device: str | None = None,
    block_dim: int | None = None,
    bindings: dict[str, int] | None = None,
    backend: str = "cce",
    cache_root: str | os.PathLike | None = None,
    force: bool = False,
) -> CompiledKernel:
    """编译一个 ascriptor kernel，返回进程内可直接调用的对象。

    Args:
        kernel: ascriptor ``@kernel`` 装饰的函数。
        device: ascriptor 设备名。``None``（默认）走 :func:`ascend_fla.platform.resolve_soc`
            按 SoC 解析（``ASCEND_FLA_SOC`` 环境变量 → 设备探测 → 报错，不再悄悄默认
            ``"a5"``）。显式传 ``"a5"`` / ``"a2"`` / ``"a3"`` 时按传入值编译。
            ``"a5"`` → 950 profile（32 cube / 64 vec）；注意 Ascend950PR 实际只有
            28 cube / 56 vec —— ``block_dim`` 超过物理核数会在硬件 barrier 上死锁，
            详见 ascriptor boards.json 的 ``cube_cores``。
        block_dim: 启动的核组数。不传则用 kernel 自带的声明。
        bindings: 整型标量的绑定值，参与编译签名。
        backend: ``"cce"``（默认）或 ``"pto_isa"``。
        cache_root: 产物根目录，默认 :func:`default_cache_root`。
        force: 忽略进程内缓存并强制重新 build。

    Returns:
        :class:`CompiledKernel`。

    Raises:
        RuntimeError: 编译未产出 vendor 树，或 vendor 树里找不到 ``libcust_opapi.so``；
            或 ``device=None`` 且 SoC 无法解析。

    Note:
        本函数只解析 device，不做 SoC 验收门控 —— 验收检查（未验收 SoC 报错）由各算子
        入口（``ops/kda`` 的 ``chunk_kda`` / ``prepare`` / ``fused_recurrent_kda``）在编译
        之前调 :func:`ascend_fla.platform.require_qualified` 完成。直接指定 ``device``
        编译任意 SoC 的 kernel（如 a2 的 GDN-2 单元）不受门控影响。
    """
    if device is None:
        from ascend_fla.platform import resolve_soc
        device = resolve_soc()
    bindings = dict(bindings or {})
    sig = _signature_memoized(kernel, device, block_dim, bindings, backend)

    with _lock:
        if not force and sig in _process_cache:
            return _process_cache[sig]

    from ascriptor.runtime.opexec import OpExec  # 延迟导入：没装 ascriptor 也能 import 本模块

    name = getattr(kernel, "name", getattr(kernel, "__name__", "kernel"))
    out_dir = pathlib.Path(cache_root or default_cache_root()) / f"{name}-{sig}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ex = OpExec(kernel, launcher="aclnn", device=device, block_dim=block_dim,
                backend=backend, bindings=bindings, out_dir=out_dir)
    ex.build(force=force)

    vendor = getattr(ex, "_vendor", None)
    if vendor is None:
        raise RuntimeError(f"{name}: ascriptor 未产出 vendor 树（out_dir={out_dir}）")
    vendor = pathlib.Path(vendor)
    libs = sorted(vendor.rglob("libcust_opapi.so"))
    if not libs:
        raise RuntimeError(f"{name}: vendor 树里没有 libcust_opapi.so（{vendor}）")

    spec = ex.spec
    op = AclnnOp(
        op_name=spec.op,
        opapi_lib=libs[0],
        vendor_dir=vendor,
        input_names=[p["name"] for p in spec.inputs],
        scalar_names=[p["name"] for p in spec.scalars],
        scalar_dtypes=[p["dtype"] for p in spec.scalars],
        output_names=[p["name"] for p in spec.outputs],
    )
    compiled = CompiledKernel(op=op, vendor_dir=vendor, out_dir=out_dir, signature=sig, spec=spec)

    with _lock:
        _process_cache[sig] = compiled
    return compiled
