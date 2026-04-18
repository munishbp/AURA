"""Sanity-check the environment for aura-ml on a Blackwell (RTX 5090) box.

Run after `uv sync`. Prints a report and exits non-zero only on hard blockers
(missing CUDA, library import failure). Soft warnings (e.g. wrong sm version)
print loudly but exit 0 so the user can investigate.
"""

from __future__ import annotations

import importlib
import platform
import sys

OK = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"

hard_failures: list[str] = []
soft_warnings: list[str] = []


def report(tag: str, msg: str) -> None:
    print(f"{tag}  {msg}")


def check_python() -> None:
    v = sys.version_info
    msg = f"Python {v.major}.{v.minor}.{v.micro} on {platform.system()} {platform.machine()}"
    if v >= (3, 11) and v < (3, 13):
        report(OK, msg)
    else:
        report(WARN, f"{msg} — pyproject targets >=3.11,<3.13")
        soft_warnings.append("python version outside pinned range")


def check_torch_and_cuda() -> None:
    try:
        import torch
    except ImportError as e:
        report(FAIL, f"torch import failed: {e}")
        hard_failures.append("torch missing")
        return

    report(OK, f"torch {torch.__version__} (compiled with CUDA {torch.version.cuda})")

    if not torch.cuda.is_available():
        report(FAIL, "torch.cuda.is_available() == False — no GPU visible")
        hard_failures.append("CUDA unavailable")
        return

    n = torch.cuda.device_count()
    report(OK, f"{n} CUDA device(s) visible")
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        major, minor = torch.cuda.get_device_capability(i)
        cap = f"sm_{major}{minor}"
        line = f"  device {i}: {name} (compute capability {cap})"
        if (major, minor) == (12, 0):
            report(OK, line + " — Blackwell")
        elif (major, minor) >= (9, 0):
            report(WARN, line + " — not Blackwell, but should still work")
            soft_warnings.append(f"device {i} is {cap}, not sm_120")
        else:
            report(WARN, line + " — older arch, may hit kernel issues")
            soft_warnings.append(f"device {i} is {cap}")


def check_lib(name: str, min_version: str | None = None) -> None:
    try:
        mod = importlib.import_module(name)
    except ImportError as e:
        report(FAIL, f"{name} import failed: {e}")
        hard_failures.append(f"{name} missing")
        return
    version = getattr(mod, "__version__", "unknown")
    suffix = f" (>= {min_version} required)" if min_version else ""
    report(OK, f"{name} {version}{suffix}")


def check_bitsandbytes() -> None:
    try:
        import bitsandbytes as bnb
    except ImportError as e:
        report(FAIL, f"bitsandbytes import failed: {e}")
        hard_failures.append("bitsandbytes missing")
        return
    report(OK, f"bitsandbytes {bnb.__version__}")
    try:
        import torch

        if torch.cuda.is_available():
            x = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
            _ = bnb.functional.quantize_blockwise(x)
            report(OK, "bitsandbytes blockwise quantize works on this GPU")
    except Exception as e:
        report(WARN, f"bitsandbytes runtime check failed: {e}")
        soft_warnings.append("bnb runtime check failed")


def main() -> int:
    print("=" * 60)
    print("aura-ml environment check")
    print("=" * 60)
    check_python()
    print()
    check_torch_and_cuda()
    print()
    check_lib("transformers")
    check_lib("diffusers")
    check_lib("peft")
    check_lib("accelerate")
    check_bitsandbytes()
    check_lib("safetensors")
    check_lib("huggingface_hub")
    print()
    print("=" * 60)
    if hard_failures:
        print(f"{FAIL}  {len(hard_failures)} hard failure(s):")
        for f in hard_failures:
            print(f"  - {f}")
    if soft_warnings:
        print(f"{WARN}  {len(soft_warnings)} soft warning(s):")
        for w in soft_warnings:
            print(f"  - {w}")
    if not hard_failures and not soft_warnings:
        print(f"{OK}  environment looks good")
    print("=" * 60)
    return 1 if hard_failures else 0


if __name__ == "__main__":
    sys.exit(main())
