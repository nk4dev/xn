import importlib
import sys
from pathlib import Path

from xn.config import Config

MIN_PYTHON = (3, 11)


def _python() -> tuple[bool, str]:
    v = sys.version_info
    ok = v >= MIN_PYTHON
    msg = f"{v.major}.{v.minor}.{v.micro} ({sys.executable})"
    if not ok:
        msg += f" - need >= {'.'.join(map(str, MIN_PYTHON))}"
    return ok, msg


def _package(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
    except ImportError:
        return False, "not installed - run: uv sync"
    return True, getattr(mod, "__version__", "installed")


def _device() -> tuple[bool, str]:
    import torch

    return True, "cuda available" if torch.cuda.is_available() else "cpu only"


def _project() -> tuple[bool, str]:
    if not Path("xn.json").exists():
        return True, "no xn.json here (run `xn init` to create a project)"
    try:
        cfg = Config.load("xn.json")
        n = len(cfg.load_text())
    except (SystemExit, Exception) as e:  # noqa: BLE001 - report any config/data error
        return False, f"xn.json: {e}"
    return True, f"xn.json ok, corpus {n:,} chars"


def doctor() -> int:
    checks = [
        ("venv", _python),
        ("torch", lambda: _package("torch")),
        ("numpy", lambda: _package("numpy")),
        ("device", _device),
        ("project", _project),
    ]
    failed = 0
    for name, check in checks:
        try:
            ok, msg = check()
        except Exception as e:  # noqa: BLE001 - a broken check must not hide the others
            ok, msg = False, str(e)
        failed += not ok
        print(f"{'✓' if ok else '✗'} {name:8s} {msg}")
    print("all good" if not failed else f"{failed} problem(s) found")
    return 1 if failed else 0
