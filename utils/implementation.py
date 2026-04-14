import importlib
import os
from pathlib import Path
from typing import Optional

OLD_IMPLEMENTATION = "rl_autoschedular"
NEW_IMPLEMENTATION = "new_rl_autoschedular"

SUPPORTED_IMPLEMENTATIONS = (OLD_IMPLEMENTATION, NEW_IMPLEMENTATION)

IMPLEMENTATION_TO_AGENT_DIR = {
    OLD_IMPLEMENTATION: "old_agent",
    NEW_IMPLEMENTATION: "new_agent",
}

IMPLEMENTATION_TO_BASE_PREFIX = {
    OLD_IMPLEMENTATION: "old",
    NEW_IMPLEMENTATION: "new",
}


def get_autoschedular_impl(default: str = "rl_autoschedular") -> str:
    """Return the selected autoscheduler implementation name."""
    impl = os.getenv("AUTOSCHEDULER_IMPL", default).strip()
    return impl or default


def get_agent_subdir(implementation: Optional[str] = None) -> str:
    """Return canonical agent subdir name under results/<experiment>/ for an implementation."""
    impl = implementation or get_autoschedular_impl()
    if impl not in IMPLEMENTATION_TO_AGENT_DIR:
        raise ValueError(
            f"Unsupported autoscheduler implementation '{impl}'. "
            f"Expected one of {SUPPORTED_IMPLEMENTATIONS}."
        )
    return IMPLEMENTATION_TO_AGENT_DIR[impl]


def get_base_prefix(implementation: Optional[str] = None) -> str:
    """Return canonical baseline filename prefix ('old' or 'new')."""
    impl = implementation or get_autoschedular_impl()
    if impl not in IMPLEMENTATION_TO_BASE_PREFIX:
        raise ValueError(
            f"Unsupported autoscheduler implementation '{impl}'. "
            f"Expected one of {SUPPORTED_IMPLEMENTATIONS}."
        )
    return IMPLEMENTATION_TO_BASE_PREFIX[impl]


def get_agent_runs_root(results_dir: str, implementation: Optional[str] = None) -> Path:
    """Return results/<experiment>/<old_agent|new_agent>."""
    return Path(results_dir) / get_agent_subdir(implementation)


def get_base_file_path(results_dir: str, implementation: Optional[str] = None) -> Path:
    """Return results/<experiment>/exec_times/<old|new>_base.json."""
    prefix = get_base_prefix(implementation)
    return Path(results_dir) / "exec_times" / f"{prefix}_base.json"


def import_autoschedular_module(module: str, implementation: Optional[str] = None):
    """Import a module from the selected autoscheduler implementation package."""
    impl = implementation or get_autoschedular_impl()
    target = impl if not module else f"{impl}.{module}"

    try:
        return importlib.import_module(target)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "")
        if missing not in {impl, target}:
            raise
        raise ModuleNotFoundError(
            f"Could not import autoscheduler implementation '{target}'. "
            f"Set AUTOSCHEDULER_IMPL to one of {SUPPORTED_IMPLEMENTATIONS} "
            f"and ensure the package exists (for example, add {impl}/__init__.py)."
        ) from exc
