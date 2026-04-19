from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT_MARKERS: tuple[str, ...] = (".git", ".env.example", "constants.py")
_PROJECT_ROOT_CACHE: Path | None = None


def find_project_root(start: Path | None = None) -> Path:
    base = (start or Path.cwd()).resolve()
    for candidate in (base, *base.parents):
        marker_hits = sum(1 for marker in ROOT_MARKERS if (candidate / marker).exists())
        if marker_hits >= 2:
            return candidate
    return Path(__file__).resolve().parent


def bootstrap_project(*, chdir: bool = False, start: Path | None = None) -> Path:
    global _PROJECT_ROOT_CACHE
    if _PROJECT_ROOT_CACHE is None:
        _PROJECT_ROOT_CACHE = find_project_root(start=start)

    root = _PROJECT_ROOT_CACHE
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    if chdir:
        os.chdir(root)

    return root


def project_path(*parts: str) -> Path:
    return bootstrap_project().joinpath(*parts)

