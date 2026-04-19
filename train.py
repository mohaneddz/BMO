from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from project_runtime import bootstrap_project


IGNORE_PARTS = {".git", ".venv", "venv", "__pycache__"}
EXPLICIT_CANDIDATES = (
    "training/train.py",
    "train/main.py",
    "TTS/train.py",
    "STT/train.py",
    "CV/train.py",
    "LLM/train.py",
)


def _skip_path(path: Path) -> bool:
    return any(part in IGNORE_PARTS for part in path.parts)


def discover_train_scripts(root: Path) -> list[Path]:
    found: list[Path] = []

    for rel in EXPLICIT_CANDIDATES:
        candidate = root / rel
        if candidate.exists():
            found.append(candidate.resolve())

    for candidate in root.rglob("train*.py"):
        resolved = candidate.resolve()
        if resolved == Path(__file__).resolve():
            continue
        if _skip_path(resolved.relative_to(root)):
            continue
        if resolved not in found:
            found.append(resolved)

    return sorted(found)


def _pick_script(root: Path, target: str) -> Path | None:
    scripts = discover_train_scripts(root)
    if not scripts:
        return None

    if target == "auto":
        return scripts[0]

    requested = Path(target)
    if requested.is_absolute():
        return requested if requested.exists() else None

    direct = (root / requested).resolve()
    if direct.exists():
        return direct

    lower_target = target.lower()
    for script in scripts:
        rel = str(script.relative_to(root)).replace("\\", "/").lower()
        if rel.endswith(lower_target) or script.stem.lower() == lower_target:
            return script

    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a training entrypoint using project-root paths.",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="auto",
        help="Training script path or name. Default: auto-discover first script.",
    )
    parser.add_argument("--list", action="store_true", help="List discovered training scripts and exit.")
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the chosen training script (prefix with --).",
    )
    return parser


def main() -> int:
    root = bootstrap_project(chdir=True)
    args = build_parser().parse_args()

    scripts = discover_train_scripts(root)
    if args.list:
        if not scripts:
            print("No training scripts found.")
            return 0
        print("Training scripts:")
        for script in scripts:
            print(f"- {script.relative_to(root)}")
        return 0

    selected = _pick_script(root, args.target)
    if selected is None:
        print("No matching training script was found.")
        print("Run `python train.py --list` after adding your train script.")
        return 1

    forwarded = args.script_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    command = [sys.executable, str(selected), *forwarded]
    return subprocess.run(command, cwd=root).returncode


if __name__ == "__main__":
    raise SystemExit(main())

