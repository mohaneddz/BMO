from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from project_runtime import bootstrap_project


ENTRYPOINT_NAMES = ("run.py", "main.py")
EXERCISES_DIR = "exercises"


def discover_exercises(root: Path) -> list[tuple[str, Path]]:
    base = root / EXERCISES_DIR
    if not base.exists():
        return []

    entries: list[tuple[str, Path]] = []
    for file_path in sorted(base.glob("*.py")):
        entries.append((file_path.stem, file_path.resolve()))

    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        for entry_name in ENTRYPOINT_NAMES:
            candidate = folder / entry_name
            if candidate.exists():
                entries.append((folder.name, candidate.resolve()))
                break
    return entries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run exercises with a simple command.")
    parser.add_argument("name", nargs="?", default="all", help="Exercise name or 'all'.")
    parser.add_argument("--list", action="store_true", help="List exercises and exit.")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Forwarded args (prefix with --).")
    return parser


def _run_one(script_path: Path, root: Path, forwarded: list[str]) -> int:
    command = [sys.executable, str(script_path), *forwarded]
    return subprocess.run(command, cwd=root).returncode


def main() -> int:
    root = bootstrap_project(chdir=True)
    args = build_parser().parse_args()

    discovered = discover_exercises(root)
    if args.list:
        if not discovered:
            print("No exercises found in /exercises.")
            return 0
        print("Exercises:")
        for name, script in discovered:
            print(f"- {name}: {script.relative_to(root)}")
        return 0

    if not discovered:
        print("No exercises found in /exercises.")
        return 1

    forwarded = args.script_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    if args.name == "all":
        failed = 0
        for name, script in discovered:
            print(f"[exercise] {name}")
            code = _run_one(script, root, forwarded)
            if code != 0:
                failed += 1
        return 1 if failed else 0

    for name, script in discovered:
        if name.lower() == args.name.lower():
            return _run_one(script, root, forwarded)

    print(f"Exercise '{args.name}' not found. Use `python exercises.py --list`.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

