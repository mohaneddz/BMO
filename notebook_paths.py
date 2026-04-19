from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from project_runtime import bootstrap_project


@dataclass(frozen=True, slots=True)
class NotebookPaths:
    root: Path
    assets: Path
    outputs: Path

    def read(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        return path

    def save(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def setup_notebook(*, chdir_to_root: bool = True, output_dir: str = "Assets/notebooks") -> NotebookPaths:
    root = bootstrap_project(chdir=chdir_to_root)
    assets = (root / "Assets").resolve()
    assets.mkdir(parents=True, exist_ok=True)
    outputs = (root / output_dir).resolve()
    outputs.mkdir(parents=True, exist_ok=True)
    return NotebookPaths(root=root, assets=assets, outputs=outputs)


if __name__ == "__main__":
    paths = setup_notebook(chdir_to_root=True)
    print(f"Project root: {paths.root}")
    print(f"Assets dir : {paths.assets}")
    print(f"Output dir : {paths.outputs}")

