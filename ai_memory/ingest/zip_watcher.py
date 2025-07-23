from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
import zipfile


def _ensure_cli(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"'{name}' command not found on PATH")


def process_zip(zip_path: Path, dest_root: Path, index: str | None, model: str | None) -> None:
    dest_dir = dest_root / zip_path.stem
    if dest_dir.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    for mem_file in dest_dir.rglob("*memory.json"):
        subprocess.run(["aimem", "import", str(mem_file)], check=True)

    text_logs = list(dest_dir.rglob("*.md"))
    json_logs = [p for p in dest_dir.rglob("*.json") if not p.name.endswith("memory.json")]

    for log_file in text_logs + json_logs:
        cmd = [
            "aimem",
            "vectorize",
            str(log_file),
            "--vector-index",
            index,
            "--factory",
            "Flat",
            "--model",
            model,
        ]
        if log_file.suffix == ".json":
            cmd += ["--json-extract", "messages"]
        subprocess.run([c for c in cmd if c], check=True)


def scan(src: Path, dest: Path, index: str | None, model: str | None) -> None:
    _ensure_cli("aimem")
    for zip_file in src.glob("*.zip"):
        process_zip(zip_file, dest, index, model)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Import memories from ZIP archives")
    parser.add_argument("--src", default="~/Downloads", help="Source directory with ZIP files")
    parser.add_argument("--dest", default="~/chatlogs", help="Destination directory for extracted logs")
    parser.add_argument("--index", default=None, help="Vector index path for vectorize")
    parser.add_argument("--model", default=None, help="Embedding model for vectorize")
    args = parser.parse_args(argv)

    src = Path(args.src).expanduser()
    dest = Path(args.dest).expanduser()
    src.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    scan(src, dest, args.index, args.model)


if __name__ == "__main__":
    main()
