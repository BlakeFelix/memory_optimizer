from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
import zipfile


def _ensure_cli(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"'{name}' command not found on PATH")


def process_zip(
    zip_path: Path,
    dest_root: Path,
    index: str | None,
    model: str | None,
    no_meta: bool,
    verbose: bool = False,
) -> None:
    dest_dir = dest_root / zip_path.stem
    if dest_dir.exists():
        return
    if verbose:
        print(f"[+] Extracting {zip_path.name}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    for mem_file in dest_dir.rglob("*memory.json"):
        if verbose:
            print(f"[+] Importing {mem_file.name}")
        subprocess.run(["aimem", "import", str(mem_file)], check=True)

    text_logs = list(dest_dir.rglob("*.md"))
    json_logs = [p for p in dest_dir.rglob("*.json") if not p.name.endswith("memory.json")]

    for log_file in text_logs + json_logs:
        cmd = [
            sys.executable,
            "-m",
            "ai_memory.cli",
            "vectorize",
            str(log_file),
            "--vector-index",
            index,
            "--factory",
            "Flat",
        ]
        if model:
            cmd += ["--model", model]
        if log_file.suffix == ".json":
            cmd += ["--json-extract", "messages"]
        if no_meta:
            cmd.append("--no-meta")
        if verbose:
            cmd.append("--verbose")
            print(f"[+] Vectorizing {log_file}")
        subprocess.run(cmd, check=True)


def scan(
    src: Path,
    dest: Path,
    index: str | None,
    model: str | None,
    no_meta: bool,
    verbose: bool = False,
) -> int:
    _ensure_cli("aimem")
    processed = 0
    for zip_file in src.glob("*.zip"):
        if not (dest / zip_file.stem).exists():
            processed += 1
        process_zip(zip_file, dest, index, model, no_meta, verbose)
    return processed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Import memories from ZIP archives")
    parser.add_argument("--src", default="~/Downloads", help="Source directory with ZIP files")
    parser.add_argument("--dest", default="~/chatlogs", help="Destination directory for extracted logs")
    parser.add_argument("--index", default=None, help="Vector index path for vectorize")
    parser.add_argument("--model", default=None, help="Embedding model for vectorize")
    parser.add_argument("--no-meta", action="store_true", help="Skip metadata file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args(argv)

    src = Path(args.src).expanduser()
    dest = Path(args.dest).expanduser()
    src.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    processed = scan(src, dest, args.index, args.model, args.no_meta, args.verbose)
    if processed == 0:
        if args.verbose:
            print("[ℹ] Nothing new to ingest.")


if __name__ == "__main__":
    main()
