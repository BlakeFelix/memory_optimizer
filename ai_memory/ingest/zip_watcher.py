from __future__ import annotations

import argparse
from pathlib import Path
import zipfile
# Workaround for kernel 6.14.0-27 Python subprocess bug: import CLI functions directly
from ai_memory.cli import import_ as cli_import
from ai_memory.cli import vectorize as cli_vectorize


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
        try:
            cli_import.callback(str(mem_file))
        except SystemExit as exc:
            raise RuntimeError(f"aimem import failed: exit code {exc.code}") from exc

    text_logs = list(dest_dir.rglob("*.md"))
    json_logs = [p for p in dest_dir.rglob("*.json") if not p.name.endswith("memory.json")]

    for log_file in text_logs + json_logs:
        if verbose:
            print(f"[+] Vectorizing {log_file}")
        json_extract = "messages" if log_file.suffix == ".json" else "auto"
        try:
            cli_vectorize.callback(
                str(log_file),
                vector_index=index,
                model=model if model is not None else "llama3:70b-instruct-q4_K_M",
                factory="Flat",
                json_extract=json_extract,
                no_meta=no_meta,
                verbose=verbose,
            )
        except SystemExit as exc:
            raise RuntimeError(f"aimem vectorize failed: exit code {exc.code}") from exc


def scan(
    src: Path,
    dest: Path,
    index: str | None,
    model: str | None,
    no_meta: bool,
    verbose: bool = False,
) -> int:
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
