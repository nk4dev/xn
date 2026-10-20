import csv
import glob
import json
from pathlib import Path

# default content_loader written by `xn init`
DEFAULT_LOADER = {
    "separator": "\n\n",
    "sources": [{"path": "data/*.txt", "format": "text"}],
}


def _expand(pattern: str) -> list[Path]:
    p = Path(pattern)
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.is_file())
    files = sorted(Path(f) for f in glob.glob(pattern, recursive=True) if Path(f).is_file())
    if not files:
        raise FileNotFoundError(f"content_loader: no files match {pattern!r}")
    return files


def _read(path: Path, fmt: str, field: str) -> list[str]:
    if fmt == "text":
        return [path.read_text(encoding="utf-8")]
    if fmt == "jsonl":
        with path.open(encoding="utf-8") as f:
            return [json.loads(line)[field] for line in f if line.strip()]
    if fmt == "csv":
        with path.open(encoding="utf-8", newline="") as f:
            return [row[field] for row in csv.DictReader(f)]
    raise ValueError(f"content_loader: unknown format {fmt!r} (text|jsonl|csv)")


def load_documents(loader: dict) -> list[str]:
    """Every document from the content_loader sources, with exact duplicates removed."""
    docs = []
    for src in loader["sources"]:
        fmt = src.get("format", "text")
        field = src.get("field", "text")
        for path in _expand(src["path"]):
            docs += _read(path, fmt, field)
    docs = list(dict.fromkeys(d for d in docs if d.strip()))
    if not docs:
        raise ValueError("content_loader: no documents loaded")
    return docs


def load_content(loader: dict) -> str:
    """Concatenate every document from the content_loader sources into one corpus."""
    return loader.get("separator", "\n\n").join(load_documents(loader))
