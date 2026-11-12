"""Regenerate SHA256SUMS (sha256sum format, UTF-8, LF, no BOM) over every tracked file of the repository.

usage: python scripts/make_hashes.py [ROOT]   (default: the repository root)
check: sha256sum -c SHA256SUMS

Not hashed: git metadata, caches, packaging metadata and reruns written to outputs/rerun/.
"""
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
skip_dirs = {"__pycache__", ".pytest_cache", ".git", "build", "dist"}
skip_files = {"SHA256SUMS"}
skip_prefixes = ("outputs/rerun/",)
paths = []
for p in root.rglob("*"):
    if not p.is_file():
        continue
    rel = p.relative_to(root).as_posix()
    parts = set(p.relative_to(root).parts)
    if (parts & skip_dirs or any(x.endswith(".egg-info") for x in parts) or rel in skip_files or p.suffix == ".pyc"
            or rel.startswith(skip_prefixes)):
        continue
    paths.append(rel)
paths.sort(key=str.lower)
lines = [f"{hashlib.sha256((root / r).read_bytes()).hexdigest()}  {r}" for r in paths]
(root / "SHA256SUMS").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
print(len(lines), "files")
