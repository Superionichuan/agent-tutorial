"""The version must be the same everywhere it is written, and on a release tag v<X> it must be X.

pyproject.toml, mini_agent/__init__.py, install.sh and install.ps1 each name it; the install
scripts download the release wheel of that version, so a mismatch breaks the one-line install.
"""
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PLACES = {
    "pyproject.toml": r'^version\s*=\s*"([^"]+)"',
    "mini_agent/__init__.py": r'^__version__\s*=\s*"([^"]+)"',
    "install.sh": r'^VERSION="([^"]+)"',
    "install.ps1": r"^\s*\$Version\s*=\s*'([^']+)'",
}

found = {}
for path, pattern in PLACES.items():
    match = re.search(pattern, (ROOT / path).read_text(encoding="utf-8"), re.M)
    if match is None:
        sys.exit(f"{path}: no version found")
    found[path] = match.group(1)

ref = os.environ.get("GITHUB_REF", "")
if ref.startswith("refs/tags/v"):
    found["tag"] = ref[len("refs/tags/v"):]

for place, version in found.items():
    print(f"{place}: {version}")
if len(set(found.values())) != 1:
    sys.exit("The versions differ; make them the same before releasing.")
print("All versions agree.")
