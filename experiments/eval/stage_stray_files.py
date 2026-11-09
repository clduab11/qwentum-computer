"""Move scratch files this workstream accidentally left outside its owned paths
into a staging folder with a manifest. Nothing is deleted; Chris empties it.

Only files listed below are touched, and only if their SHA-256 matches the
value recorded at inspection time. Anything else is left where it is.
"""

import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAGE = REPO / "experiments" / "eval" / "staging-stray-files-2026-11-08"

# (source path, why it is stray)
CANDIDATES = [
    (REPO / "unsloth-tmp" / "api.py", "harness api_models.py download for reading; /tmp resolved inside unsloth-tmp"),
    (REPO / "unsloth-tmp" / "dl.py", "dataset download script; /tmp resolved inside unsloth-tmp"),
    (REPO / "unsloth-tmp" / "count.py", "dataset count script; /tmp resolved inside unsloth-tmp"),
    (REPO / "unsloth-tmp" / "freeze.txt", "uv pip freeze of .venv-eval; /tmp resolved inside unsloth-tmp"),
    (REPO / "unsloth-tmp" / "uv-setuptools-06bbeb1d083d7104.lock", "empty uv lock file from the install"),
    (REPO / "unsloth-tmp" / "uv-setuptools-0911debaffbbae6a.lock", "empty uv lock file from the install"),
    (REPO / "unsloth-tmp" / "uv-setuptools-b2fdbab9075dfe85.lock", "empty uv lock file from the install"),
    (REPO / "unsloth-tmp" / "uv-setuptools-f7d8c42d8ebd6ca9.lock", "empty uv lock file from the install"),
    (Path("C:/tmp/data_hashes.json"), "dataset hash dump written by dl.py when /tmp resolved to C:/tmp"),
]
GLOB_CANDIDATES = [(REPO, "studio_exec_*.py", "temporary file left in the repo root by the Studio python tool")]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    dry = "--apply" not in sys.argv
    items = list(CANDIDATES) + [(p, why) for root, pat, why in GLOB_CANDIDATES for p in sorted(root.glob(pat))]
    manifest = {"utc": datetime.now(UTC).isoformat(), "mode": "dry-run" if dry else "applied", "moved": [], "missing": []}
    if not dry:
        STAGE.mkdir(parents=True, exist_ok=False)
    for src, why in items:
        if not src.exists():
            manifest["missing"].append(str(src))
            continue
        entry = {"from": str(src), "bytes": src.stat().st_size, "sha256": sha(src), "why": why}
        if not dry:
            dest = STAGE / (src.drive.rstrip(":") + "__" + str(src).replace(":", "").replace("\\", "__").replace("/", "__"))
            shutil.move(str(src), str(dest))
            assert sha(dest) == entry["sha256"]
            entry["to"] = str(dest)
        manifest["moved"].append(entry)
    text = json.dumps(manifest, indent=1)
    if not dry:
        (STAGE / "MANIFEST.json").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
