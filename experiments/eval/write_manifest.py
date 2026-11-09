"""Write (once) or verify the EXP-E-001 manifest.

    .venv-eval\\Scripts\\python.exe experiments\\eval\\write_manifest.py           # create, refuses to overwrite
    .venv-eval\\Scripts\\python.exe experiments\\eval\\write_manifest.py --verify  # recompute and compare
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwentum_eval import REPO_ROOT, pins  # noqa: E402
from qwentum_eval.manifest import manifest_digest, verify_manifest, write_manifest  # noqa: E402


def main() -> int:
    target = REPO_ROOT / pins.MANIFEST_FILE
    if "--verify" in sys.argv:
        print(verify_manifest(target))
        return 0
    m = write_manifest(target)
    print(f"wrote {target.relative_to(REPO_ROOT)} sha256 {manifest_digest(m)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
