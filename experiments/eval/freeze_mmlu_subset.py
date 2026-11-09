"""Freeze the MMLU-Pro 1,000-question subset. Run once, before any inference.

Usage (from the repository root):
    .venv-eval\\Scripts\\python.exe experiments\\eval\\freeze_mmlu_subset.py

The output file is opened in exclusive-create mode: an existing frozen subset is
never overwritten. To verify an existing file instead, pass --verify.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwentum_eval import REPO_ROOT, mmlu_subset, pins  # noqa: E402


def main() -> int:
    target = REPO_ROOT / pins.MMLU_SUBSET_FILE
    rebuilt = mmlu_subset.build_from_dataset()
    if "--verify" in sys.argv:
        frozen = mmlu_subset.load_frozen()
        same = frozen["question_ids"] == rebuilt["question_ids"]
        print(f"frozen {frozen['question_ids_sha256']} rebuilt {rebuilt['question_ids_sha256']} identical={same}")
        return 0 if same else 1
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(rebuilt, fh, indent=1)
        fh.write("\n")
    frozen = mmlu_subset.load_frozen()
    print(f"wrote {target.relative_to(REPO_ROOT)}")
    print(f"size {len(frozen['question_ids'])} sha256 {frozen['question_ids_sha256']}")
    print(json.dumps(frozen["category_allocation"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
