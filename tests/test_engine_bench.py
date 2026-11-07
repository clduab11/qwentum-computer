"""Stage 3: frozen held-out circuits and a complete engine log."""

import json
from pathlib import Path

from qwentum.benchmark import HELDOUT_PATH, RECORD_PATH, build_heldout, build_record
from qwentum.seeds import DEVELOPMENT, HELDOUT

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_seed_ranges_are_disjoint_and_written_down():
    text = (REPO_ROOT / "docs" / "benchmark-protocol.md").read_text(encoding="utf-8")
    assert "Development seeds: 0 inclusive through 1000 exclusive." in text
    assert "Held-out seeds: 1000000 inclusive through 1000064 exclusive." in text
    assert "before generating the held-out file" in text
    assert set(DEVELOPMENT).isdisjoint(set(HELDOUT))
    assert DEVELOPMENT == range(0, 1000)
    assert HELDOUT == range(1_000_000, 1_000_064)


def test_heldout_file_uses_only_the_heldout_range_and_required_programs():
    document = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    assert document == build_heldout()
    circuits = document["circuits"]
    seeds = [circuit["seed"] for circuit in circuits]
    assert len(seeds) == len(set(seeds))
    assert set(seeds) <= set(HELDOUT)
    assert set(seeds).isdisjoint(set(DEVELOPMENT))
    for width in range(1, 9):
        kinds = {
            circuit["kind"]
            for circuit in circuits
            if circuit["width"] == width and str(circuit["kind"]).startswith("random")
        }
        assert kinds == {"random-short", "random-deep"}
        for circuit in circuits:
            if circuit["width"] != width or not str(circuit["kind"]).startswith("random"):
                continue
            for op, *args in circuit["program"]:
                if op in {"H", "S", "X", "Y", "Z", "CX", "CY", "CZ"}:
                    assert args[-1] < width
                if op == "CNOT":
                    assert args[0] < width and args[1] < width
    ghz = next(circuit for circuit in circuits if circuit["kind"] == "ghz")
    assert ghz["width"] == 8
    assert ["H", 0] in ghz["program"]
    assert ["CNOT", 0, 7] in ghz["program"]
    assert ghz["program"].count(["M", 0]) >= 1
    idle = next(circuit for circuit in circuits if circuit["kind"] == "idle")
    assert idle["width"] == 8
    gated = {arg for op, *args in idle["program"] if op != "M" for arg in args}
    assert {2, 3, 4, 5, 6, 7}.isdisjoint(gated)
    expected = {
        "reject-t": "UnsupportedOperation",
        "reject-toffoli": "UnsupportedOperation",
        "reject-rx": "UnsupportedOperation",
        "reject-ry": "UnsupportedOperation",
        "reject-rz": "UnsupportedOperation",
        "reject-qubit-8": "RejectedInstruction",
        "reject-cnot-equal": "RejectedInstruction",
        "reject-missing-correction": "RejectedInstruction",
        "reject-length-257": "RejectedInstruction",
    }
    for name, kind in expected.items():
        match = next(circuit for circuit in circuits if circuit["id"] == name)
        assert match["kind"] == "rejection"
        assert match["expect"] == kind


def test_engine_record_lists_every_case_and_keeps_failures():
    saved = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    fresh = build_record(build_heldout())
    assert saved == fresh
    assert saved["stim_version"] == "1.16.0"
    assert saved["prompt"] is None
    assert saved["checkpoint_revision"] is None
    assert saved["decoding_settings"] is None
    assert saved["token_budget"] is None
    assert saved["command"]
    assert [case["id"] for case in saved["cases"]] == [
        circuit["id"] for circuit in build_heldout()["circuits"]
    ]
    assert saved["failures"] == [case for case in saved["cases"] if case["failure_class"] is not None]
    assert saved["failure_count"] == len(saved["failures"])
    for case in saved["cases"]:
        assert case["seed"] in HELDOUT
        assert "failure_class" in case
        assert case["command"] == saved["command"]
        assert case["checkpoint_revision"] is None
