# qwentum-computer

Classical simulation of 8-qubit stabilizer circuits, aimed at a compiled tableau unit inside a checkpoint built from `Qwen/Qwen3.5-0.8B`. This is not a quantum computer, and it does not claim a quantum speedup.

Milestone 1 is an integer 8-qubit destabilizer tableau. Milestone 2 runs GHZ and teleportation on that machine. Milestone 3 freezes a held-out circuit list and logs the engine against Stim, including the rejection cases. There is no model checkpoint in this tree. No model-accuracy result exists yet. The checks compare measurement bits with Stim.

The GitHub remote is https://github.com/clduab11/qwentum-computer.

## Stages

| Stage | What it would show | Status |
| --- | --- | --- |
| 0 | Stim 1.16.0 imports and the 8-qubit Bell support check passes | Passed |
| 1 | An integer tableau matches Stim on Bell, idle qubits, and rejection | Passed |
| 2 | 8-qubit GHZ and teleportation with feed-forward | Passed |
| 3 | Held-out engine record, failures included | Passed |
| 4 | One bf16 row gadget matches the integer function on its whole domain | Not started |
| 5 | The untouched base model calls an external simulator | Not started; needs a download approval |
| 6 | Tableau layers merged into a checkpoint built from the 0.8B base | Not started; needs a download approval |

A passing stage does not prove the next one. Describe any future merged system as built from the 0.8B base. That phrase is not a claim that the finished system has 0.8B parameters or fits an 8 GB GPU.

## Oracle check

From this directory, with Python 3.14:

```text
py -3.14 -m venv .venv
.\.venv\Scripts\python -m pip install "stim==1.16.0" pytest
.\.venv\Scripts\python -c "import stim; print(stim.__version__)"
.\.venv\Scripts\python -m pytest
```

On 2026-11-07 the oracle commands passed here: Stim printed `1.16.0`, and the Milestone 0 pytest run reported 3 passed. After the held-out engine record, pytest reported 25 passed. `AGENTS.md` lists the oracle run. Pins, the Stim Pauli order, and unanswered questions are in `docs/provenance.md`. The model arms in `docs/benchmark-protocol.md` are still not run. The engine-only log is `experiments/records/engine-benchmark.json`.

## License

MIT. See `LICENSE` and `NOTICE`. Stim, Qwen weights, and the other named upstream projects keep their own licenses.
