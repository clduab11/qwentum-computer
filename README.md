# qwentum-computer

Classical simulation of 8-qubit stabilizer circuits, aimed at a compiled tableau unit inside a checkpoint built from `Qwen/Qwen3.5-0.8B`. This is not a quantum computer, and it does not claim a quantum speedup.

Milestone 0 is the oracle and the documents. There is no tableau implementation and no model checkpoint in this tree. No accuracy result exists yet.

The GitHub remote is https://github.com/clduab11/qwentum-computer.

## Stages

| Stage | What it would show | Status |
| --- | --- | --- |
| 0 | Stim 1.16.0 imports and the 8-qubit Bell support check passes | This milestone |
| 1 | An integer tableau matches Stim on Bell, idle qubits, and rejection | Not started |
| 2 | 8-qubit GHZ and teleportation with feed-forward | Not started |
| 3 | Held-out engine record, failures included | Not started |
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

On 2026-11-07 those commands passed here: Stim printed `1.16.0`, and pytest reported 3 passed. `AGENTS.md` lists that run. Pins, licenses, and unanswered questions are in `docs/provenance.md`. The benchmark has empty result slots in `docs/benchmark-protocol.md`.

## License

MIT. See `LICENSE` and `NOTICE`. Stim, Qwen weights, and the other named upstream projects keep their own licenses.
