# Agent invariants

- This repository simulates stabilizer circuits classically. It is not a quantum computer.
- The host, when there is one, is built from `Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17`. Do not call the result an 0.8B model or an 8 GB fit without a measurement in `experiments/records/`.
- Stim is an oracle pinned at 1.16.0. It is not the embedded implementation.
- Supported width is 8. Qubit 8 and unsupported gates fail visibly. The embedded machine is the tableau unit, not an x86 or CUDA GPU.
- Do not download checkpoints, train, push, or edit Unsloth unless the current milestone says so.
- Do not weaken a test to obtain a pass. After three failed repairs of the same check, stop.
- Verified commands, from `C:\Users\cld-main\Desktop\github-projects\qwentum-computer` on 2026-11-07. Python 3.14.3 and Stim `1.16.0`. Milestone 0 reported 3 passed. After the bf16 row gadget, pytest reported 28 passed:

```text
py -3.14 -m venv .venv
.\.venv\Scripts\python -m pip install "stim==1.16.0" pytest
.\.venv\Scripts\python -c "import stim; print(stim.__version__)"
.\.venv\Scripts\python -m pytest
```
