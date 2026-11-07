# Provenance

Access date for every source below: 2026-11-07. Pins are the revisions fetched that day. This file records claims that were read, not results this repository has reproduced.

Milestone 0 did not download Qwen, did not run a GPU or TPU job, and did not implement a tableau. The only execution added in that milestone is the Stim 1.16.0 sampler test in `tests/test_stim_oracle.py`. Milestone 1 adds an integer destabilizer tableau and compares it to Stim by measurement behavior.

## Pins

- Qwen base, for a later merge: `Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17`, last modified 2026-03-02. Safetensors total 873,438,784 parameters (873,436,192 BF16 and 2,592 FP32). Hub `usedStorage` 1,759,750,582 bytes. License Apache-2.0, copyright 2026 Alibaba Cloud.
- Qwen text config at that revision: hidden size 1024, 24 layers, vocab 248,320, tied embeddings, declared dtype bfloat16, architecture `Qwen3_5ForConditionalGeneration` with a vision config included in the parameter total.
- Qwenputer `main`: `cb1c969cf4e9b6d703c16678be42c9cf3a2d42fe`, committed 2026-11-05. Code license MIT, copyright 2026 Divyansh Agrawal. CHIP-8 ROMs are excluded from that grant.
- Stim release `v1.16.0`, published 2026-05-22, PyPI `stim==1.16.0`. Apache-2.0. `main` had moved to `ffe0d043598ed5df0851a3e6b4920359a23fafbd` (2026-10-29) and is not the pin. The Windows wheel `stim-1.16.0-cp314-cp314-win_amd64.whl` was listed at 2,825,951 bytes.
- Qwenx86 compact: `Quazim0t0/Qwenx86-4B-compact`, Hub revision `20fb583f5456a5850702b8125eb22e7572526bdc`, last modified 2026-11-07. No public GitHub repository by that name was found. Base is Qwen3.5-4B, not the 0.8B model.
- QwenPU: `divagr18/QwenPU` commit `6e2767aaa59b69ea03300c0a5199d7b36debbe50` (2026-11-05). Model card `divagr1925/QwenPU`.
- qwenGPU: `vixhal-baraiya/qwenGPU` commit `e2695f6a0a818ebc68e2e14b6b990eddf04c51b3` (2026-11-01).

## Claims read from those sources

- `docs/q35.md` at the Qwenputer pin reports a CHIP-8 merge of Qwen3.5-0.8B as hidden 2048, 24 Qwen layers plus 15 chip layers, and 2.27B parameters. That number was not remeasured here. The official 0.8B text hidden size is 1024, so 2048 is a widened merge.
- `docs/q35-size.md` says stock Qwen3.5 loading uses one hidden size and one intermediate size for every layer, and its measured byte tables are for a 4B checkpoint. It discusses shrinking an 0.8B chip stack from 16 to 13 layers without giving a new measured 0.8B total.
- Aaronson and Gottesman, arXiv:quant-ph/0406196, describe an efficient tableau for CNOT, Hadamard, phase, and measurement. The abstract page's published field is 2008-06-18. The paper names GHZ and teleportation as uses of the formalism. Its short gate list does not name `Y`; `Y` is a Pauli in that formalism.
- Stim's README says `stim.Circuit` has no T or Toffoli and supports single-control Pauli feedback. The 1.16.0 PyPI page documents `TableauSimulator`, `current_inverse_tableau`, `Circuit.append`, `compile_sampler`, and `sample`.
- Qwenx86's card describes 13 compiled layers on Qwen3.5-4B, a fork of Qwenputer's RV32 core, 64-bit values stored as halves, flag micro-ops, translation of x86-64 into that core, about 9 GB of GPU memory, and 2,000 of 2,000 Unicorn lockstep cases. `x86vm` is described as unshipped. v2 memory tools are host Python. The Hub tags the repo as a finetune; the card says the added weights are compiler output.
- QwenPU's card describes 13 layers, hidden size 5120, 26.7 GB bf16, and trace costs of 2,430,000, 225,000, and 53,000 tokens per story token for plain C, the NPU instructions, and the vector unit. The full checkpoint is described as needing a 40 GB GPU.
- qwenGPU's README describes Qwen3.5-2B plus 45 compiled layers, a 4-lane SIMT machine, host-side PTX lowering, and model-written launch tokens. The author's MPS results say 49 of 49 kernels matched a Tesla T4, text KL was about 7e-4, normal mode scored 20 of 24, and advanced mode scored 4 of 5. About 16 GB free memory is requested.
- Kaggle's TPU page says up to 20 hours per week and up to 9 hours per session, and the same page says that tutorial was written for earlier TPUs Kaggle no longer supports. Its hardware section still describes a TPU v3-8 with 8 cores and 128 GB.
- Google's v5e pages agree on 16 GB HBM per chip and 8 chips per host. The page last updated 2025-05-08 says 819 GBps. The current docs host says 800 GiBps and says usable HBM is lower than nominal. Eight chips are eight memory spaces.

## Local machine

Python 3.14.3 was reported by `py -3` on 2026-11-07 before this milestone's virtualenv. The GPU named for later measurement is an RTX 4060 Ti with 8 GB VRAM and 48 GB RAM. Unsloth Desktop was not inspected.

The GitHub repository https://github.com/clduab11/qwentum-computer is the remote for this project. Milestone 0's local oracle run, on 2026-11-07, printed Stim `1.16.0` and pytest reported 3 passed. That run checks the oracle install. It does not test a tableau. Milestone 1's suite reported 13 passed. After GHZ and teleportation, pytest reported 20 passed.

## Stim 1.16.0 conventions recorded from the installed docstrings

These notes are from `stim==1.16.0` imported in this environment on 2026-11-07. They are API facts, not a copy of Stim's tableau.

Pauli-string left-to-right order. `stim.PauliString.__init__` says a dense string is characters from `_IXYZ`, and its examples map qubit indices onto that string from the left:

- `stim.PauliString({0: "X", 2: "Y", 3: "X"})` is `stim.PauliString("+X_YX")`.
- `stim.PauliString("-X2*Y6")` is `stim.PauliString("-__X___Y")`.
- `stim.PauliString({"X": [1], 2: [4], "Z": [0, 3]})` is `stim.PauliString("+ZX_ZY")`.

The leftmost character is qubit 0. The qubit index increases to the right. `"XZ"` means X on qubit 0 and Z on qubit 1.

A one-qubit check on this install agrees with that reading. After `X` on qubit 0, `peek_observable_expectation` of `"ZI"` is `-1` and of `"IZ"` is `+1`. After `H` then `Z` on qubit 0, `"XZ"` is `-1` and `"ZX"` is `0`.

Forced-measurement API. `TableauSimulator.measure` and `measure_many` return the result and do not take a desired outcome. The methods that document forcing a collapse are:

- `postselect_z(targets, *, desired_value: bool)`. The docstring says postselection forces the qubit to collapse as if it had been measured with that result. `False` is `|0⟩` and `True` is `|1⟩`. It raises `ValueError` when that result is impossible. `postselect_x` uses `False` for `|+⟩` and `True` for `|-⟩`. `postselect_y` uses `False` for `|i⟩` and `True` for `|-i⟩`. `postselect_observable` uses `False` for the `+1` eigenstate and `True` for the `-1` eigenstate.
- `measure_kickback(target)` returns `(result, kickback)`. The docstring's `pseudo_post_select` example forces a desired bit: when the random result differs, `do(kickback)` flips to the other post-measurement state. A deterministic measurement has kickback `None`, and the example raises if the desired result cannot occur.

The Bell tests force Stim with `postselect_z` and compare those bits to the engine. They do not compare tableau matrices.

## Unanswered

- This account's live Kaggle device, usable memory, and remaining quota.
- Unsloth Desktop's actual loaders and export tools.
- A patent search for the tableau algorithm. Publication is not a clearance opinion.
- The language-versus-vision split inside the 873,438,784 parameters.
- A public git commit for Qwenx86, and the neural-n64, neural-x86-doom, and neural-riscv repositories named on its card.
- Any x86-in-weights checkpoint whose base is Qwen3.5-0.8B.
- A newer measured parameter count for a shrunk 0.8B Qwenputer merge.
- Actual VRAM of a local bf16 load. Not run.
- Whether a stock `generate()` loop can host the future circuit without a driver.

## Sources

1. https://github.com/divagr18/qwenputer
2. https://github.com/divagr18/qwenputer/blob/main/docs/q35.md
3. https://raw.githubusercontent.com/divagr18/qwenputer/main/docs/q35-size.md
4. https://raw.githubusercontent.com/divagr18/qwenputer/main/LICENSE
5. https://divagr.com/blog/qwenputer
6. https://huggingface.co/Qwen/Qwen3.5-0.8B
7. https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/config.json
8. https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/LICENSE
9. https://huggingface.co/api/models/Qwen/Qwen3.5-0.8B
10. https://arxiv.org/abs/quant-ph/0406196
11. https://github.com/quantumlib/Stim
12. https://pypi.org/project/stim/1.16.0/
13. https://www.kaggle.com/docs/tpu
14. https://cloud.google.com/tpu/docs/v5e
15. https://docs.cloud.google.com/tpu/docs/v5e
16. https://huggingface.co/Quazim0t0/Qwenx86-4B-compact
17. https://huggingface.co/api/models/Quazim0t0/Qwenx86-4B-compact
18. https://github.com/divagr18/QwenPU
19. https://huggingface.co/divagr1925/QwenPU/blob/main/README.md
20. https://github.com/vixhal-baraiya/qwenGPU
