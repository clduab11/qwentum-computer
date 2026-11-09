# qwentum-computer

**A small, honest experiment: can a tiny open AI model learn to use (and eventually carry inside itself) a calculator for quantum circuits?**

> **Not a quantum computer.** Everything here runs on ordinary computers. Nothing in this repository claims a quantum speedup.

---

## The idea in plain English

Quantum computers are described with *circuits*: lists of simple steps (with names like `H`, `CNOT` and "measure") applied to *qubits*, the quantum version of bits. A useful family of these circuits, called *stabilizer circuits*, can be simulated exactly on a normal computer using a compact bookkeeping table called a **tableau**.

This project has two goals:

1. **Build that simulator ourselves**, small and checkable: 8 qubits, a fixed list of allowed steps, and anything outside that list refused with a clear error.
2. **Connect it to a small AI language model**, `Qwen3.5-0.8B` (about 0.8 billion parameters, small enough for a home PC). The long-term goal is to build the simulator *into* the model's weights, so the model computes the answer itself rather than calling an outside program.

**Qwentum** is our working name for models built from that Qwen base. **Qwentum Core-4** is the test suite we use to check that the model stays generally capable while we change it.

## Where things stand

| Part | Status | What that means |
| --- | --- | --- |
| 8-qubit simulator | ✅ Working | Matches Google's **Stim** (an independent, widely used simulator) on Bell, GHZ and teleportation circuits and on a frozen set of 64 test circuits. It correctly refuses unsupported steps and a 9th qubit. |
| One piece of the simulator in AI-style arithmetic | ✅ Checked | One step of the simulator, re-done in the 16-bit number format AI models use (`bf16`), matches the exact version on every legal input. |
| Model talking to the simulator | ⚠️ Early, mixed | In an 8-prompt trial the model wrote one valid circuit (`M 0`). Run on our simulator, it gave the same answer as Stim. The model wrongly refused 5 of the 8 requests. These 8 prompts are a wiring check, not a benchmark. |
| Baseline test scores (Qwentum Core-4) | ⏳ In progress | The test system is built and tested offline. The first real run is happening on Kaggle's free GPUs. **No score yet.** |
| Simulator built into the model | ⛔ Not started | Requires further approvals and measured memory/size results first. |
| Training / fine-tuning | ⛔ Not started | Planned (see `docs/plan-phase2.md`), but nothing has been trained. |

We only mark something ✅ when a saved test or record proves it. Results that failed, stopped early or were cut short are kept, not deleted.

## How we check the model: Qwentum Core-4

Before changing the model, we measure the unmodified model so we have a fair "before" picture. Core-4 is four well-known public tests:

| Test | What it asks the model to do | Questions |
| --- | --- | --- |
| **MATH-500** | Solve competition-style math problems | 500 |
| **GSM8K** | Solve grade-school word problems | 1,319 |
| **IFEval** | Follow precise formatting instructions ("answer in under 50 words", "use no commas") | 541 |
| **MMLU-Pro (1,000 questions)** | Answer hard multiple-choice questions across 14 subjects | 1,000 (a fixed, pre-chosen sample) |

The **Core-4 score** is the average of the four. Every question, prompt and answer is saved so anyone can check the work. We use the established [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (pinned to version 0.4.13) and its grading code rather than writing our own grader, and we pin every dataset to an exact version.

### Current run: EXP-E-002

| Setting | Value |
| --- | --- |
| Model | `Qwen/Qwen3.5-0.8B`, exact revision `2fc06364715b967f1860aea9cf38778875588b17` (weights file hash recorded) |
| Where it runs | Kaggle's free NVIDIA T4 GPUs, using the vLLM serving engine |
| Number format | **float16**: Kaggle's T4 GPUs can't run the model's native `bfloat16`, so the weights are converted at load time. We could not measure how much this changes answers, and the result will say so. |
| Answer style | Greedy (always the single most likely next word), "thinking" mode off |

An earlier attempt on a local GPU with a compressed (Q8_0) version of the model was **retired** after the computer crashed partway through. Its partial records are kept but never scored.

## Try the simulator tests yourself

You need Windows with Python 3.14 (the commands are the same on other systems, with different paths):

```text
py -3.14 -m venv .venv
.\.venv\Scripts\python -m pip install "stim==1.16.0" pytest
.\.venv\Scripts\python -m pytest
```

On 2026-11-09 this reported **56 passed, 5 skipped**. The 5 skipped tests belong to the separate evaluation setup below and skip automatically when it isn't installed.

The evaluation code needs its own Python 3.11 environment with `lm-eval[api,ifeval,math]==0.4.13` and `pytest`. Its offline tests talk to a fake local server, never to a real model:

```text
.venv-eval\Scripts\python -m pytest tests/test_eval_*.py
```

On 2026-11-09 that reported **53 passed**.

## What's in this repository

| Path | Contents |
| --- | --- |
| `src/qwentum/` | The 8-qubit simulator, the instruction parser, and the language-response scorer |
| `tests/` | Tests for the simulator (checked against Stim) and for the evaluation tools |
| `experiments/eval/` | The Core-4 evaluation code: safety checks, request/response recording, task setup, Kaggle runner |
| `experiments/manifests/` | Frozen run definitions: exact versions, hashes and settings, plus dated amendments |
| `experiments/records/` | Saved results and logs, including failures |
| `docs/` | Plans and details: `plan-phase2.md`, `evaluation-phase2.md`, `benchmark-protocol.md`, `architecture.md`, `provenance.md` |

## Ground rules this project follows

- **Measured, not assumed.** We don't claim a model size, memory fit or score without a saved measurement.
- **Failures stay on the record.** Incomplete or failed runs are labeled, never quietly redone until they pass.
- **Test questions stay separate from tuning.** The final test questions are never used to adjust the model.
- **Credit where due.** Ideas and results from other projects are attributed (see `NOTICE` and `docs/provenance.md`).

## License

Our code is MIT-licensed (see `LICENSE`). Stim, the Qwen model weights, the benchmark datasets and the other projects we reference keep their own licenses (see `NOTICE`). Model weights and benchmark data are not stored in this repository.
