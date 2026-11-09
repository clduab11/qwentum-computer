# Qwentum Core-4 evaluation (workstream E)

## Update 2026-11-09: EXP-E-001 retired, EXP-E-002 on Kaggle

- **EXP-E-001 (cached Q8_0 GGUF via the local Unsloth service) is deprecated** by Chris's decision. Its smoke run (`20261108T195352Z-smoke`) completed. Its first full-run attempt was killed by the agent shell after 3 requests, and its second full-run attempt stopped at 376/500 MATH-500 items when the host computer crashed. Both full attempts are kept as incomplete records and are never scored.
- **EXP-E-002** evaluates `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17` (weights file SHA-256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`, measured on Kaggle) with vLLM 0.33.0 on Kaggle Tesla T4 GPUs. Manifest: `experiments/manifests/eval-EXP-E-002.json`, plus amendments 1-4 (delivery mechanics only: harness-file verification on Kaggle, bundle discovery, and an embedded payload after the private Kaggle dataset could not be attached and a 10.5 MB script was rejected).
- **float16, not bfloat16.** Kaggle assigned T4s (compute capability 7.5) in every session, including one that requested an L4. T4s cannot run bfloat16, so vLLM casts the weights to float16 at load. A float32 reference could not be run either (vLLM's Gated DeltaNet kernel rejects float32). float16-vs-reference agreement is therefore **unmeasured** and every EXP-E-002 result carries that caveat.
- Discovery evidence is in `experiments/records/EXP-E-002/discovery-v1..v4/` (not committed: Kaggle raw outputs). On v4: 3/3 smoke prompts correct, 48/48 concurrent requests answered, about 993 generated tokens/s, no garbled output.
- What carries over unchanged from EXP-E-001: harness pin and task files, dataset revisions and file hashes, frozen MMLU-Pro subset, Protocol D samplers, generation caps, truncation rule v2. MATH-500 uses the upstream grader unchanged on Linux; the Windows-safe wrapper below applies only to Windows runs.
- Offline tests on 2026-11-09: evaluation suite 53 passed; project suite 56 passed, 5 skipped.
- **No Core-4 score exists yet.**

The sections below describe the EXP-E-001 package as built on 2026-11-08; the transport, records, tasks and subset code are shared by EXP-E-002.

Status 2026-11-08: protocol package built and tested offline. **No model request has been sent and no Core-4 score exists.** The smoke run and the full baseline each need a separate approval (Linear ADV-49).

This document describes the package in `experiments/eval/`, how it is pinned, what was verified, and what remains unverified. Historical records in `experiments/records/` are untouched; `_arm1_raw.json` still hashes to `15a07a45cef6535a81e953363e416e09cf426989b8a3a39c7509304ca0632774` (rechecked 2026-11-08T19:48:42Z).

## What is measured

Qwentum Core-4, Protocol D (deployment comparison): the model as served, thinking off, greedy, one generation per item.

| Task | Harness task (lm-eval 0.4.13) | Items | Few-shot | Primary metric | Cap (max_tokens) |
| --- | --- | --- | --- | --- | --- |
| MATH-500 | `minerva_math500`, grading wrapped (below) | 500 | 4 fixed Minerva examples | `math_verify` (secondary `exact_match`) | 4096 |
| GSM8K | `gsm8k_cot_llama` | 1319 | 8 fixed | `exact_match`, `strict-match` filter | 2048 |
| IFEval | `ifeval` | 541 | 0 | `prompt_level_strict_acc` | 1280 |
| MMLU-Pro-1k | `mmlu_pro_<subject>` x14, frozen subset | 1000 | 5 per subject from validation | `exact_match`, `custom-extract`, micro-averaged | 2048 |

Core-4 = mean of the four primary metrics on a 0-100 scale. The eight circuit prompts are integration evidence and are not part of this suite.

Sampler fields sent on every request: temperature 0, top_p 1, top_k 1, min_p 0, presence_penalty 0, frequency_penalty 0, seed 20261108, `chat_template_kwargs.enable_thinking=false`, plus `max_tokens` and at most four stop strings. The repeat penalty is not an OpenAI field and is not sent; the server default (1.0) is to be confirmed from the llama-server log at run time. The arm-4 historical request ran with an inherited presence penalty of 1.5; this profile does not.

## Pins

| Item | Value |
| --- | --- |
| Artifact | `unsloth/Qwen3.5-0.8B-GGUF` @ `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0`, `Qwen3.5-0.8B-Q8_0.gguf`, 811,843,840 bytes, SHA-256 `0ad885ff...6a7a6c` (rehashed 2026-11-08T06:12:34Z). Derivation from upstream `2fc06364...8b17` unverified. |
| Harness | `lm-eval==0.4.13`, tag commit `ddd67220430a2470529f25fd5c05a576ca1057a0`, MIT. The 14 task/transport files the suite depends on match that commit byte for byte. |
| Environment | `.venv-eval`, uv CPython 3.11.15; versions recorded in the manifest. The Stim `.venv` (3.14) is separate and unchanged. |
| MATH-500 | `HuggingFaceH4/MATH-500` @ `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`, 500 items. No license field on the card; sources prm800k (MIT) and hendrycks/math (MIT). |
| GSM8K | `openai/gsm8k` @ `740312add88f781978c0658806c59bc2815b9866`, config `main`, test 1319 (train 7473 reserved for development). MIT. |
| IFEval | `google/IFEval` @ `966cd89545d6b6acfd7638bc708b98261ca58e84`, 541 items. Apache-2.0. |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` @ `b189ec765aa7ed75c8acfea42df31fdae71f97be`, test 12032, validation 70. MIT on the card. |
| MMLU-Pro subset | `experiments/manifests/eval-mmlu-pro-1k-subset.json`: 1000 IDs, proportional by category, seed 20261108, ID-list SHA-256 `4de5c852d91c8095b6569b2f7a5cd0c6b976e03002334eef9106cc8e6106e606`. Frozen before any inference. |
| NLTK | `punkt_tab` only, in `experiments/eval/data_cache/nltk_data`. |
| Manifest | `experiments/manifests/eval-EXP-E-001.json`, SHA-256 `c36bacac7d06cbd1d4d19039e3cb2d6d4dc9efa83e5fbb080e5eccece010ba44`. Rebuilt and compared before every run; any difference refuses the run. |

All datasets load offline from `experiments/eval/data_cache/hf`; the package sets `HF_HUB_OFFLINE=1` on import.

## Package layout

| Path | Role |
| --- | --- |
| `experiments/eval/qwentum_eval/pins.py` | Every frozen identity and policy number. Changing it is a protocol change. |
| `guard.py` | Endpoint allowlist (two status GETs and chat completions only; load/unload paths raise) and the authorization rule: active model `unsloth/Qwen3.5-0.8B-GGUF` Q8_0, nothing loading, `is_training_running` false. |
| `transport.py` | Subclass of the harness `local-chat-completions` model. Allowlisted payload, explicit samplers, per-task cap, no credentials or Unsloth extension fields, one retry with every attempt recorded, runtime recheck every 25 requests, wall-clock and truncation stops, likelihood/perplexity refused. |
| `records.py` | Write-once JSONL per run directory: chat attempts (raw request, raw response, hashes, timing, usage), guard checks, events. |
| `math_grading.py` | Windows-safe wrapper (approved option a) around the unchanged upstream `minerva_math` grader. |
| `mmlu_subset.py` | Subset selection and integrity checks. |
| `tasks.py` | Core-4 task configs loaded from the pinned harness YAML with three overrides: dataset revision pinned, MATH grading wrapped, MMLU-Pro filtered to the frozen IDs. |
| `manifest.py` | Build/verify the run manifest. |
| `run_core4.py` | Runner. Refuses without `--i-have-approval`. `--mode smoke` (5 items per task, 15 min) or `--mode full`. Writes to a new `experiments/records/EXP-E-001/<run_id>/`. |
| `freeze_mmlu_subset.py`, `write_manifest.py` | One-shot writers; refuse to overwrite. |
| `tests/test_eval_*.py` | Offline tests against a loopback fake server; skip when `lm_eval` is absent. |

## The Windows grading substitution

Both upstream MATH graders fail silently on Windows: `is_equiv` needs `signal.SIGALRM`, and Math-Verify's Windows timeout spawns a child process that fails (`WinError 6`) and returns False. Under defaults, a correct `0.5` against gold `\frac{1}{2}` scores 0 with no error.

Fix: the upstream `process_results` runs unchanged inside a separate worker process; only inside that process are the two timeouts disabled. The parent enforces a 10 s wall limit per item; on timeout the worker is killed and restarted and the item is scored 0 with `grader_timeout=1`, reported as its own metric. Verified: `\frac{1}{2}` vs `0.5` scores 1 on both metrics; `24` vs `25` scores 0.

One upstream limitation stays: sympy's LaTeX parser reads `x(x+2)` as a function call, so `exact_match` scores it 0 against `x^2+2x` on every platform, while `math_verify` scores 1. That is why `math_verify` is primary.

## Verified offline (2026-11-08)

`experiments/records/eval-offline-pytest-2026-11-08.txt`: 37 passed in `.venv-eval`. The full project suite in the Stim `.venv` reports 56 passed, 3 skipped (the eval modules skip there). Covered:

- Guard refuses a wrong model, a load in progress, a training run, and load/unload/other paths; periodic recheck stops a run when the model changes mid-run.
- Payload has only allowlisted keys, every sampler field, `presence_penalty` 0, thinking off, no `Authorization` or `X-Unsloth-*` header, and the harness's own gen_kwargs cannot override the cap or temperature.
- Every attempt is recorded with request/response hashes; a run directory cannot be reopened; a failed request stops the run after one retry and is never scored.
- Wall-clock and truncation-rate stops; a response from another model is refused.
- Grader equivalence cases and the timeout path; subset determinism, stratification, hash, and tamper detection; manifest verify and drift detection.
- End-to-end through the real harness for all four tasks against the fake server: prompt structure (4/8/0/5 few-shot turns as multi-turn chat), per-task caps, stop strings (GSM8K's fifth stop is dropped and recorded), MMLU-Pro subject filtering to the frozen IDs, and metric plumbing with known right/wrong fake answers. Those numbers are plumbing checks only.

## Not verified yet

- Anything about the real server: stop-string handling, effective repeat penalty, context fit at 5-shot MMLU-Pro + 2048 generation, throughput. The loaded context on 2026-11-08 was 262,144 tokens.
- Item wall time; the 6-8 h estimate scales the historical 65 tok/s decode figure and is not a measurement.
- The MATH-500 license basis for publication.
- Whether the WSP fixed MMLU-Pro subset IDs are obtainable; our subset is our own.

## Acceptance rules proposed for approval

- Baseline integrity: at least 99% of items with a parsed response, zero manifest mismatches, every request and response saved, truncation and refusal rates reported.
- Candidate release: Core-4 gain of at least 1 pp with the paired 95% bootstrap CI lower bound above 0.
- Per-task non-inferiority margin 2 pp (1 pp is not resolvable at these sizes); results inside the margin with a CI crossing it are labeled inconclusive.

## Runtime rules

Direct requests to `http://127.0.0.1:8888/v1/chat/completions` are allowed only for this project. The runner reads both status endpoints first and refuses unless the authorized model is loaded and training is idle; it never loads, unloads or starts a second model. Records are saved under `experiments/records/EXP-E-001/`. Stray scratch files from this session were moved, not deleted, to `experiments/eval/staging-stray-files-2026-11-08/` with a manifest.
