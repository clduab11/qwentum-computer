# Benchmark protocol

The original protocol preceded scoring. A Q8_0 variant now has eight preserved raw language responses and a separately recorded post-hoc host-engine continuation. The original unquantized baseline and embedded arms remain not run. See `experiments/records/unsloth-base-language.json` and `experiments/records/unsloth-q8-load-audit.json`.

## Arms

Use the same pinned prompts and seeds once models exist.

1. Untouched `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`, no tools. Status: not run.
2. The same integration shape with quantum execution switched off, if that shape exists. Widening, extra layers, and added tokens change the model even when the engine is idle, so this arm is not identical to arm 1. Record the difference. Status: not run.
3. Embedded tableau unit. Status: not run.
4. The same base model with a real host call to our engine, or to Stim for an oracle-tool control. Status: not run.
5. Stim 1.16.0 alone, for engine truth. The Milestone 0 smoke test checks only the Bell support and the initial state. It is not this arm. Status: not run.

## Scores, kept separate

- Tableau or observable correctness against Stim.
- Parse of model text into the instruction set.
- Whether the call was routed to an engine.
- Whether the final sentence reports the engine's result.
- Whether that sentence matches Stim.

Unsupported and malformed cases stay in the denominator. A failure of one score is not entered as a failure of another.

## Circuits

The grammar uses only `RESET`, `H`, `S`, `X`, `Y`, `Z`, `CNOT`, computational-basis `M`, and single-bit classical `CX`, `CY`, and `CZ`. Widths are 1 through 8. Include short programs, deeper programs, the 8-qubit GHZ program, at least one program that leaves some qubits idle, and a rejection set (`T`, Toffoli, rotations, qubit 8, and a correction that names a missing measurement).

## Seed ranges

These ranges were written down before generating the held-out file. They do not overlap. Development seeds are not a reported score.

- Development seeds: 0 inclusive through 1000 exclusive.
- Held-out seeds: 1000000 inclusive through 1000064 exclusive.
- Language-prompt seeds: 2000000 inclusive through 2000008 exclusive.

The held-out file may use only the held-out range. It is `experiments/circuits/heldout.json`.

Each log stores the seed, prompt, checkpoint revision, decoding settings, token budget, failure class, and command. Measurement comparisons use supported outcomes or a statistical distance. They do not require an uncontrolled shot stream to match another stream.

## Engine-only record

Stage 3 compares the integer tableau with Stim 1.16.0 on the held-out file. It does not load a checkpoint. Prompt, checkpoint revision, decoding settings, and token budget are recorded as not applicable. The log is `experiments/records/engine-benchmark.json`. Cases that disagree stay in the `failures` list. This record is not arm 5.

## Resources to record with any future run

Peak RAM, peak VRAM, latency, compile time, steady-state tokens per instruction, total parameters, learned parameters, and nonzero circuit parameters. After a merge exists, a small fixed language set is the regression check.

The Qwen model card's own non-thinking MMLU-Pro for this base is 29.7, and its thinking-mode BFCL-V4 is 25.3. Those are Qwen's published numbers. They are not scores for this repository.

## Results

| Arm | Engine | Parse | Routing | Faithfulness | End to end | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 Base | not run | not run | not run | not run | not run | |
| 2 Execution off | not run | not run | not run | not run | not run | Control is imperfect once the shape changes |
| 3 Embedded | not run | not run | not run | not run | not run | |
| 4 External tool | not run | not run | not run | not run | not run | |
| 5 Stim | not run | n/a | n/a | n/a | not run | Milestone 0 smoke test is not this arm |
| 1 Q8_0 variant | no execution | 1/8 strict format | 0/8 host calls | not assessable | not assessable | Original raw responses preserved |
| 4 Q8_0 post-hoc continuation | 1/1 executed program agrees with Stim | reuses arm-1 outputs | 1/8 host calls | pending | pending | Automatic dispatch, not autonomous tool selection |

## Q8_0 continuation contract and limitations

The eight original prompts and responses in `_arm1_raw.json` are immutable evidence. Seven responses are bare `REJECT`, which violates the requested `REJECT <reason>` format; two nevertheless have the appropriate rejection intent (T and qubit 8). Five reject supported requests. The remaining `M 0` satisfies the short-program task. Task-intent success (3/8) is not strict format success (1/8).

The strict host parser accepts only the documented uppercase instructions, integer operands, and a nonempty rejection reason. It does not strip code fences, repair circuits, or execute generated Python. It validates width and measurement references before dispatch. A valid program is passed to `Machine.run`; Stim executes that same program independently with identical injected bits only for random measurements. HALT and RESET preserve the engine's semantics. Rejection text is not an engine exception and does not trigger an engine call.

The original prompt required program-only output. It did not request a final sentence, so original faithfulness/end-to-end scores are null. The separate arm-4 continuation executed the preserved `M 0`: engine `[0]`, Stim `[0]`. No generated circuit was substituted. Its report follow-up remains pending. The original teleportation prompt does not specify preparation/readout sufficient for a general state-transfer claim; future prompt revisions must be labeled separately. These eight prompts have been inspected and are not a fresh held-out evaluation.

The next report request will show only the host result to the model, not the oracle answer, and ask for `Engine measurement bits: [0].` for the actual successful case. Exact sentence match is the declared reporting check; deviations are retained. This is result reporting, not independent derivation. Task-aware end-to-end requires task correctness as well as a faithful oracle-matching report.

Continuation model requests must use LiteLLM. Its current model listing has no Qwen3.5-0.8B route. Do not use a different model's alias, bypass the proxy, change settings, or auto-load another model. The original direct-port requests are retained with that transport deviation recorded. No new model requests were made in this continuation.

Update 2026-11-09: the arm-4 report request was later sent directly to the local Unsloth endpoint under a project-specific direct-access exception granted by Chris on 2026-11-07. The model returned `Engine measurement bits: [0]`, which reports the engine result faithfully but omits the requested final period, so the exact-format check failed. That request inherited a server presence penalty of 1.5; it is not a penalty-free greedy result. Records: `experiments/records/q8-arm4-report-v2-*.json`. General benchmark evaluation moved to Qwentum Core-4 (`docs/evaluation-phase2.md`).

Requested original decoding: temperature 0, top_p 1, top_k 1, min_p 0, max_tokens 512, reasoning disabled, tools disabled. Installed backend source forwards these sampling controls and reasoning kwargs; the historical log has no per-request sampler dump. Responses report zero reasoning tokens, stop termination, and the loader log records no truncation. Effective sampling is not fully verified. A refusal is not by itself evidence of a loader or fidelity problem.

The loaded artifact is Q8_0, not untouched BF16. Its upstream base revision is not verified. Loader evidence and exact artifact hash are in `unsloth-q8-load-audit.json`. Peak model VRAM and RAM are unmeasured. Hand-authored Bell/GHZ/teleportation harness tests are labeled fixtures and never counted as language-model successes. No weight merge, training, or new model download is part of this continuation.
