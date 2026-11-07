# Benchmark protocol

This protocol is fixed before any score exists. Every arm below is **not run**. No accuracy number in this file is a result.

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

Development seeds and held-out seeds are disjoint ranges. Write the ranges down before generating the held-out file. That file does not exist yet.

Each log stores the seed, prompt, checkpoint revision, decoding settings, token budget, failure class, and command. Measurement comparisons use supported outcomes or a statistical distance. They do not require an uncontrolled shot stream to match another stream.

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
