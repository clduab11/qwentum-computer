# Phase 2: Qwentum model improvement and Qwentum Core-4 evaluation

Status: finalized planning document; approved for saving, not blanket execution approval.
Planning date: 2026-11-07 (America/Los_Angeles). Some evidence timestamps fall on 2026-11-08 UTC.
Repository: `C:\Users\cld-main\Desktop\github-projects\qwentum-computer`.

## 1. Charter, decisions, and immediate priority

Develop a useful Qwen-derived model through two complementary tracks:

- **Track A:** reproduce a published targeted weight repair, then compare repair alone, controlled Unsloth LoRA alone, and their combination.
- **Track B:** investigate a trainable Clifford-enhanced tensor-network residual adapter against parameter- and budget-matched classical controls.

The first required result is **our own plain, unmodified Qwen3.5-0.8B-derived Q8_0 baseline using the already-downloaded artifact**. Do not substitute published BF16 numbers, another quantization, the repaired model, or the eight circuit prompts for that baseline. There are no new public-benchmark results from this planning conversation.

### Decisions Chris confirmed

1. Continue in this existing repository; do not create another project.
2. Pursue both tracks, with research and planning preceding execution.
3. Use an established GitHub evaluation harness. Primary: [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Keep the original Weight-Space Programming (WSP) protocol as a separate reproduction track.
4. **Qwentum** is the working model-family name. **Qwentum Core-4** names the four-benchmark evaluation suite: MATH-500, GSM8K, IFEval, and MMLU-Pro. Four is the number of benchmarks, not qubits, experts, layers, or billions of parameters.
5. Benchmark the existing Q8_0 artifact first, then compare future modified artifacts at matched quantization.
6. Save this document as `docs/plan-phase2.md` and provide a new-conversation handoff.

The numerical release thresholds, budgets, dataset choices beyond the core suite, and training hyperparameters below remain recommendations for explicit preregistration. Approval of the plan does not authorize downloads, installs, inference, model switching, training, remote execution, publishing, or MemPalace filing.

### Credible release objective

Release a reproducibly evaluated checkpoint if it improves the approved primary metric while meeting capability and deployment constraints. Publish code, evaluation configurations, provenance, measured resource use, and per-task gains/regressions. Attribute the WSP repair as reproduction; possible original contributions are our controlled repair-plus-training study, the adapter design and ablations, and verified deployment results. Novelty remains a research question, not a claim already established by a literature search.

An attributed repair reproduction is a useful intermediate artifact, not by itself proof that we invented an improved model. Do not claim that a non-thinking or quantized reproduction is the first of its kind without a dedicated search.

This repository simulates stabilizer circuits classically. It is not a quantum computer and demonstrates no quantum speedup. No final model parameter count or 8 GB fit may be claimed without a measured artifact record.

## 2. Inspected state and preservation requirements

### Repository state at planning

Branch `main` tracked `origin/main`, HEAD `b6c68d0`. Nothing was staged. Existing modified files were `.gitignore`, `README.md`, and `docs/benchmark-protocol.md`. Existing untracked work included `src/qwentum/language_eval.py`, `tests/test_language_eval.py`, the two Q8 continuation/finalization scripts, records, OpenAPI files, and `unsloth-tmp/`. Preserve all of it. Do not stage, move, delete, commit, or publish unrelated files under this plan.

Read the active workspace and repository `AGENTS.md` before implementation. Those instructions govern runtime authority; this document does not override them. In particular, a model change needs approval and must not interrupt an active run. Soft-delete only, and commit/push only on request.

### What exists

- Classical integer destabilizer tableau, width 8, qubits 0 through 7; unsupported gates and qubit 8 visibly rejected.
- Stim 1.16.0 as independent oracle, not the embedded engine.
- Bell, GHZ, teleportation/feed-forward and rejection tests; frozen engine circuit list and engine record.
- A software bf16-arithmetic CNOT row gadget checked against integer updates; not a CUDA BF16 kernel or merged model.
- Strict language parser and separate task-intent, routing, engine, oracle, semantic-report and exact-format scores.
- Q8_0 inference artifact and original language evidence.

No compiled-tableau checkpoint, weight merge, Track A fine-tune, or Track B trainable adapter exists in the inspected tree.

### Immutable language evidence

Preserve `experiments/records/_arm1_raw.json` byte-for-byte:

```text
SHA-256: 15a07a45cef6535a81e953363e416e09cf426989b8a3a39c7509304ca0632774
```

This hash was freshly rechecked while saving this plan. Seven outputs are bare `REJECT`; one is `M 0`. Strict parsing is 1/8; task intent is 3/8, including two appropriate unsupported-task rejection intents; five supported tasks were rejected. Original arm-1 engine calls: zero.

The separately recorded arm-4 continuation reused `M 0`, called the real `qwentum.tableau.Machine.run`, and independently checked it with Stim 1.16.0. Both returned `[0]`. The model saw only the engine result and returned `Engine measurement bits: [0]` without the requested final period. Semantic faithfulness and report/oracle agreement passed; exact format and strict task-aware end-to-end failed. No repaired answer or retry replaced it. This was automatic host dispatch, not autonomous tool selection or embedded execution.

Relevant evidence:

- `experiments/records/unsloth-base-language.json`
- `experiments/records/unsloth-q8-load-audit.json`
- `experiments/records/q8-arm4-report-v2-*.json`
- `experiments/finish_q8_report.py`
- `experiments/finalize_q8_evaluation.py`

Do not rerun these scripts casually: the first can issue inference; the second historically rewrites a derived summary and uses machine-specific paths. Historical summaries and preserved prior-record snapshots must not be overwritten to simulate compliance with later rules. New verification produces new records.

The saved `language-eval-pytest.txt` reports **48 passed**, including 20 language-evaluator cases, before subsequent evaluator additions. The prompt also reports a later targeted run of 28 language-evaluator tests; no fresh full-suite result was obtained in this planning conversation. Do not update historical 28-test milestone evidence to a larger number retroactively. Closeout requires a new full-suite run with saved output.

### Documentation closeout still needed

`README.md` and `docs/benchmark-protocol.md` still describe the model report as pending and LiteLLM as mandatory. Their current-status sections must acknowledge the completed continuation and the project-specific direct-access exception, while retaining historical transport deviations. The sampler audit records inherited presence penalty 1.5 for the report request; this was not a penalty-free greedy baseline. Architecture/provenance documents also predate the new training and adapter direction.

Saving this plan does not itself authorize edits to those files. Assign them to the coordinator/E owner before editing. MemPalace result filing remains paused pending explicit permission.

## 3. Model lineage and baseline identity

### Existing baseline artifact: no replacement download required

```text
Repository: unsloth/Qwen3.5-0.8B-GGUF
Artifact revision: 6ab461498e2023f6e3c1baea90a8f0fe38ab64d0
Variant: Q8_0
Path: C:\Users\cld-main\.cache\huggingface\hub\models--unsloth--Qwen3.5-0.8B-GGUF\snapshots\6ab461498e2023f6e3c1baea90a8f0fe38ab64d0\Qwen3.5-0.8B-Q8_0.gguf
Bytes: 811843840
Historical SHA-256: 0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c
```

File existence and size were freshly checked when saving. Rehash before the benchmark and record the result; a blob filename alone is not a fresh hash verification. The exact derivation of this GGUF from the pinned upstream revision remains unverified. Label it an unmodified cached Q8_0 deployment baseline, not untouched BF16 or a verified conversion of that upstream pin.

### Trainable source, separately pinned

```text
Qwen/Qwen3.5-0.8B
Revision: 2fc06364715b967f1860aea9cf38778875588b17
```

The small text configuration is dense, not MoE. Recorded text dimensions are hidden size 1024, 24 layers, MLP intermediate 3584, with hybrid Gated DeltaNet and gated attention. The recorded full multimodal checkpoint count is 873,438,784 parameters; do not substitute that count for a text-only export or a final augmented model. Inventory tensor names, shapes, tied storage, dtype and language/vision split before calculating run budgets.

The GGUF is inference-only for the planned Studio workflow. Repair reconstruction and LoRA require a separate safetensors source; dequantizing the Q8_0 file is not an exact reconstruction of the published BF16 source.

### Required baseline comparisons

1. **BASE-Q8-CACHED:** existing unmodified Q8_0 artifact, required first public-benchmark measurement.
2. **BASE-BF16-PINNED:** pinned upstream trainable checkpoint after download approval.
3. **BASE-Q8-MATCHED:** convert that pinned source with the same pinned conversion/quantization toolchain used for every modified candidate.
4. Candidate BF16 against BASE-BF16-PINNED; candidate Q8_0 against BASE-Q8-MATCHED; also report the cached-baseline comparison Chris requested.

Q8_0 alone does not make two conversions identical. Track converter commit, precision before quantization, tensor exclusions, tokenizer, chat template, EOS and runtime. If cached and newly converted baselines differ, report the difference rather than treating their lineage as identical.

## 4. Evaluation architecture: reuse established harnesses

### Primary harness

Use a pinned commit of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), MIT license. Its [API guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/API_guide.md) documents `local-chat-completions` for existing OpenAI-compatible endpoints. No second model process or direct GGUF binding is permitted.

Use existing task loaders, prompts, filters and graders wherever suitable. Add a small project-owned transport wrapper only for runtime guards, explicit sampler fields, request/response capture, and compatibility with the current endpoint. Add narrowly scoped task configurations only when the exact research dataset/protocol is missing. Do not rewrite benchmark graders from scratch unnecessarily.

Compatibility gate before implementation approval is converted into execution:

1. Pin a harness commit and inspect actual task YAML/Python, not names alone.
2. Verify that each selected task uses generated answers (`generate_until`) rather than likelihood scoring.
3. Offline mock test: rendered messages, sampler settings, stop sequences, token cap, reasoning kwargs, logging, transport errors, resume and cache identity.
4. After inference approval, one task smoke against the existing service; inspect both request and response records.
5. Confirm supported sampler fields reach the backend. Do not assume the SDK forwards arbitrary kwargs.

Chat completions do **not** provide the required loglikelihood interface for ordinary likelihood-based tasks or perplexity. MMLU-Pro's inspected generation template is compatible in principle, but defaults to five-shot and its own extraction regex. Do not silently describe it as zero-shot or as the original WSP protocol. WikiText perplexity is deferred to an approved HF/in-process backend and is not a prerequisite for the cached chat-endpoint baseline.

### Research reproduction reference

Use [RikiyaT/lm_understand-logs/exp8_math](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math) as the source of the published `think35` protocol, dataset identity, sample counts and repair manifests. Some imported IFEval/multiple-choice graders are not included, and execution uses vLLM. It is not a turnkey drop-in local benchmark package. Reconcile graders with their official implementations and document substitutions.

[Hugging Face lighteval](https://github.com/huggingface/lighteval) is the alternative, not a second mandatory implementation. It has broad task and custom-model support; switch only for an identified compatibility advantage, preserving a frozen comparable protocol.

### Dataset identity matters

Inspected harness task `minerva_math500` uses `HuggingFaceH4/MATH-500`. The harness `minerva_math` family is not automatically the separate research Minerva dataset. WSP's script uses `math-ai/minervamath` (272 items), `math-ai/amc23` (40), and `math-ai/olympiadbench` (674 English text-only items). Pin these identities and actual item counts after approval; do not infer equivalence from task names.

## 5. Protocols, metrics and evaluation tiers

### Protocol D: deployment comparison

Recommended starting profile, frozen before final evaluation:

- Tools off; thinking off with the pinned tokenizer template and `chat_template_kwargs.enable_thinking=false`.
- Greedy: temperature 0, top_p 1, top_k 1, min_p 0; explicitly set presence and frequency penalties to 0 and runtime-supported repetition penalty to 1.
- Default generation cap 2048; MATH-500 and advanced math 4096. These are proposals, not historical settings or claims of sufficiency.
- Context must fit rendered input plus generation budget. The historical slot was 2048; long-output testing therefore needs a separately approved context configuration and memory measurement. Do not silently truncate to fit the old slot.
- Few-shot counts, prompt rendering, answer extraction, stop conditions and sample counts are task-specific and pinned. For the initial Core-4 profile, recommend the selected harness task's documented few-shot protocol; any override gets its own named profile.
- One generation per item for the primary deterministic profile. Use paired item comparisons and resample items, not correlated samples as independent observations.

A separate development-only sampling study can compare model-card settings, presence penalties and token budgets. Freeze any winning profile before final testing, then evaluate both original and modified weights under that same profile. Sampling changes are not weight engineering. Current model cards can change; save the exact card revision instead of treating one setting as timeless.

### Protocol R: WSP `think35` reproduction

Third-party protocol, separate from D: thinking on, temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, up to 32,768 new tokens, published prompts and last-boxed math grading. Samples per item: MATH-500 4; GSM8K 2; IFEval 2; AMC23 16; Minerva 4; OlympiadBench 4; GPQA Diamond 4; fixed MMLU-Pro 1,000 subset 1. Report mean sample accuracy (average pass@1), not pass@N.

The published card describes a 43.4-million-generated-token MATH-500 rerun. Do not extrapolate H200 timings to this host. Benchmark a small fixed pilot to estimate wall time before scheduling the complete reproduction. Any truncated budget or backend substitution is a partial reproduction, labeled as such.

### Qwentum Core-4

Macro-average the four task accuracies, each on a 0-100 scale:

`Core-4 = (MATH-500 + GSM8K + IFEval strict prompt-level + MMLU-Pro-1k) / 4`.

Publish all four scores, item counts, prompt profile and CIs alongside the aggregate. Freeze MMLU-Pro subset IDs and seed before inspecting results; stratify by subject and predeclare micro-accuracy within that subset. Do not choose the subset after seeing performance. If extending to full MMLU-Pro, report it as a separately specified profile.

### Proposed success and regression rules: need explicit approval before preregistration

- Primary improvement: Core-4 on exported Q8_0 versus matched original Q8_0, paired 95% bootstrap CI for the difference excluding zero on the positive side; target at least +1 percentage point as a practical improvement.
- Per-core-task non-inferiority margin: 1 percentage point. Coding margin: 2 points. A point estimate inside the margin with an overly broad CI is **inconclusive**, not proven preservation. Small samples may not resolve these tight margins; collect an approved larger evaluation rather than relaxing them after results.
- Median/mean generated tokens no more than +10%; decode throughput no worse than -5%, using repeat measurements and fixed context/batch/hardware conditions. Report prefill and total wall time too; lower token counts can make a model more useful even if tokens/sec is unchanged.
- Optional HF-only perplexity guard: no more than +2% relative drift, on a separately pinned corpus; cannot be measured from the current chat interface.
- Existing eight circuit prompts remain integration diagnostics, not a statistically credible release threshold. Require new generated-format/semantics tests on disjoint seeds for a dedicated circuit capability claim.

### Tiers

| Tier | Data and scope | Use |
| --- | --- | --- |
| Development | Disjoint validation drawn from training-source splits, newly authored verifiable instruction tasks, unseen synthetic circuits; no Core-4 final items | Hyperparameters, sampling and checkpoint selection |
| Plumbing smoke | Tiny explicitly identified public-task subset after approval | Transport/grader correctness, not optimization or release scoring |
| Final D / Core-4 | Full MATH-500 500, GSM8K test 1319, IFEval 541, frozen MMLU-Pro subset 1000; verify counts from pinned data | Mandatory cached-Q8 baseline and matched release comparisons |
| Extended D | AMC23, research Minerva, OlympiadBench, gated GPQA Diamond; HumanEval+/MBPP+ using a verified EvalPlus integration; general coverage such as ARC-Challenge, HellaSwag, WinoGrande with backend-appropriate tasks | Breadth, coding and regression evidence |
| Reproduction R | Original eight research tasks with published repeat counts and 32k budget | Separate comparison to published WSP results |

Do not tune on the first 100 MATH-500 questions or other final-test subsets. Final runs are locked and run only on selected finalists. Baseline final-test results may be recorded, but test examples must not enter the training/failure-analysis feedback loop used to optimize candidates. Use separate development data for that purpose.

For coding, run generated programs only in an isolated runner with no secrets, network or host write access, with time/memory limits. The local terminal has full host rights and is **not** an acceptable code-evaluation sandbox. Prepare runner configuration without executing generated code until approved. Use EvalPlus for the plus tests; standard HumanEval/MBPP tasks are not automatically HumanEval+/MBPP+.

Artificial Analysis methodology informs reporting, but our suite is not its Intelligence Index or an official leaderboard evaluation. Neither adopting a public harness nor using familiar datasets grants that status.

### Every evaluation manifest must pin

Model/artifact IDs, SHA-256 and revisions; tokenizer and template hashes; harness commit; task configuration and grader versions; dataset revisions/config/split/item IDs; few-shot examples and order; rendered prompts; reasoning mode; all effective samplers; input/context and generation caps; seed/sample mapping; stop/EOS rules; backend/build/driver; precision/quantization; retry policy; cache key; command; timestamps; resource samples; code revision plus hashes of dirty source files.

Record raw responses, extraction results, grader failures, truncations, refusals, generated-token counts and per-item timing. Transport errors are not wrong model answers: retain them, retry only under a predeclared policy, and report unresolved items as incomplete evaluation. Never regenerate a wrong answer until it passes. Resume only with matching manifest hashes and never let cache reuse hide a model/protocol change.

## 6. Track A: repair and controlled fine-tuning

### A0: baseline and source inventory

Complete cached Q8_0 baseline first. Later obtain the pinned trainable source only after approval and an hf-mem/artifact-size check. Confirm installed Unsloth/Transformers capability without modifying Studio. Keep the existing Python 3.14 Stim environment intact; choose a separate approved training environment based on actual wheel/kernel compatibility.

### A1: reconstruct the published repair on CPU

Primary references: WSP blog, source repository and model card (sources S1-S3). Review `apply_edit.py`, manifests, file formats and licenses before running any reconstruction; reading a source is not approval to execute its code.

Published p50c edit targets 64 MLP `down_proj` columns in each of layers 15 and 19: 131,072 addressed entries, with 130,188 distinct from the originals after BF16 rounding. It suppresses a direction associated with repetitive thinking. These are third-party findings; reproduce them, do not claim invention.

Expected raw BF16 tensor hashes from the released card:

```text
model.language_model.layers.15.mlp.down_proj.weight
82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53
model.language_model.layers.19.mlp.down_proj.weight
8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2
```

Verify base hashes, output hashes, names/shapes/dtypes, changed-entry counts, and that every other tensor remains identical. Exact diffs, not an approximate guessed rank-one edit, are the bit-exact reference. Save a new reconstruction manifest and complete output hashes. Stop after three failed repairs of the same check.

Published gains are strongly associated with reduced looping/truncation in long thinking-mode generations. They do not establish the same gains in non-thinking Q8_0 deployment. Measure productive completion, repetitive n-grams, truncation, correctness and formatting on disjoint development data before choosing training targets.

### A2/A3: train and compare

Minimum arms:

- A-BASE: original, no repair, no LoRA.
- A-REPAIR: exact reproduced repair only.
- A-SFT: LoRA on original.
- A-REPAIR-SFT: the same LoRA procedure on repaired original.

Fix training-example order, token budget, optimizer, seed and checkpoint selection for matched arms. Repeat finalists across seeds when affordable. Apply repair **before** training in the combined arm. Post-training reapplication is a different experiment and can erase learned changes.

Initial bounded recommendation: text MLP projections `gate_proj`, `up_proj`, `down_proj` only; rank 16, alpha 16, dropout 0, bias none; freeze original weights, embeddings, vision modules and other projections. With hidden 1024/intermediate 3584 across 24 layers, this is **5,308,416 LoRA parameters**, conditional on actual module inventory. This replaces the earlier approximate all-projection count: gated query projections and GDN projections must be inventoried before any broader exact count is claimed.

The initial MLP-only target is an isolation/memory starting point, not a claim that GDN weights are unimportant. Optional A-ATTN expands targets to verified attention projections; optional A-GDN includes verified GDN input/output projections. Both are separate ablations. Do not assume a conventional `q_proj` target matches GDN modules.

| Setting | Initial proposal |
| --- | --- |
| Base precision | BF16/16-bit LoRA; no 4-bit QLoRA |
| Sequence | 2048 ceiling, shorter pilot first; no silent answer truncation |
| Batch | Microbatch 1, accumulation 8; record actual nonpadding supervised tokens |
| Optimizer | AdamW, preferably FP32 states for this small adapter; approved supported 8-bit variant is a separately recorded option |
| LR/schedule | 1e-4, cosine, 20 warmup steps, gradient clip 1.0 |
| Budget | 300 optimizer steps maximum or one pass through the approved capped token corpus, whichever comes first |
| Checkpointing | Supported gradient checkpointing; save every 50 steps and on cooperative stop; dev check every 50 |
| Randomness | One fixed pilot seed; up to three seeds for finalists under separate run budgets |

Unsloth's current guide reports BF16 LoRA support for Qwen3.5, a roughly 3 GB 0.8B planning figure, Transformers v5, and advises against 4-bit QLoRA for dense and MoE Qwen3.5. These are documentation claims, not a measured local fit. Custom kernels and Windows support must be validated in the actual installed environment; do not update Studio or use WSL as an unapproved workaround.

### Training loss and data

Use assistant-token-masked SFT cross-entropy, preserving the pinned chat template and EOS behavior. Keep meaningful reasoning examples and direct-answer examples; reasoning-token conventions must match Qwen rather than inventing special tokens. Unsloth's suggested minimum 75% reasoning examples is a starting hypothesis for preserving reasoning, not a guarantee. Development metrics in both thinking and non-thinking modes decide whether an approved mixture needs changing; never final-test feedback.

Training candidates, not approved downloads: GSM8K train; a legally available stable MATH-training source; an audited permissively licensed general instruction mix; our own synthetic circuit-ISA tasks. A proposed initial mixture by supervised tokens is 50% math/reasoning, 40% general instructions, 10% circuits; match it across A and B. First pilot cap: approximately one million supervised tokens, adjusted and frozen before running to fit the step budget. Prefer licensed concise verified solutions over unbounded traces that cannot fit sequence length.

GSM8K train is separate from test. MATH-500 is drawn from MATH test, but split names do not prove complete absence of overlap; audit all chosen sources. Do not assert a mirror inherits a license without checking its actual distribution terms.

## 7. Track B: bounded differentiable reference architecture

### Relationship and priority

Track B augments learned computation. It is not the old compiled-tableau-in-weights experiment and is not a model calling an external simulator.

Prioritize baseline and Track A first; develop the Track B CPU reference in parallel. Retain compiled-tableau stage 6 as a separate deferred architectural line. Its exact ISA execution cannot be inferred from Track B's observables. The classical engine provides correctness references; it is not itself a differentiable adapter.

### Source-backed limitation

Maronese et al. (S4) study Clifford-enhanced MPS, `|CMPS> = U_C |MPS>`, using Pauli conjugation to compute observables classically. They compare ensembles that vary the Clifford. They explicitly note that one fixed Clifford preserves the MPS fidelity-distribution expressibility statistics. High expressibility guarantees neither trainability nor better ML performance. Their results do not demonstrate improved LLM benchmarks.

A fixed Clifford is algebraically equivalent to changing the observable dictionary. That is an exact identity, not a novel computational primitive. The useful hypothesis is whether the resulting **structured nonlocal feature dictionary**, under a low-bond representation and fixed resource budget, improves learning over local or unstructured readouts.

### Corrected v1 specification: width 8, open boundary, real tensors

Earlier discussion considered width 16. Start at **8** to respect the current engine reference boundary. Wider adapters require a separate proposal and must not silently extend the engine's supported width. Use open boundaries; the paper's periodic complex MPS parameter counts are not our architecture's counts.

For each token residual `h` of shape `[batch, sequence, 1024]`:

1. Take a non-mutating normalized copy `x = h / sqrt(mean(h^2) + 1e-6)` in FP32. No trainable normalization scale in v1.
2. Input projection without bias: `theta = pi * tanh(W_in x)`, `W_in` shape `[8,1024]`.
3. Encode the input product state at site i as `v_i = [cos(theta_i/2), sin(theta_i/2)]`. No dense 256-amplitude state is materialized in the production contraction.
4. Train a real open-boundary matrix-product operator: core `A_i[a,s,t,b]`, shape `[d_(i-1),2,2,d_i]`, with bonds `[1,4,4,4,4,4,4,4,1]`. Contract input physical index t against v_i to form input-dependent MPS core `B_i[a,s,b] = sum_t A_i[a,s,t,b] v_i[t]`.
5. Compute norm `Z = <psi|psi>` and normalized Pauli expectations `f_j = <psi|P'_j|psi>/Z`. Use `max(Z,1e-12)` only as a defensive guard; any clamp hit is recorded and investigated, not silently accepted. Normalize transfer environments by positive scalars with numerator/denominator scaling kept consistent. Avoid differentiating SVD/canonicalization in v1.
6. Fixed observable dictionary of M=32 distinct strings: 8 single-site X, 8 single-site Z, 7 adjacent ZZ, 7 adjacent XX, and two explicitly pinned longer Z products (sites 0,2,4 and 1,3,5). Freeze the order and masks in a manifest.
7. Choose one fixed real Clifford from H and CNOT (other real Clifford gates require an explicit convention). Compile `P'_j = U_C^dagger P_j U_C`, signs included, offline. Distinct strings remain distinct under conjugation; validate Hermiticity and real expectation values.
8. Output without bias: `delta = W_out f`, shape `[1024,32]`, W_out initialized to zero; add delta to h in the base dtype.

Initialize MPO cores as an identity/product path plus small seeded noise to expose bond paths, rather than an all-zero state. Validate nondegenerate norms and feature variance. Zero W_out preserves the base at initialization. On the first backward pass, upstream adapter gradients are zero; W_out updates first, then input/core gradients should become nonzero. Never zero-initialize both the feature network and its output.

Real Clifford gates can produce Pauli strings with an even number of Y factors. Handle their real matrix elements with exact phase bookkeeping (for example real `J=[[0,-1],[1,0]]`, `Y=iJ`, and the resulting real sign), not by dropping Y or phases. A complex128 dense reference may be used in small CPU tests; the production operator path is real FP32.

### Integration and parameter counts

Start with one wrapper after the last decoder block, before final normalization, in the frozen text model. This is an additional token-wise nonlinear residual feature map. It does not add a cross-token state or replace attention/GDN. Preserve cache/output signatures and compare token-by-token decoding to full-sequence forward. A later insertion after layer 19 or multiple insertions is a separate budgeted experiment.

Conditional on hidden 1024, width 8, bond 4, M=32, no biases/scales:

- Input projection: 8,192 parameters.
- MPO: `4 * (2*4 + 6*4^2) = 416` parameters.
- Output projection: 32,768 parameters.
- **Total per insertion: 41,376 trainable parameters.**

Conventional bias-free bottleneck adapter `1024 -> 20 -> 1024`, GELU, has 40,960 parameters (about 1% fewer). Report the difference; do not add unused parameters merely to match a headline count. Total checkpoint parameters are measured from the actual loaded/exported model, accounting for tied tensors and retained vision components.

For width n, bond chi and M Pauli strings, observable contraction is O(M*n*chi^3) plus encoding/projections. Base reference cost at n=8,chi=4,M=32 is 16,384 contraction-scale units per token, **not an exact FLOP count**. Include projection costs, backward activation storage, batching, chunking and backend overhead in measured profiles. Long Pauli strings do not change this polynomial contraction order but can affect gradient signal. Record forward/backward peak memory; no claim that training memory is only tensor-parameter storage.

### Differentiability and correctness gates

- Autograd through input projection, sin/cos, MPO contractions, normalization and output; fixed integer conjugation is outside the graph.
- Dense-statevector FP64/complex128 CPU reference for widths 1-8; norm, signs and expectations versus contractions; FP64 finite-difference/gradcheck on nonzero-output test initialization.
- FP32 output error target <=1e-5 absolute versus reference on conditioned normalized cases; gradient checks use declared tolerances and tiny inputs. Investigate ill-conditioned cases rather than widening tolerance automatically.
- A new conjugation helper is required: current `Machine.run` applies state evolution and is **not** a public arbitrary-Pauli conjugation API. Reuse validated conventions and cross-check with Stim; do not claim the API already exists.
- Base equivalence with W_out=0, gradient emergence after its update, bounded/finite expectations, no norm collapse, full-sequence versus cached decode parity, save/load round trip.

CPU reference first. After authorization, freeze the best Track A checkpoint and train adapter-only with identical data/token budgets across controls. Pilot: FP32 adapter, BF16 frozen base, AdamW FP32 states, lr 3e-4, clip 1, microbatch 1, accumulation 8, sequence 1024, 200 steps maximum, save/dev check every 25 steps. Reduce sequence for an approved memory pilot; record changes as a new manifest.

### Required controls and attribution

- B0: best Track A, frozen, no adapter.
- B1: conventional GELU bottleneck adapter (~matched parameters).
- B2: MPO adapter, local/short dictionary, U_C identity.
- B3: same MPO architecture, fixed Clifford-conjugated dictionary.
- B4: same MPO with a frozen random nonlocal Pauli dictionary, matched Pauli-weight/Y-parity profile as closely as possible.
- B5 equivalence test: B3 without runtime Clifford, directly using the compiled P'_j dictionary. It must reproduce B3; it tests our implementation, not a performance advantage.

Match data order, supervised tokens, optimizer/steps, seeds, insertion, checkpoint selection and inference conditions. Report differences in parameter counts, FLOPs, wall time, initialization and feature statistics. Also compare a compute-matched conventional adapter if parameter matching leaves a large compute mismatch.

With the full unrestricted Pauli dictionary, a fixed Clifford is a signed permutation and can be absorbed into W_out. With a restricted dictionary it changes which correlations are exposed; that cannot generally be synthesized by merely linearly mixing the original local expectations. It can still be absorbed into a different fixed feature dictionary. A B3 win over B2 is evidence about inductive bias/readout selection, not uniquely quantum computation or a general expressibility increase. Compare B4 before attributing a win specifically to Clifford structure.

Discrete Clifford search is a later experiment with its own search compute budget and matched control tuning. Do not use final benchmark feedback to choose circuits. If B1 wins, keep it as a useful deployable alternative; if B3 wins, pursue validated custom deployment or separate distillation.

## 8. Data governance and licensing

Use a manifest per dataset/source: revision and hashes, original creator/license/terms, permitted use, attribution, redistribution limits, source split, filtering/dedupe, transformation code, generated-answer provenance and derivative distribution plan. Dataset licenses, model licenses and harness-code licenses are different layers.

Our original code can remain MIT where compatible. Retain MIT and Apache notices and any required attribution. The WSP materials distinguish code and other content licenses; audit the exact files before copying or distributing. We cannot relicense upstream data or waive gated-access terms by changing this repository's LICENSE.

GPQA access is gated; its card requests no public question disclosure. Chris handles acceptance/sign-in. Keep restricted questions and generations that reproduce them out of GitHub, public logs and model-card examples. Publish permitted scores/item identifiers or hashes instead. Do not route restricted prompts through external services without permitted terms and authorization.

Split new training sources by source/problem family before augmentation, not by randomly splitting near-duplicate paraphrases. Proposed train/dev split 90/10 by source group; keep the independent public final suites entirely evaluation-only. Synthetic circuit tasks use new disjoint generator seed ranges for train/dev/final capability tests, recorded before generation; preserve original development/heldout seed bounds and already-inspected cases. Do not change historical bounds to reclassify data.

Contamination checks: normalized exact matching, answer-independent problem matching, 13-gram overlap, near-duplicate candidate review, and source IDs against all evaluation sets including extended math/coding. Maintain quarantines and removal counts. These checks reduce identifiable overlap; they do not prove the original pretrained model never saw public benchmarks. If gated data cannot be audited locally yet, do not claim a completed contamination audit.

## 9. Resources, runtime safety, checkpointing and TPU gates

### Observed local hardware (snapshot, not current availability)

Earlier read-only planning inspection returned RTX 4060 Ti 8,188 MiB total, about 7,037 MiB used; physical RAM about 47.75 GiB, free RAM about 2.50 GiB; C: about 142.84 GiB free; HF cache about 120 GB. The active model was Nemotron, not Qwen, and training status was idle. **These are historical snapshots.** Low available RAM and an occupied GPU mean capacity is not currently reserved. Recheck before resource decisions; never stop processes to make space without approval.

Use CPU for repair reconstruction and small tensor references. Use local Unsloth GPU for documented Track A training after a scheduled handover. Track B starts on CPU, then approved local GPU, with Kaggle optional. No package upgrades or environment creation in Studio as a side effect.

### Direct local inference exception

For this project only, the active workspace instructions permit chat completions to `http://127.0.0.1:8888/v1/chat/completions`. Before any model request:

- Read `/api/inference/status`; require the authorized Qwen artifact/variant and no load in progress. If another model is loaded, stop and ask Chris; do not unload/load or use its alias.
- Read `/api/train/status`; require `is_training_running=false`. Coordinate other inference users; training idle alone is not an exclusive GPU lease.
- Never call load/unload, start a second llama process, or use a direct GGUF Python binding.
- Send the active model ID, standard completion fields and chat_template_kwargs only, no Unsloth provider/MCP/session extension fields or X-Unsloth headers.
- Set temperature, top_p, top_k, min_p and max_tokens on every request, plus all other declared penalties/seed controls. Save request and response under `experiments/records/` without credentials.

Other clients keep their existing routes. No provider or LiteLLM edits are authorized.

### Memory pilot and checkpoint contract

After explicit training approval, reserve a named exclusive GPU window. Record PID/run ID, actual loaded tensor inventory, free RAM/VRAM and disk before load; sample per-process RSS and whole-card VRAM, torch peak allocated/reserved (synchronize/reset at defined boundaries), compile time, first optimizer step, steady-state token throughput and end-of-run peaks. Budget scratch space for source, merged checkpoints, conversion intermediates and retained optimizer states; never estimate from a GGUF alone.

A first short pilot is bounded to 10 optimizer steps at reduced sequence, followed by an explicit fit review before a full run. Suggested operating headroom: >=1 GiB VRAM and >=8 GiB available system RAM before the sustained run; these are planning preferences for Chris to accept, not automatic permission to close applications. Actual peaks decide the fit.

Each training checkpoint contains adapter weights, optimizer/scheduler, global step, RNG states, data order/cursor, source and tokenizer hashes, config, dataset manifest, runtime versions and dirty-code hashes. Save to a new step directory, verify files/hashes, then mark complete. Keep latest two, best-dev and final checkpoints; stage older ones with a manifest only after approval, never hard-delete. Durable backups go to Proton Drive, not another unapproved backup service.

Stop cooperatively for nonfinite loss/gradients/norms, data/provenance mismatch, unexpected backend fallback, exhausted approved budget, or sustained resource pressure. Save a stop record; on OOM retain evidence and ask for a smaller configuration, do not silently restart a run with altered settings. Test save/resume on the pilot, including the next batch/step. Resume bitwise where deterministic, otherwise declare numerical tolerances and discontinuities. Do not interrupt another run to resume ours.

### Kaggle/TPU

Kaggle access exists, but actual generation, chips, usable HBM, quota, session limits and image are unverified. Kaggle's public docs quote 20 hours/week and 9 hours/session while disclaiming their older hardware examples; do not treat those figures as this account's guaranteed allocation. Community reports of v5e-8 do not establish Chris's runtime.

TorchTPU has native PyTorch/autograd/distributed support and HF demonstrates Qwen3.5 inference; that is not proof of Kaggle, this exact training path, PEFT, GDN backward or our custom contractions. Unsloth on TPU is not assumed. Prefer minimal portable PyTorch reference code, then select TorchTPU, PyTorch/XLA or JAX from actual compatibility measurements, not preference.

Before substantial work, require:

1. Approved runtime discovery: device type/count/topology, memory per device, OS/Python/package versions, available quota and expiry. Chris handles account/UI information unavailable from safe runtime discovery.
2. Exact-operation forward/backward tests: adapter einsum, sin/cos, norm/scaling, GDN, attention, masking, loss, optimizer and checkpoint path.
3. FP32/BF16 output and gradient parity with CPU reference using declared tolerances.
4. Unsupported-operation and CPU-fallback report; successful output alone does not prove TPU execution.
5. Save/resume and state-dictionary weight round trip, including tied weights and sharded layouts.
6. Measured compile cost, recompiles across fixed shape buckets, synchronization-correct steady-state throughput and memory.

Replication puts a complete model and relevant training state on every replica; it does not pool HBM. Sharding partitions tensors/state, requires supported plans/collectives and extra validation. Eight chips do not automatically form one shared 128 GB space. For a small frozen base, replicated data parallel may be simpler; start on one device before distribution. Kaggle uploads/checkpoints and session creation each need approval. No cloud billing or HF Jobs without Chris naming the job in the current conversation.

## 10. Export and deployment

### Track A

Preserve the trainable adapter and exact base lineage. Merge into a new BF16 checkpoint without mutating source; convert the pinned baseline and candidates through the same pinned HF-to-GGUF and Q8_0 quantization toolchain. Unsloth documents GGUF export. A reported Qwen3.5 standalone LoRA-to-GGUF failure at value-head reordering (S10) motivates merge-and-reconvert; issue status and applicability must be checked at implementation time, not treated as a permanent universal limitation.

Do not require an identical tensor count to the cached GGUF: multimodal/text-only export choices can differ. Instead validate the expected tensors, shapes, architecture, tokenizer and precision policy for the declared export. Record actual bytes and parameters. Verify save/load and BF16 merged-vs-unmerged parity, then quantized-vs-BF16 differences on a frozen diagnostic set. Exact greedy output agreement is not a universal quantization requirement; record logit/token differences and benchmark impact rather than pretending bit equality is guaranteed.

Test the final GGUF through the real approved serving surface, not only the pre-export checkpoint. Hold context, sampler, prompt, runtime build, batch and hardware conditions fixed. Full memory measurement includes caches, recurrent state and scratch, not just model buffers.

### Track B

A custom MPO/observable residual is not automatically supported by llama.cpp, GGUF or stock Unsloth inference. Primary research artifact: versioned adapter safetensors/config plus a reviewed HF/PyTorch wrapper and reference tests. Do not claim `trust_remote_code` alone makes serving compatible or upload executable code without review.

Options after a demonstrated improvement: keep custom reference deployment; explicitly implement/test a backend extension; or distill into a conventional LoRA/plain checkpoint for GGUF. Distillation is a new experiment with its own data/license/budget and re-evaluation, not a lossless merge of the nonlinear adapter. No hidden-state sidecar is assumed available from the current chat API.

## 11. Milestones and constructive alternatives

| Milestone | Evidence required | Alternative if blocked |
| --- | --- | --- |
| P0: this plan | Saved document and handoff; no runtime changes | Resolve remaining protocol choices before execution |
| E0: evidence closeout | Authorized current-status doc updates; fresh full suite and diff check; raw hash unchanged; all generated evidence accounted for | Keep historical results clearly labeled until closeout approved |
| E1: protocol package | Harness commit/task matrix/license register; offline payload/grader tests; approved primary thresholds | Narrow to supported generation tasks, documenting missing coverage |
| E2: cached plain-Q8 baseline | Complete Core-4 with artifact hash, per-item records, CIs, timing/resources; no weight edits | Publish partial completion explicitly, not a fabricated aggregate |
| A1: source and repair | Approved source acquired; all reconstruction hashes and unchanged tensors verified | Stop after three failed repairs of a check; inspect source conventions |
| A2: matched A arms | Approved pilots then trained arms with same data/budget; development-selected finalists | Shorter sequence/rank or documented simpler target set as a new run |
| A3: deployment validation | Matched original/candidate Q8_0 exports and final evaluation meeting approved criteria | Release attributed repair plus measurements; retain a useful simpler winner |
| B1: CPU reference | Dense/parity/gradient/base-equivalence/cache/save-load tests and exact parameter counts | Improve conditioning or reduce bond/observable count under a new config |
| B2: learned comparisons | B0-B4 ablations, repeated seeds where approved, capability and resource table | Ship the conventional adapter if superior; publish bounded TN findings |
| T1: optional TPU | All six compatibility gates and account budget established | Keep TPU off the critical path; local smaller experiments |
| R1: optional full WSP reproduction | Exact published task/profile mapping, full original and repaired runs, own records | Label reduced-budget/backend-substituted study as partial reproduction |
| Release | Reviewed model cards, licenses/notices, dataset provenance, safety/limitations, final artifacts/hashes, reproducible commands | Code/results release can proceed separately if no improved checkpoint qualifies |

No metric target is guaranteed. Fallbacks aim to find a useful contribution rather than make failure the purpose of the project.

## 12. Ownership, artifact lineage and parallel conversations

Coordinator owns this plan and approves shared-file handoffs. No conversation edits the same file concurrently. Paths below are proposed new paths, not claims that files exist.

| Workstream | Owned paths | Dependencies |
| --- | --- | --- |
| E: evaluation | `experiments/eval/`, `experiments/manifests/eval-*.json`, `docs/evaluation-phase2.md`, proposed `tests/test_eval_*.py`; benchmark-protocol changes only under an explicit coordinator lock | Existing records; no training dependency |
| A: repair/SFT | `experiments/trackA/`, `experiments/manifests/EXP-A-*.json`, `docs/trackA.md`, proposed repair tests | Frozen E manifest; approved trainable source/data |
| B: TN reference | `src/qwentum/adapter/`, `tests/test_adapter*.py`, `experiments/trackB/`, B manifests, `docs/trackB.md` | CPU reference can start independently; learned B0 comparison needs selected A checkpoint |
| T: accelerator | `experiments/tpu/`, T manifests, `docs/tpu-gates.md` | Actual approved runtime; E protocol for eval work |

Existing `src/qwentum/tableau.py`, `language_eval.py`, old tests/scripts, README and AGENTS remain coordinator-owned unless explicitly assigned. The B conjugation helper lives in B-owned paths first. Do not create another repository. Separate working directories/worktrees are optional only with explicit setup approval and respect for current dirty work; do not create branches or commit automatically.

IDs: `EXP-E-001...`, `EXP-A-001...`, `EXP-B-001...`, `EXP-T-001...`. Preassign IDs centrally to avoid collisions. Each run gets an immutable manifest and dedicated record/checkpoint directories; exclusive-create raw records. A GPU reservation records owner, run ID, scope and release in a shared coordinator-owned place, with atomic acquisition; messages in separate chats alone do not establish exclusivity. Chris arbitrates. Never infer permission to evict another model from a file lock.

Lineage DAG: cached GGUF -> cached baseline record; pinned upstream -> matched baseline export; pinned upstream -> reproduced repair; original/repaired -> LoRA; selected Track A -> B controls/adapters; any distillation is a new child. Hash inputs and outputs for every edge. Never describe an experiment's parent using a mutable `latest` alias.

### Copy-paste scoped handoffs

**E -- evaluation (recommended next conversation):** use the complete handoff in section 15.

**A -- Track A:**

> Continue qwentum-computer in the existing repository. Read both AGENTS.md files, git status/diff, and docs/plan-phase2.md sections 3, 6, 8-10 and 12-14. Own only Track A paths. First prepare the source/repair manifests and reviewed CPU reconstruction plan, then matched LoRA configurations and export tests. The cached GGUF is not the trainable source. Verify the two released BF16 tensor hashes, all unchanged tensors and attribution. Do not download/install/run remote code/train/load/unload/commit/publish without separate approval. Preserve existing records and dirty work. Inventory actual module shapes before counting LoRA parameters; no invented GDN target coverage. Coordinate the E protocol and any GPU reservation with Chris.

**B -- architecture/reference:**

> Continue qwentum-computer. Read both AGENTS.md files, git status/diff, and docs/plan-phase2.md sections 7-10 and 12-14. Own only B paths. Prepare the width-8 real open-boundary MPO adapter, Pauli conjugation helper, dense reference and gradient/cache/save-load test plan. No extra simulator width is authorized. Fixed Clifford equals a compiled observable dictionary; compare local and random-nonlocal readouts and conventional adapter controls. Current Machine.run is not an arbitrary conjugation API. CPU-only work first; package installs and any model/GPU operations need approval. Do not change the running model or historical records.

**T -- Kaggle:**

> Continue qwentum-computer. Read both AGENTS.md files, git status/diff, and docs/plan-phase2.md sections 9 and 12-14. Own only T paths. Prepare device/runtime discovery and six TPU compatibility gates, including exact-operation backward tests, fallback detection, numerical parity, round-trip checkpointing and compile/steady-state timing. Do not start a session, remote job, paid service, install or upload. Do not assume Kaggle's hardware, quota, a flat memory pool, or Unsloth TPU support. Ask Chris only for account/runtime facts unavailable through approved discovery, one at a time.

## 13. Risk register and source map

Evidence classes: **local evidence** is a read record/file or measured snapshot; **third-party** is a claim in a cited source; **hypothesis** requires our experiment. No planning estimate is a measured fit.

| Risk | Evidence/source | Response |
| --- | --- | --- |
| Published loop repair may help long thinking much more than non-thinking | Third-party S1-S3 | Separate D/R protocols, own matched baselines, development failure analysis |
| Fixed Clifford expressibility or uniqueness overstated | Third-party S4 plus exact conjugation identity | B2/B3/B4 and compiled-dictionary equivalence; claim structured features, not quantum advantage |
| Weak gradients/norm instability | S4 trainability caveat; architecture hypothesis | Conditioned initialization, norm telemetry, FP32 contraction, finite-difference tests, output warm-up behavior |
| Incorrect Pauli phases/width/reference claims | Existing width-8 engine and source conventions | New tested helper; Stim oracle; real even-Y handling |
| Benchmark names mistaken for matching datasets/protocols | Actual harness YAML and WSP source S3,S6 | Pin dataset IDs, few-shot, graders and prompts; Minerva distinction |
| Repeated test-set tuning or synthetic leakage | Evaluation design risk | Independent development, locked final suites, grouped split/dedupe audits |
| Current GPU/RAM occupied | Historical local snapshot, not reservation | Fresh resource discovery and explicit scheduled handover |
| QLoRA/family kernels or Windows backend mismatch | Third-party S5 | BF16 path, isolated approved environment, memory/compatibility pilot |
| GGUF conversion/standalone LoRA mismatch | Third-party S10; local unverified upstream derivation | Match converter, merge before conversion, recheck current issue, test actual final artifact |
| Custom B layers unsupported by serving backend | No verified custom-layer support | HF reference artifact first, separate backend/distillation project gate |
| TPU/Kaggle claims exceed evidence | Third-party S7-S9; Kaggle page caveat | Actual runtime and six gates; optional, no blind remote execution |
| License or question leakage | Dataset-specific terms including S11 | Per-source license register, no relicense of upstream rights, restricted logs private |
| False official-leaderboard implication | S12 methodology distinct from our suite | Call results local public-benchmark measurements, not official Index |

### Sources consulted during planning (mutable unless a revision is recorded)

S1. https://rikiyat.github.io/blog/weight-space-programming/ -- published experiment and claimed mechanism/results.
S2. https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming -- reconstruction hashes, rerun and attribution.
S3. https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math -- README, `code/evaluate.py`, `code/profiles.py`, `code/apply_edit.py` and p50c manifests; review/pin before use.
S4. https://arxiv.org/html/2506.13605v2 -- CMPS construction, fixed-Clifford limitation and trainability caveats.
S5. https://unsloth.ai/docs/models/qwen3.5/fine-tune and https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf -- documented BF16 LoRA and export; recheck guide/code inconsistencies before implementation.
S6. https://github.com/EleutherAI/lm-evaluation-harness and https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/API_guide.md -- task implementations and API constraints. https://github.com/huggingface/lighteval -- alternative harness.
S7. https://huggingface.co/blog/hf-torchtpu -- native PyTorch TPU support and Qwen3.5 inference example, not our training validation.
S8. https://github.com/google-pytorch/torch_tpu -- backend capabilities and install/runtime requirements; pin a compatible version only after discovery.
S9. https://www.kaggle.com/docs/tpu and https://docs.cloud.google.com/tpu/docs/v5e -- platform caveats and hardware specs, not account allocation.
S10. https://github.com/ggml-org/llama.cpp/issues/21125 -- reported Qwen3.5 LoRA conversion issue; unresolved/fixed status was not conclusively established by the extracted discussion. Do not claim every version is broken.
S11. https://huggingface.co/datasets/Idavidrein/gpqa -- gated terms, attribution and question-disclosure restriction/request.
S12. https://artificialanalysis.ai/methodology/intelligence-benchmarking -- different suite/methodology; not an endorsement of our aggregate.
S13. https://huggingface.co/Qwen/Qwen3.5-0.8B -- source model architecture, model-card evaluation/sampling and loop caveat. Pin the revision before treating current settings as historical ones.

Outstanding source checks: final harness commit/dependencies/task configurations, EvalPlus exact integration and isolation, every selected dataset's revision/license, installed Unsloth training compatibility, source-model tensor inventory, and current TPU support on the actual runtime. Never turn these open items into facts by copying this plan.

## 14. Explicit approvals and remaining decisions

Already authorized: saving this document. No weights/datasets were downloaded, packages installed, inference issued, model switched, training started, remote runtime started, commits/pushes/publication performed, or MemPalace filed as part of saving it.

Before each action, obtain the relevant approval:

- Additional repository/code retrieval or vendor execution beyond reading; package installs/environment setup; dataset acquisition, tokenizer files or source weights. Model weights require hf-mem/artifact sizing first. Existing cached GGUF needs no replacement download.
- Benchmark execution with the approved manifest, budget and resource window. If Qwen is not already active, Chris authorizes/handles switching under the governing instructions; never call load/unload automatically.
- Context/configuration changes, training pilots and each named sustained training run; approval must account for other clients and actual resource use.
- Kaggle sessions, uploads, quota consumption, or paid/cloud/HF Jobs (Chris names a billed job in that conversation).
- Historical-doc closeout edits, staging unrelated files, MemPalace filing, commits, pushes, model/dataset uploads, release names/licenses and public posts.

Remaining decisions are asked one at a time at the relevant milestone: Core-4 few-shot/profile and numerical margins; legal training mix; actual expanded LoRA targets; budget for full R reproduction; custom B deployment versus distillation. Do not ask for TPU account details before the optional TPU milestone. Recommended first implementation decision: freeze the Core-4 protocol package, then request one bounded install/data/inference approval at a time.

## 15. Go-forward prompt for the next conversation

```text
Continue the existing qwentum-computer project at:
C:\Users\cld-main\Desktop\github-projects\qwentum-computer

Read the active workspace and repository AGENTS.md first, inspect git status and the current diff, then read docs/plan-phase2.md in full. Preserve all unrelated and uncommitted work. Do not create another project.

This conversation owns evaluation preparation (workstream E). Our first required result is a plain/unmodified baseline for the already-downloaded Qwen3.5-0.8B-derived Q8_0 artifact, followed later by quantization-matched candidate comparisons. Qwentum is the model-family name; Qwentum Core-4 is the evaluation suite, not a model architecture.

Existing artifact (do not download a replacement):
C:\Users\cld-main\.cache\huggingface\hub\models--unsloth--Qwen3.5-0.8B-GGUF\snapshots\6ab461498e2023f6e3c1baea90a8f0fe38ab64d0\Qwen3.5-0.8B-Q8_0.gguf
Recorded size: 811843840 bytes.
Recorded SHA-256: 0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c.
Verify the bytes before using it. Its exact derivation from pinned upstream source revision 2fc06364715b967f1860aea9cf38778875588b17 remains unverified. The GGUF is our cached inference baseline, not the trainable source.

Use EleutherAI/lm-evaluation-harness as the primary framework. Do not implement a new benchmark harness or load a second GGUF process. Plan a small guarded local-chat-completions integration with the existing Unsloth service. Generate-only tasks through chat; no loglikelihood/perplexity claims from that API. Verify actual task configs, graders, few-shot settings, dataset identities and licenses. In particular, minerva_math500 uses MATH-500; WSP's separate Minerva benchmark is math-ai/minervamath, not automatically the harness minerva_math family. Standard HumanEval/MBPP are not automatically EvalPlus.

Core-4: MATH-500, GSM8K, IFEval strict prompt-level, frozen MMLU-Pro 1000-item subset. Prepare the exact protocol and proposed numerical acceptance thresholds for approval. Use disjoint development data; do not tune on a public final-test subset. Keep WSP think35 reproduction separate from our non-thinking deployment results. No new benchmark results exist yet.

Read README.md, docs/architecture.md, docs/benchmark-protocol.md, docs/provenance.md, pyproject.toml, src/qwentum/, tests/test_language_eval.py, experiments/finish_q8_report.py, experiments/finalize_q8_evaluation.py and the relevant saved records. Preserve _arm1_raw.json unchanged (SHA-256 15a07a45cef6535a81e953363e416e09cf426989b8a3a39c7509304ca0632774). The arm-4 report is complete: correct bits, faithful report, missing final period, strict-format failure; those eight prompts are integration evidence, not general benchmark results. Do not rerun the old inference/finalization scripts or overwrite historical records.

First deliverable: a pinned harness/task compatibility matrix, protocol-manifest design, license/access register, offline mock-test plan and bounded execution budget. Proposed owned paths are experiments/eval/, experiments/manifests/eval-*.json, docs/evaluation-phase2.md and tests/test_eval_*.py. Ask before first implementation writes; README/protocol/AGENTS and old source/records remain coordinator-owned unless explicitly assigned. Do not edit Track A/B/T paths.

Read-only inspections and primary-source research are allowed. Do not download weights/datasets, install packages, run benchmark inference, execute remote code, load/unload/switch models, start training or remote sessions, change Unsloth/LiteLLM/routes, commit/push/publish, hard-delete or file MemPalace without separate approval. This handoff authorizes continuation of evaluation preparation, not all actions described in the plan.

Before any later approved inference, read /api/inference/status and /api/train/status. Require the authorized Qwen Q8_0 already active, no loading and training false; otherwise stop and ask Chris. Direct chat completions only at http://127.0.0.1:8888/v1/chat/completions under the project exception; no load/unload or second process. Explicit sampler fields and request/response records are mandatory. Recheck actual memory and coordinate other clients; earlier planning found a different model loaded, which is historical, not a current-state claim.

Keep the response focused on the next gate. Ask material unresolved questions one at a time with a recommendation, starting with the exact Core-4 protocol/acceptance criteria or the first needed preparation approval. Do not ask for unrelated TPU account details now.
```
