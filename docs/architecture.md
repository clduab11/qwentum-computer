# Architecture

Status: design for a system that does not exist in this tree yet. Milestone 0 has no tableau, no compiled weights, and no Qwen checkpoint.

## Destination

An 8-qubit classical stabilizer engine embedded in a checkpoint built from `Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17`. Qubits are numbered 0 through 7. Supported operations are `H`, `S`, `X`, `Y`, `Z`, `CNOT`, computational-basis measurement, a measurement record, and single-bit classical Pauli corrections. `T`, Toffoli, and arbitrary rotations are rejected. `SWAP` stays unsupported until a later milestone adds it. RV64 is deferred.

The state is an Aaronson–Gottesman destabilizer tableau: 16 generators, each with 16 X bits, 16 Z bits, and a phase in `{0,1,2,3}`. That is 544 bits, plus at most 256 measurement bits. Generator order is not a result. Tests compare deterministic bits, the allowed set of random outcomes, and Pauli expectations in `{+1,-1,0}`.

The embedded form, when it is built, puts those bits, a program counter, and the measurement record in Gated DeltaNet rows. Fixed weights implement one record per supported instruction. Learned Qwen weights stay frozen. Circuit weights are compiled, not trained. Sixteen generator rows are the lanes of a row update. Measurement and a one-bit correction are a predicate on the record, not eight quantum cores.

## What the other machines contribute

These projects were read on 2026-11-07. None was run here, and none uses the 0.8B base.

- Qwenx86 puts a translated x86-64 integer path on a forked 32-bit RISC-V circuit inside Qwen3.5-4B. Its card is the reason to keep an external lockstep oracle and to treat a raw ISA decoder as a separate, heavier machine. Its compact loader is described at about 9 GB.
- Qwenputer's NPU and QwenPU show one compiled vector or dot-product instruction replacing a long software loop. Reported trace costs drop from 2,430,000 tokens per output token to 53,000. A C simulator on an in-weight CPU is the expensive shape.
- qwenGPU splits the host from the weights: the host lowers PTX and reads results, and the model writes every value-producing step. Its README also says a 2B model writing arbitrary kernels was unreliable, so this project uses a closed instruction list. Forty-five GPU layers and a request for about 16 GB are the caution against a general machine.

The host in an embedded run may tokenize, apply the chat template, lower a parsed circuit into instruction records, log seeds, and read the result. It does not apply `H` or `CNOT` during that run. A callback that applies a gate is the external-tool arm. Stim remains an independent oracle and is not linked into the weights.

## Evidence order

1. Stim 1.16.0 sampler smoke test. Implemented in Milestone 0.
2. Integer tableau versus Stim, including Bell on qubits 0 and 1, idle qubits 2 through 7, and rejection of qubit 8.
3. 8-qubit GHZ and teleportation. Teleportation is scored by destination measurements that are deterministic after feed-forward, not by matching intermediate shots.
4. A frozen held-out circuit list and a complete engine log.
5. One bf16 row gadget, checked on every legal input against the integer function.
6. External tool call from the base model.
7. Merged tableau layers, then a free-running Bell trace against the integer tableau.

bf16 exactness is a test, not an assumption. A dtype change reruns the integer function on the gadget's whole domain.

## Resources

The fetched 0.8B index is about 1.747 GB of parameter bytes and 1.760 GB of Hub storage. Runtime activations are unmeasured. The upstream CHIP-8 merge figure of 2.27B parameters is about 4.54 GB of BF16 parameter bytes if that count is exact, and it was not weighed here. Local 8 GB is a later measurement target. Kaggle is not a target for stages 0 through 4. Eight TPU chips are not one memory pool. No training is proposed, so gradients and optimizer state are omitted.
