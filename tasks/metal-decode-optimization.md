# Exact IQ2 Metal decode experiment

The tested IQ2_S integer-domain sign change regresses full-model generation by 3.23% and is rejected. Keep the original kernel and installed launcher. GPU profiling identifies IQ2-containing projection groups as the largest measured cost, but does not establish whether bandwidth or dequantization arithmetic dominates within them. No profiler code is installed in production.

## GPU profiling

The exact local Qwen 27B IQ2_M target ran with 131072 configured context, Q8_0 K/V, five CPU threads and full GPU offload. The prompt occupied 50 tokens. These measurements do not establish performance or quality with 128K occupied tokens.

The local Instruments Metal System Trace recording ended with a corrupt/incomplete macOS log archive error. Some GPU intervals could be exported, but that trace was not used for kernel attribution. A standalone hardware probe established that this M3 Pro supports encoder-stage timestamp sampling and does not support per-dispatch timestamp sampling.

A temporary Metal encoder wrapper attached two timestamp samples per encoder and reported pipeline dispatch counts after command-buffer completion. CPU/GPU timestamp calibration follows the SDK's nanosecond CPU timestamp semantics. Real GPU smoke tests agreed with command-buffer GPU durations within about 42 ns, checked profiling off/on at runtime, serial/concurrent encoders and the bounded encoder count. The instrumentation source was restored exactly after building isolated artifacts.

Four normal-graph greedy decode steps measured 121.59 ms of encoder GPU time per step. A separate two-step diagnostic synchronized the graph at each matrix multiplication output using llama.cpp's existing evaluation callback. Its encoder intervals can include operations preceding the named projection, and splitting changes fusion and scheduling. Treat the following as diagnostic groups, not an undisturbed per-kernel percentage breakdown.

| Diagnostic group | GPU ms per decode step | Share of split intervals |
| --- | ---: | ---: |
| IQ2_S projections and preceding work | 86.96 | 69.62% |
| BF16 output projection and preceding work | 19.23 | 15.40% |
| Q4_K projections and preceding work | 11.84 | 9.48% |
| IQ3_S projections and preceding work | 5.49 | 4.40% |
| Remaining groups | 1.38 | 1.10% |

The split total was 124.90 ms per step. All 1130 encoder timestamp records were valid; greedy tokens agreed between normal and split runs on their shared steps. Profiling runs are excluded from throughput comparisons. Peak sampled RSS was 16.60 decimal GB, physical footprint 4.97 GB, with no watchdog events. Both measurements are needed because mapped weights and private Metal allocations have different accounting.

## Candidate and validation

The candidate changes only the IQ2_S scalar grid/sign dot product in `ggml/src/ggml-metal/kernels/mul_mv.metal`. It converts a grid magnitude to an integer, applies its sign there, then converts the signed result to float. The hypothesis is that this removes a floating sign multiplication/conversion. It preserves weights, matrix geometry, row tiling and accumulation order. Compiler lowering and memory traffic can make this slower despite fewer source-level floating operations.

Baseline and candidate Release builds use the same configuration. Each passed 37 existing IQ2_S MUL_MAT numerical cases against the CPU reference, including single-token decoding at k=4096. A bounded single-pass microcheck at m=4096, k=14336, n=1 measured 226.33 us for baseline and 243.27 us for the candidate. That one shape/run is diagnostic only; it is not the target's complete workload.

The full-model harness alternates five fresh baseline servers and five fresh candidate servers, using 128-token greedy coding completions with the same prompt, ngram policy, context, cache types, threads, warmup and Release settings. It checks four generated Python functions on arithmetic inputs, token count, slot context, loaded libraries and sampled process-group memory. Profilers are disabled. The installed local launcher is restarted in a finally block. Active user applications and thermal variation remain possible sources of noise.

## End-to-end result and disposition

| Variant | Median generated tok/s | Range | Runs |
| --- | ---: | ---: | ---: |
| Original kernel | 9.3460 | 9.3303-9.3528 | 5 |
| Integer-domain sign candidate | 9.0442 | 9.0260-9.0489 | 5 |

The candidate is 3.23% slower at the median; every candidate trial is below every baseline trial. All ten replies contain identical output, pass arithmetic checks and generate exactly 128 tokens. Baseline peak sampled RSS is 17.091 GB and candidate peak is 17.091 GB; peak physical footprint is 5.424 GB across both. No memory watchdog events occurred. The two source lines changed in the experiment are restored; the candidate is retained only as an unapplied patch and isolated scratch artifacts.

The 9.3460 tok/s baseline is from this session's workload and conditions. Its difference from the earlier 8.936 tok/s record is not a code speedup claim. The meaningful comparison is the alternating pair in this table.

The installed launcher was restored. Health, exact-model discovery, 131072 slot context and a chat answer of 42 passed (`tasks/metal-iq2-restored-check.json`). `git diff --check` passes and there is no working-tree change under the Metal backend. The wider user's branch retains its prior unrelated edits.

The next code hypothesis should require stronger evidence about memory traffic versus instruction cost. Repeating sign or row-tile tweaks has no demonstrated path to the requested 25/50 tok/s. The BF16 head is a secondary measured group, but eliminating its measured time entirely would still leave substantial IQ2 work; that hypothetical is not an attainable optimization claim.

## Evidence and reproduction

- `tasks/metal-decode-profile-summary.json` and `tasks/metal-decode-profile-run.json`: GPU intervals, token checks, commands, process exits, memory samples and restoration.
- `tasks/metal-iq2-signed-grid-results.json`: alternating end-to-end results and final restoration.
- `tasks/metal-decode-experiment/build-check.json`: build commands, artifact hashes, numerical checks and microchecks.
- `tasks/metal-decode-experiment/timestamp-instrumentation.patch`: temporary profiler patch; not applied to production.
- `tasks/metal-decode-experiment/signed-grid-candidate.patch`: isolated kernel candidate.
- `tasks/metal-decode-experiment/profile-decode.cpp`, `run-profile.py`, `summarize-profile.py`, and `compare.py`: local experiment tools. Paths identify this workspace and scratch artifact directories; rebuild artifacts before rerunning after scratch cleanup.

No commits, pushes or submissions were made. Existing unrelated changes are preserved.
