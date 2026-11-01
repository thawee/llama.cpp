# Implementation lessons

- Optimize for the user's complete outcome, not the latest named technique. The target is the exact selected 27B model at 128K context, around 18 GiB with up to 20 GiB if needed, at least 25 generated tokens/sec and preferably over 50. Treat ANE as one candidate mechanism; prefill speed alone does not satisfy the generation target. Report measured generation and first-token latency separately and do not claim hardware feasibility before testing.

- Memory-budget integration: parsed parameters contain sentinel-padded tensor overrides even when no override was requested. Check the terminating pattern, not whether the vector or pointer is nonempty. Exercise the actual CLI/server parser before validating optional configuration guards.
- Backend identity checks: verify registry names in the backend implementation before comparing them. Metal's registry identifier is `MTL`; display names and marketing names are not registry identifiers.
- Runtime sizing: preserve the user's explicit context target when checking a budget. The requested target is 128K with a 16 GiB estimate budget; use IQ2 if IQ3 cannot fit, and verify the actual model before reporting success.
- Decode tuning: verify which backend control a flag changes before attributing a speed difference. This checkout fixes Metal command-buffer submission count at 1; changing CPU threads does not change that count. CPU samples waiting for Metal establish a GPU execution wait, but do not distinguish bandwidth from dequantization or GPU arithmetic.
- Budget accounting: prompt caches, prompt checkpoints, and speculative rollback checkpoints have separate lifetimes. Disabling --cache-ram does not disable the latter two. Check host serialization paths as well as backend buffer breakdowns before claiming an inference estimate covers retained state.
- Speculative configuration: --spec-type appends to the default NONE entry and to earlier flags. Recognize enabled modes while ignoring NONE, reject mixed unsupported modes, and test the actual parser. Appending --spec-type none does not disable an earlier ngram-simple setting; the local launcher uses an environment switch instead.

- Native-engine feasibility: a low-acceptance DFlash experiment in llama.cpp does not establish Splash or Kapitsa performance. Compare the native engine, exact target weights, draft, occupied context, and measured memory before concluding that draft retraining is required. Separate ANE prefill gains from generated-token throughput.

## Speculative acceptance callbacks

Some speculative callers report zero acceptance without a draft. Track a pending attempt before updating rejection backoff; callbacks during cooldown must not extend it. Keep target-feature processing active during pauses so the next draft uses current conditioning.

- CPU tuning scope: when the user clarifies that a history/upstream question concerns threads and M3 CPU cores, inspect performance/efficiency core counts, active runtime thread parameters, and prior thread benchmarks before broadening into Git housekeeping or unrelated upstream changes.

- macOS CPU accounting: proc_pid_rusage CPU time fields use Mach ticks on this M3 Pro. Convert with mach_timebase_info and calibrate against process_time before reporting CPU use. Do not confuse low host CPU use with a specific GPU bottleneck or performance-core affinity.

- Benchmark reuse: when the user asks to save time by using existing baseline results, do not rerun the current configuration or schedule alternating baseline confirmations. Reuse the closest compatible saved workload/settings and label comparisons as unpaired when conditions differ.

- Short optimization passes: reuse saved baselines, use one bounded candidate screen, and expand only candidates that cross the stated decision threshold. Keep bulk model conversion/hashing outside GPU measurement windows; overlapping measurements cannot select a winner.
