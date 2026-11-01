# Adaptive DFlash2 inside llama.cpp

The local opt-in `--spec-draft-adaptive` pauses DFlash2 after eight consecutive fully rejected proposals, skips 32 subsequent opportunities, then retries. Any accepted draft token clears the rejection streak. State is separate per sequence and resets at request start. Acceptance callbacks without a drafting attempt are ignored. Other speculative implementations do not update this policy.

The existing initializer tries cheap ngrams before learned drafts, independent of CLI type order. On an adaptive pause it can continue with other implementations or ordinary target decoding. Target feature processing continues during pauses so a resumed draft has current conditioning. The draft model and its buffers remain allocated; this does not reduce the reserved memory.

This applies the idea of reacting to observed speculative usefulness. It is an independently written bounded backoff policy, not a port of Kapitsa's learned cost model or KFlash tree, Splash's fused Metal kernels, or an ANE backend. The selector and dynamic convolution formulas inspected in the native engines match the existing llama.cpp graph; no concrete math bug was established. Low draft acceptance remains unexplained.

## Repeated real-model comparison

Five alternating static/adaptive pairs used the same Release Metal build, exact DavidAU IQ2 target, Q4_K_M DFlash2 draft, Q8 target/draft KV, one slot, 131072 configured context, batch/ubatch64, threads5, and the same ngram N3/M3 implementation. Each fresh server ran an eight-token warmup and a fixed 128-token greedy coding answer to a 50-token prompt. The only option difference was `--spec-draft-adaptive`.

| Measurement | Static DFlash2 + ngram | Adaptive DFlash2 + ngram |
| --- | ---: | ---: |
| Median generation, tokens/sec | 6.307 | 8.054 |
| Generation range, tokens/sec | 6.243-6.363 | 8.046-8.231 |
| Median request wall time, seconds | 21.058 | 16.685 |
| Proposed tokens per measured answer | 539 | 149 |
| Skipped DFlash2 opportunities, including warmup | 0 | 69 |

Generation improved 27.7% within this DFlash2 configuration. All five paired answers were identical and passed the coding AST checks. All ten test servers exited 0; request errors and watchdog events were zero. Peak sampled RSS was 18117.84 MiB (18.998 decimal GB). The watchdog terminates the experimental server at 20 decimal GB RSS or memory pressure level 4. Sampling every three seconds cannot establish an OS hard cap.

This is less wasted speculative work, not faster individual target kernels. It remains below the previous ngram-only result of 8.65 tokens/sec and below the user's 25/50 tokens/sec goal. The launcher therefore keeps ngram-only settings. Configured 128K capacity is not a full-128K occupied-context speed, quality, or memory test.

Raw measurements: `tasks/adaptive-dflash-results-confirmation.json`. The first screen is separate in `tasks/adaptive-dflash-results.json`. The repeated pairs preceded the defensive callback check. The final-build screen in `tasks/adaptive-dflash-final-results.json` confirmed 6.335 versus 8.073 tokens/sec, identical answers, and 69 adaptive skips. The Release build and both focused CTest tests passed after that check.

## Installed runtime verification

The final server and companion libraries were installed in `build-m3-local/bin`, with previous binaries retained in the backup directory recorded in `tasks/adaptive-installed-runtime.json`. The local server was restarted on port 8080 with the unchanged ngram-only launcher. Health, `/v1/models`, configured context 131072, one slot, binary hashes, and a chat arithmetic request passed. Five fresh 128-token coding requests matched the final screen output and passed AST validation: median 8.936 tokens/sec, range 8.890-8.952, peak sampled RSS 17.085 decimal GB, no watchdog events. This confirms ngram-only remains faster than adaptive DFlash2 on this workload. No full-128K occupied-context claim follows from these short requests.

## Reproduce

Stop the ordinary local server first. The harness refuses occupied ports 8080 and 18092; it does not manage that server's lifecycle.

```sh
python3 tasks/dflash-check.py --adaptive-comparison --confirm --binary /tmp/llama-memory-budget-build/bin/llama-server --result tasks/adaptive-dflash-results.json
```

These experiments disable `--memory-budget` after the earlier combined-model preflight and use the RSS watchdog. The production budget path still rejects learned drafts. Do not add this model-draft combination to the budgeted launcher: the new flag does not make it pass that accounting or prove the user's full-context memory target.

## Validation and remaining work

The focused unit test first failed without the policy. An additional test reproduced zero-acceptance callbacks keeping a cooldown active indefinitely; the policy now tracks pending attempts. Unit tests cover rejection boundaries, success reset, retries, stale callbacks, independent sequence state, and request reset. CLI parser tests cover the default-off setting and actual flag parsing.

Remaining engine work is candidate-tree verification and specialized Metal execution. Their published speeds cannot be inferred from this backoff result. Those paths require separate correctness and memory validation, especially for the target's recurrent state and exact IQ2 weights.
