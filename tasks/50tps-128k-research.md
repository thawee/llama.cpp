# Research a path to 50 tokens/sec at 128K

Research date: 2026-11-01. Keep the exact DavidAU 27B IQ2 GGUF, M3 Pro with 36 GB, 131072 total context tokens and at most 20 decimal GB process memory. Interpret the user's "128kb" as the established 128K-token requirement. No inference code or running-server configuration changed during this investigation.

## Recommendation

Use native Splash as the reference experiment for the exact IQ2 target, then bring proven execution changes into llama.cpp. It supports the target's tensor formats without Kapitsa's dense-weight requantization. The first question is whether native drafting accepts materially more tokens on these exact weights. If it does, compare target hidden features, candidate scores and accepted prefixes between engines before porting kernels. If it does not, investigate an exact-target-conditioned draft. Do not port a tree to compensate for an unexplained near-zero acceptance rate.

Fifty tokens/sec with a nearly full 128K context remains an unproven research target. Published native results use other weights, hardware or occupied prompt lengths. No source checked establishes all the user's constraints together.

## Evidence and limits

The installed llama.cpp result is 8.936 tokens/sec, median of five 128-token greedy coding replies, range 8.890-8.952, sampled peak RSS 17.085 decimal GB. Those replies used a short prompt and configured 128K capacity. They do not establish long-context generation or quality. Earlier pure-DFlash screens accepted 16 of 756 proposed tokens with the Q4 draft and 9 of 798 with Q8, on the recorded short coding workload. This is evidence about those screens, not every domain or native runtime. Raw records are `dflash-results.json` and `dflash-q8-results.json`. The prior CPU profile shows waits for Metal completion; it does not distinguish bandwidth, dequantization and GPU arithmetic. Evidence is in `adaptive-installed-runtime.json` and `decode-profile-notes.md`.

Splash documents direct GGUF support for IQ2_S, IQ3_S, Q4_K and BF16, which cover the target's principal stored types. Its family detection uses configuration and shapes, rather than the repository name. The actual local file passed Splash's Python header and tensor-type validation for all 851 tensors, reading only 10,994,252 header bytes. The generated configuration is saved in `splash-exact-iq2-header-check.json`. This supports a native compatibility experiment; it does not prove native model-check, weight loading or effective speculation. Source: https://github.com/incoai/splash/blob/main/docs/models.md . Local inspection at Splash 031a25a, docs/models.md lines 137-218.

Kapitsa imports dense IQ2 into 3-bit group-64 and dense IQ3 into 4-bit group-32. Only certain routed MoE experts retain their IQ blocks. This target is dense, so its import changes quantized values and can grow memory. It cannot serve as an unchanged-IQ2 comparison. Source: https://github.com/nace-ai/kapitsa/blob/main/docs/usage.md#gguf-files . Local inspection at Kapitsa f979136, src/gguf/import.cpp lines 208-235 and 711-735.

Kapitsa's KFlash explicitly handles Qwen's recurrent Gated DeltaNet branches by restoring parent state and replaying ancestor updates. Attention masks, depth positions, convolution state and accepted-path commits must also agree. It is feasible engineering for this architecture; the earlier explorer's suggestion that recurrent support was undocumented was contradicted by the current primary documentation. Its reported 27B improvement over the preceding native chain engine is 6% across three greedy prompts on M4 Pro. That tree increment is not evidence of a fivefold gain in llama.cpp. Source: https://github.com/nace-ai/kapitsa/blob/main/docs/kflash.md .

Splash publishes 74 tokens/sec on M5 Pro and 92 on M3 Max for standard Qwen 27B UD-Q4_K_M. That comparison disables ANE. It also publishes 43.5 tokens/sec for a different 27B IQ3 file on a 24 GB M6, with an advertised context below 128K. The context is capacity, not the occupied prompt used for speed. These results motivate a native reference experiment, not an M3 Pro throughput prediction. Source: https://github.com/incoai/splash/blob/main/docs/performance.md .

The incoai and z-lab Qwen3.8-27B-DFlash2 repositories currently expose the same model.safetensors SHA256, `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`, and config blob `79279cc5665bced6f3cdaa2095a2ffe819497b2e`. Verified through both Hugging Face model APIs with `?blobs=true`. Their repository revision hashes differ. A publisher name difference does not establish a different draft. This does not verify the local GGUF conversion against that source. Local metadata is in `dflash-draft-inventory.json`.

A known llama.cpp layer-input row-order fix addresses concurrent batches. It reports mean accepted length 3.35 to 3.92 at concurrency 16 on standard Q4 targets. Our single-slot near-zero acceptance cannot be assigned to that bug without reproduction. Source: https://github.com/ggml-org/llama.cpp/pull/29019 .

## What the hardware arithmetic implies

Apple specifies 150 GB/sec memory bandwidth for M3 Pro. Source: https://support.apple.com/en-us/117737 . The target inventory has 11.673 GB of file data, including a 0.546 GB embedding table. Excluding that table gives a rough 11.127 GB dense scan proxy. This includes small header and other costs, so it is an approximation, not a measured memory-traffic counter.

At full advertised bandwidth that proxy takes about 74 ms. The configured 128K Q8_0 KV allocation is 4352 MiB, or 4.563 GB. A simplified single read of that occupied cache adds about 30 ms. The resulting 105 ms is an optimistic traffic model, not a profiled latency bound. Cache reuse, kernel layout, partial residency and additional traffic affect the result. It explains why a one-token-per-scan design is an implausible route to 50 tokens/sec.

For speculative decoding, output throughput is mean emitted tokens per cycle divided by draft-plus-verification cycle time. Fifty tokens/sec allows 20 ms per output token. A mean of 4 emitted tokens needs cycles at most 80 ms; 6 needs 120 ms; 8 needs 160 ms. Eight rows do not imply eight accepted tokens. The full-context cache must be reused across verification rows for these optimistic economics to help.

Q4_0 has 18 bytes per 32 values versus Q8_0's 34. Applying that ratio to the same KV geometry gives 2304 MiB rather than 4352 MiB, saving 2048 MiB, or 2.147 decimal GB. This is allocation arithmetic, not a tested Q4-cache result for this target. It can create room for the draft and scratch buffers and reduce cache traffic. Accuracy, Metal kernel support and sustained speed need validation. TurboQuant targets cache compression too; its paper does not demonstrate this engine/model/device combination. Source: https://arxiv.org/abs/2504.19874 .

## Ranked experiments

| Order | Experiment | Decision it settles |
| --- | --- | --- |
| 1 | Run Splash's header/configuration checks for the local exact GGUF, then one native target-plus-draft screen under a 20 decimal GB RSS watchdog. Match temperature and reasoning mode. | Whether the exact weights load, fit and gain useful acceptance in the native execution path. |
| 2 | Compare native and llama.cpp on identical teacher-forced prefixes and draft conditioning. Check embedding/tokenizer, mask vector, feature layer indexing, normalization, convolution history, KV positions, candidate rankings and committed recurrent state. | Native good / llama poor points to a conversion or execution gap. Both poor makes target/draft mismatch a hypothesis worth testing. Neither outcome proves retraining is required. |
| 3 | Test Q4 target KV and smaller retained recurrent state/checkpoints. Account for draft, host mirrors, scratch and rollback state separately. | Whether there is enough memory for an effective speculative block at occupied 128K. |
| 4 | Profile target verification GPU time, then prototype one high-cost IQ2/IQ3 multi-row projection with register tiling and shared dequantization across rows. | Whether kernel changes can make verification cheap enough at observed acceptance. Port only a measured win. |
| 5 | Add a recurrent-state-correct candidate tree if top-k coverage is good but the selected chain is poor. | Whether branching recovers useful draft choices. Preserve target verification for every committed token. |
| Conditional | Train or distill a small feature-conditioned draft against the exact target's outputs and conditioning, if both engines show poor acceptance. | Whether acceptance can recover without replacing the target. Training compute and data requirements are unresolved; no local-Mac training promise. |

ANE is lower priority for generation. The local synthetic prototype proved an FFN operation runs, not a decode speedup. Native published comparisons already achieve their reported gains with ANE off. CPU/GPU concurrency shares memory bandwidth; ANE does not multiply that bandwidth. Prefix caching can reduce repeated prefill latency but cannot establish faster fresh-token generation at occupied 128K. Context eviction, rolling windows, a smaller model or a faster remote GPU would change the requirement and are not counted as solutions.

## Acceptance gate

First screen short prompts to avoid spending hours on an ineffective path. Then measure fresh inputs at 4K, 32K, 64K and near 128K total occupancy, leaving space for output inside the 131072-token limit. Use code and prose rather than only repetitive text. Record actual tokenized input length, 512 or more generated tokens where feasible, temperature, reasoning mode, emitted tokens per verification, cycle GPU time, prompt time, RSS, Metal allocations, memory pressure and swap deltas. Keep warm-prefix results separate. Check exact greedy answers within each engine with speculation on and off; cross-engine float arithmetic need not be bit-identical. Add long-range retrieval and quality checks before accepting compressed KV.

The final success criterion is single-request sustained generation at or above 50 tokens/sec near 128K occupied context, no context shrink or eviction, at most 20 decimal GB observed process memory, and acceptable retained quality. A sampled watchdog is not an OS hard cap. No experiment in this investigation ran the model or altered the active server.

Principles used: Attack the Premise ruled out another small scheduler or ANE tweak as the assumed solution. Explain the Number required explicit hardware arithmetic and separated configured context, occupied context, single-request throughput and published aggregate results.
