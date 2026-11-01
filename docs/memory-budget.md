# Local CPU/Metal memory budget

Production now uses IQ3 with a Q8 output head and native MTP5 at 128K. That profile uses `--fit off`, not the IQ2 budget estimator described below. See [local installation](local-install.md) for current settings and rollback. Historical IQ2 measurements below do not describe current production memory.

## Local launcher

From this checkout, run:

```sh
./scripts/run-local-128k.sh
```

The launcher uses the downloaded IQ2_M model and the verified runtime copied to `build-m3-local/bin`. It binds to `127.0.0.1:8080`, runs offline, and uses one slot, 128K context, Q8 KV cache, bounded ngram-simple speculation and an 18 GiB estimate budget (18432 MiB). Open `http://127.0.0.1:8080` for chat, or use its local API. Stop it with Ctrl+C. Additional server arguments can be appended, for example `--port 8081`.

To disable speculation, run `LLAMA_LOCAL_SPECULATIVE=0 ./scripts/run-local-128k.sh`. To also restore the earlier 16 GiB allowance, append `--memory-budget 16384`. Use the environment switch because repeated `--spec-type` arguments add modes rather than replace them.

The runtime copy is local build output, not a rebuild mechanism. After changing C++ code, rebuild and refresh it before using the launcher to test those changes.

## Local measurements

Five alternating trials per batch size used the same 359-token prompt and generated exactly 64 tokens, with prompt reuse disabled, temperature 0, and ubatch64. Each server was warmed before measurement. The Release Metal build ran with existing applications left open.

| Logical batch | Prompt tokens/sec, median (range) | Generated tokens/sec, median (range) | Completion request seconds, median |
| --- | --- | --- | --- |
| 64 | 63.42 (59.43-63.54) | 7.59 (7.49-7.61) | 13.965 |
| 128 | 63.52 (63.21-63.65) | 7.59 (7.41-7.70) | 13.976 |

All ten requests completed and reported the expected token work. The ranges overlap; these trials do not establish a speed advantage for batch128. The launcher keeps batch64 because it reserves about 61 MiB less output memory. This comparison holds ubatch fixed and does not establish optimal settings for every workload.

Arithmetic and JSON checks passed on both configurations. Both produced the correct Python addition function but included Markdown despite an instruction to omit it. Both retrieved the correct key from a 3945-token synthetic prompt with a label added to the answer. These are small correctness checks, not a general quality score or a measurement of IQ2 quantization loss.

An additional batch64 retrieval check used 9057 prompt tokens and clearer value-only instructions. It returned the exact key in about 137 seconds. All nine quality requests returned the expected content; four missed the original strict formatting checks. Full-128K retrieval and accuracy remain untested.

Across these runs, sampled process RSS peaked at 16156.66 MiB (about 15.78 GiB). System pressure readings stayed at level 1, and swap use ranged from 3240.06 to 3248.06 MiB, without increasing above its starting value. Existing swap was already in use before testing. RSS is not a complete unified-memory accounting or an enforced cap. Verbose logs confirmed Metal execution with all 65 model layers offloaded to the GPU.

The reproducible harness and raw measurements are in `tasks/local-runtime-check.py` and `tasks/local-runtime-results.json`. Run `python3 tasks/local-runtime-check.py` to repeat the checks; it replaces the results file and can take several minutes.

## Decode investigation

A separate CPU profile found the decode worker predominantly waiting for Metal completion. That identifies GPU execution as the main wait; it does not establish whether bandwidth, dequantization, or GPU arithmetic is the limiting part. Fusion, concurrency and graph optimization were enabled, with all layers offloaded.

Bounded screens found no reliable benefit from changing CPU thread count or disabling GPU concurrency. A later five-run comparison kept the existing five-thread default: eleven generation threads were 1.3% slower and eleven batch threads changed prefill by only 0.1%; see [CPU thread experiment](../tasks/cpu-core-report.md). Two IQ2_s kernel row-reuse experiments passed numerical checks but did not clear the threshold for repeated confirmation. The original kernel and baseline settings were restored. Evidence is in `tasks/decode-profile-notes.md`, `tasks/decode-tuning-results.json` and `tasks/decode-kernel-results.json`.

The first ngram-simple experiment used no second model. Five alternating code-generation pairs gave median generation speeds of 7.52 tokens/sec without speculation and 8.20 with speculation, a 9.1% gain; median end-to-end time fell 7.9%. All five paired outputs matched exactly. Summary and repeated-copy screens also matched, with the copy case showing a much larger workload-dependent gain. These exploratory processes disabled the budget gate after a separate preflight and used host rollback checkpoints. They are not the final budget-backed configuration. Raw results are in `tasks/speculative-results.json`.

The budget-backed candidate instead requests three recurrent rollback snapshots on the GPU. The existing estimator includes those buffers. With ngram-simple lookup/draft size 3, batch64, Q8 KV and 128K context, the estimate is 16743.64 MiB including a speculative prompt-copy allowance. It fits the new 18432 MiB allowance and correctly rejects the earlier 16384 MiB allowance. Final runtime and throughput validation are recorded separately in `tasks/speculative-rs-results.json`.

Five alternating budget-backed code-generation pairs gave median generation speeds of 7.526 without speculation and 8.652 tokens/sec with it, a 15.0% gain. The observed ranges were 7.443-7.621 and 8.502-8.902 respectively. Median end-to-end time fell from 17.820 to 15.626 seconds (12.3%). Every paired output matched exactly and the generated functions passed structural checks. These are workload-specific measurements, not a guarantee for every prompt.

The final path completed 16 measured requests with zero errors. Peak sampled RSS was 16308.19 MiB (15.93 GiB), memory pressure stayed at level 1, and swap decreased from 3128.06 to 3120.06 MiB. A separate verbose run confirmed three rollback snapshots, 598.50 MiB of recurrent buffers, disabled prompt checkpoints, and no speculative host checkpoint. A 1857-token prompt retrieved `amber-7319` exactly with speculation enabled. This validates more than short initialization, but full-128K quality remains untested. The default launcher now enables this bounded mode; use `LLAMA_LOCAL_SPECULATIVE=0` when it is unhelpful for a workload.

## Budget behavior

`llama-server` and `llama-cli` accept `--memory-budget MiB` to limit estimated inference buffer allocations. The default is `0`, which preserves the existing fitting behavior.

For a 16 GiB budget and an explicit 128K context target:

```sh
llama-server -m /path/to/model.gguf --memory-budget 16384 -np 1 -c 131072 -b 64 -ub 64 -fa on -ctk q8_0 -ctv q8_0
```

The estimate includes model weights, KV/context buffers, compute buffers, a conservative persistent logits reservation, and any explicitly reserved prompt cache. For supported ngram-simple speculation it also includes recurrent rollback buffers and a prompt-copy allowance. It sums CPU and Metal buffers because Apple Silicon shares system memory. This option supports CPU execution or one Metal device. The model must support the requested context and cache types.

With an explicit context size, an estimate above the budget stops model initialization. To let the budget choose a smaller context, omit `-c`, keep `--fit on`, and use `--fit-ctx` to set the smallest acceptable context. If that context still does not fit, initialization stops. Other explicit settings are preserved.

The example uses one server slot so the full context belongs to one conversation. Multiple slots increase memory requirements or divide the configured total context, depending on cache settings.

Locally verified with `models/local/qwen3.8-27b-iq2_m.gguf`: the example configuration estimates 16293.76 MiB, the server reports 131072 context tokens and one slot, and a short chat request generates successfully. The downloaded IQ2_M file passed SHA256 verification. This checks initialization and short generation; it does not establish accuracy over a full 128K prompt.

Budget mode disables the optional prompt cache unless you explicitly set `--cache-ram`. A finite explicit cache size is reserved within the budget; an unlimited cache is rejected.

Budget mode also disables context checkpoints, including explicitly requested ones, with a warning. These are separate from `--cache-ram`: recurrent/hybrid models can otherwise retain multiple host copies of recurrent state during prompt processing. Those copies are not included in the buffer estimate. Disabling checkpoints can reduce prompt reuse after edits, while retaining the configured context capacity.

Budget mode supports only ngram-simple speculation with one slot and `1 <= size-n <= size-m <= 4`. Recurrent/hybrid models must support the requested rollback count; SWA models are rejected for this path. Other speculative modes, multimodal projection models, adapters, tensor placement overrides, backend sampling, embeddings, and encoder models are rejected because their additional allocations are not included in the estimate. In router mode the budget applies separately to each loaded model; it is not an aggregate limit across models.

This is not a process/RSS limit or a live macOS memory-pressure controller. Tokenizer metadata, HTTP and sampler state, transient allocations, and other applications can consume additional memory. Leave headroom and watch Activity Monitor's memory pressure and swap use. A budget cannot make model weights smaller or guarantee faster inference.
