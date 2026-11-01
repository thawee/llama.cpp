# IQ3 at 64K: llama versus saved private Kapitsa results

Kapitsa was faster on all four matched workloads. Llama reached 14.166 generated tokens/sec on short code; its other measured rates were 7.302-10.535 tokens/sec. This is one new llama trial against five saved Kapitsa trials, not an alternating rerun or an isolated implementation speedup measurement. Production remains IQ2, 128K, ngram speculation on port 8000.

| Workload | Total input tokens | Cached input | llama tokens/sec | Kapitsa saved median (range), tokens/sec | llama request sec | Kapitsa median request sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short_prose | 39 | 0 | 7.302 | 12.643 (12.621-12.668) | 14.167 | 8.087 |
| short_code | 43 | 0 | 14.166 | 23.047 (22.603-23.114) | 7.820 | 4.798 |
| long_prose | 4606 | 0 | 8.007 | 12.569 (12.283-12.655) | 79.363 | 35.111 |
| long_code | 4610 | 4567 | 10.535 | 17.004 (16.979-17.198) | 10.130 | 6.297 |

## What matched

- Same published IQ3_M GGUF source as the native Kapitsa conversion, 65536 configured context, MTP depth 5, non-thinking generation, temperature zero, maximum 96 output tokens, four sequential requests after warmup.
- Questions and the 350-constant system message were checked against the saved Kapitsa benchmark source and match exactly. Both engines counted 39/43/4606/4610 input tokens, including the same 4567 cached tokens on long code. All new requests generated 96 tokens.
- Throughput uses 95 decode intervals for 96 generated tokens. Llama returns predicted_per_second; saved Kapitsa rates were recomputed as (completion_tokens - 1) / decode_seconds from raw milliseconds, rather than rounded console output.
- Short and long code outputs matched all five saved Kapitsa outputs exactly. Prose outputs differed. The code samples are truncated and were not executed as complete programs. Arithmetic smoke check returned 42; this is not a general quality evaluation.

## Configuration differences and limits

- Llama: Metal GPU, no ANE, one slot, batch/microbatch 64, five CPU generation/batch threads, explicit Q8_0 target and draft KV, 22 decimal GB sampled-memory watchdog. The runtime memory-budget fitting gate rejects MTP, so no --memory-budget flag was passed; --fit off retained the explicit context.
- Saved Kapitsa: private.6 fixed-depth runs, with private.7 default-off output parity recorded in its report. Native IQ3 kernels plus ANE, INT8 KV, two configured slots with sequential requests, 4096 prefill chunk and 26 GB launcher budget. Q8_0 and native INT8 KV are different implementations. This compares these configurations, not identical backends or equal tuning.
- Long prose prefill took 67.466 seconds in llama. Long code reused the system prefix. Request latency therefore depends strongly on cache and prefill settings as well as decode.
- One new trial cannot establish repeatability or eliminate changing background load. Saved Kapitsa results were reused as requested. Earlier IQ2 general-thinking tests have different prompts and sampling, so they do not isolate the effect of switching IQ2 to IQ3.
- Maximum new input was 4610 tokens. Full occupied 64K speed, quality and peak memory remain unverified. The saved Kapitsa comparison also used only about 4600 input tokens.

## Memory and restoration

Peak sampled RSS was 20.238 decimal GB; physical footprint peaked at 5.952 GB. The watchdog sampled every 0.5 seconds and checked the larger measure against 22 GB; this is not an enforced OS cap. Pressure stayed at level 1, no watchdog events occurred. System swap used was 1187.56 MiB before and after; counters are system-wide.

The corrected single-slot no-allocation preflight estimated 18.699 GB before excluded metadata, HTTP and temporary compilation allocations. Its estimate is not a runtime RSS bound. The first sizing invocation omitted -np 1 and failed on the server automatic-slot sentinel; the corrected command and successful result are preserved separately.

The production controller restored its original IQ2 model at 131072 context. Ownership, readiness and /props context were verified after the test; no production launcher settings were changed.

## Weight provenance and artifacts

A resumed IQ3 download was stopped when a complete local source was identified. The native-iq3-source.safetensors file was an APFS clone of the full original GGUF with its first 65544 bytes replaced by a safetensors header. recover-iq3.py cloned it separately and restored those bytes from the pinned GGUF header fixture. The complete recovered 14,532,048,416-byte file matched publisher SHA256 19ee0a94937a3b028f7f2565f9b210c2a15f32738b237ea466b0a96cb61717b0. Original Kapitsa and IQ2 files were unchanged.

- recovered-iq3.json: source path, publisher revision, file size and full hash proof.
- results-65536-mtp5-q8_0-limit22-iq3-kapitsa.json: commands, responses, timing, samples, swap counters and restoration.
- comparison-iq3-kapitsa.json: recomputed saved medians/ranges and output comparisons, with original-results hash.
- preflight-iq3-64k-slot1.log and context65536-q8_0-limit22-iq3-kapitsa-mtp5-1.log: sizing and runtime evidence.
- Saved private Kapitsa reference: ~/.local/bin/tasks/adaptive-mtp-20261031/report.md and benchmark-results.json. The separate 20.7 tokens/sec grouped-kernel result used an 8K GPU-only setup and a different prompt, so it is not the baseline in this table.
