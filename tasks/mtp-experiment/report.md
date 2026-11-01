# Native MTP screen on M3 Pro

Depth 5 at 131072 configured tokens with Q8 draft KV passed this short screen at 19.920 GB peak sampled RSS, leaving only 80 MB below the watchdog limit. It did not reach 25 tokens/sec. Production remains at 131072 tokens with ngram speculation on port 8000.

| Configured context | Speculation | Summary tokens/sec | Planning tokens/sec | Peak sampled RSS, decimal GB | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| 131072 | baseline | 10.461 | 8.777 | 17.086 | Passed smoke check |
| 131072 | mtp1 | 9.655 | 8.345 | 19.453 | Passed smoke check |
| 131072 | mtp2 | 10.912 | 9.422 | 19.782 | Passed smoke check |
| 131072 | mtp3 | 12.073 | 9.534 | 19.945 | Passed smoke check |
| 131072 | mtp5 | 11.373 | 8.579 | 20.106 | Failed 20 GB watchdog |
| 65536 | mtp5 | 11.444 | 8.645 | 17.649 | Passed smoke check |

| 131072 | mtp5, Q8 draft KV | 11.424 | 8.643 | 19.920 | Passed smoke check |

## Method and limits

- Each row is one trial, two short general-task prompts, 128 generated tokens per prompt including reasoning. General-thinking sampling uses temperature 1, top-p 0.95, top-k 20 and seed 1234. Counts do not measure completed useful answers.
- Reused the saved 128K baseline as requested. No baseline was rerun for the 64K or Q8 draft-cache experiment. Comparisons are unpaired, and the 64K row also changes context allocation. Sampled summary outputs differ, so these are observations rather than isolated causal speedup measurements.
- Prompts contain only 125 and 126 tokens. No occupied-64K or occupied-128K throughput, memory or quality claim follows from this screen.
- One slot, five generation/batch threads, Metal, batch/microbatch 64. Target KV is Q8_0; MTP draft KV used the independent F16 default except for the explicitly labeled Q8 draft-KV row.
- The arithmetic smoke check returned 42 for the 64K depth-5 trial. This is not a general quality evaluation. The 128K depth-5 trial exceeded the sampled 20 GB limit during its quality check and was stopped.
- The original preflight copied target Q8 cache settings to the draft, underestimating the tested F16 draft configuration. The sizing helper now uses common_base_params_to_speculative, matching server draft settings. Its earlier 19.30 GB estimate must not be treated as a validated runtime bound.
- The Q8 candidate explicitly set --spec-draft-type-k q8_0 and --spec-draft-type-v q8_0. Both measured responses generated 128 tokens; the arithmetic smoke check returned 42. Draft acceptance was 90/180 for summary and 78/240 for planning. Relative to the earlier F16 depth-5 run, observed throughput changed by less than 1%; this unpaired single run establishes no speed improvement.
- Retain production ngram speculation: the Q8 MTP candidate leaves only 80 MB below 20 GB in short-prompt testing and does not demonstrate a reliable speed advantage. A full history can change memory and speed. Pressure stayed at level 1 and the watchdog recorded no events. Background applications remained open; load averages before the trial were 2.32, 1.98 and 2.18.

## Model and evidence

The original IQ2_M target is unchanged. A separate APFS clone plus split GGUF adds the pinned compatible MTP head. Preparation compared SHA256 of the full target tensor payload with the original; hashes match. See prepared.json and source.json for provenance.

- results.json: original 128K screen and baseline.
- results-65536-mtp5.json: requested 64K depth-5 trial, watchdog samples, smoke response and successful service restoration.
- results-131072-mtp5-q8_0.json: Q8 draft-cache candidate, samples, smoke response and successful restoration.
- Per-trial logs preserve launch flags and native speculative acceptance counters.

## 22 GB ceiling and longer prompts

A follow-up tested 3018 and 12118 input tokens at 128K configured context. Generation measured 8.763 and 7.872 tokens/sec, peak sampled RSS 19.879 GB, pressure level 1 and no extra swapouts. See [longer-prompt measurements](report-22gb-long.md). This did not reach 25 tokens/sec; production remains unchanged.
