# Short optimization screens for M3 Pro

The goal remains the selected 27B model at configured 128K, within 22 decimal GB, at least 25 generated tokens/sec and preferably 50. Neither speed target was reached. The useful candidate is IQ3 with only its BF16 output head requantized to Q8_0. It was subsequently promoted to production at the user's request, with native MTP5, 128K and port 8000. The original IQ2 remains available for rollback.

## Results

Each candidate received one screen. The saved baseline was reused without rerunning it. All measured prose/code requests generated 96 tokens; throughput uses 95 decode intervals. These are unpaired screening results, not confirmed improvements or confidence intervals.

| Configuration | Context | Short prose tok/s | Short code tok/s | Peak sampled RSS, GB | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Saved IQ3, BF16 head, MTP5 | 65536 | 7.302 | 14.166 | 20.238 | Existing reference |
| MMA threshold 1 for IQ3_S/BF16 | 65536 | 7.092 | 12.511 | 18.663 | Discard from selection; overlapped model creation |
| MMA threshold 7 for IQ3_S/BF16 | 65536 | 3.698 | 6.960 | 19.225 | Discard from selection; startup overlapped model creation |
| Target batch/microbatch 256, draft microbatch 16 | 65536 | 7.584 | 14.143 | 20.486 | No clear gain |
| Bulk MTP hidden-state snapshot | 65536 | 7.349 | 13.620 | 19.322 | No clear gain |
| Q8 output head, otherwise original settings | 65536 | 9.561 | 17.124 | 19.065 | Promising; qualified at 128K |
| Q8 output head, installed runtime | 131072 | 9.508 | 17.558 | 21.127 | Experimental option; within short-screen budget |

At 64K the Q8-head screen was 30.9% faster on short prose and 20.9% on short code than the saved reference. Long prose was 9.250 tok/s versus 8.007; cached long code was 11.820 versus 10.535. The 128K row is a configuration qualification, not a matched-context delta against the 64K reference.

The additional 128K general-thinking screen generated 128 tokens at 15.660 tok/s with temperature 1, top-p 0.95 and top-k 20. It stopped during reasoning before the final answer. Its rate includes reasoning tokens and does not establish completed-answer quality. Arithmetic smoke checks returned 42. Short prose and both code outputs matched the saved reference; long prose differed after head quantization. Truncated code was not executed as a complete program.

Batch 256 reduced cold 4606-token prompt processing from 67465.574 to 64593.614 ms, only 4.3%, while long-prompt generation was about 5% slower. This did not meet the screening threshold, so batch 128 and combinations were not expanded.

## What changed

- Metal routing has opt-in `GGML_METAL_MMA_IQ3_MIN_ROWS` and `GGML_METAL_MMA_BF16_MIN_ROWS` overrides, accepting 1-17. Defaults remain the upstream thresholds measured on M3 Ultra. IQ3 dequantization was already grouped; no duplicate grouped decoder was added.
- `LLAMA_MTP_BULK_HIDDEN=1` uses one contiguous host snapshot per sequence, avoiding repeated row getters. Owned verification snapshots, pending states and acceptance rollback remain intact. Full GPU-resident handoff was not implemented: current APIs expose host float pointers and the draft graph uploads host embedding state. Removing those transfers safely needs tensor ownership and backend API changes. The tested small change did not justify that larger patch.
- The head-only creator changes `output.weight` from BF16 to Q8_0 and saves 1,191,936,000 bytes. All other 865 tensors passed byte-preservation checks. Source SHA256 matches the publisher, and the compact candidate passed native GGUF parsing. Conversion changes model weights and the selected Metal kernel; the measured delta cannot be assigned solely to reduced byte traffic.

## Verification and limits

The Release experimental server built successfully. Metal override boundary checks passed. UBSan snapshot checks covered eight combinations, sequence isolation, 5120-float rows, target-buffer lifetime and acceptance boundaries. Independent source review found no blocking issue in the snapshot or routing changes. A second review checked the head quantizer ABI and compact GGUF layout. Its publication-race finding was fixed with atomic no-replace linking and a passing concurrent-destination regression check.

All model processes ran serially with a 22 GB sampled RSS/physical-footprint watchdog. The Q8-head creator/hash work overlapped the two MMA screens through 19:12:53 Bangkok, so those results are excluded from selection. Later batch, bulk and head screens ran without that work. The 128K head test used the installed runtime, without experimental code flags. Memory pressure remained normal and system swap used stayed at 1182.94 MiB during that check. Sampling every 0.5 seconds is not an OS memory cap.

Maximum input was 4606 tokens, or 125 for the thinking screen. Full occupied 64K/128K memory, quality and speed remain unverified. No five-run confirmations or production baseline reruns were performed, per the request for short phases. Old Kapitsa results have different backend and batching settings and are not a causal comparison here.

## Use the experimental option

The launcher uses general-thinking sampling, MTP5, Q8 target/draft KV, five threads, batch 64 and the installed runtime. It does not install experimental kernels or modify the shared controller. Its measured memory is not an enforced cap.

Stop the shared server before running it in the foreground:

```sh
llama-local stop
./scripts/run-local-iq3-q8head-128k.sh
```

After stopping that foreground process, `llama-local start` starts the managed production profile. See `docs/local-install.md` for IQ2 rollback.

Raw evidence is in `fast-first-65536.json`, `fast-head-65536.json`, `fast-head128-131072.json` and their logs. `fast-summary.json` includes comparison values and executable/dylib hashes for both runtimes. Head provenance is in `../iq3-head-experiment/candidate.json`; snapshot evidence is in `../mtp-fastpath/report.md`.

## Production promotion

The user selected this candidate for production after the screens. The model moved to `models/local/qwen3.8-27b-iq3-mtp-q8head.gguf`; the shared controller now uses native MTP5 with the existing verified installed runtime. The service was restarted on port8000, API context131072 and process argv were verified, and a completed non-thinking arithmetic request returned42. Pi/Copilot retain their model alias and endpoint. `tasks/production-iq3-q8head.json` records this state. Earlier restoration statements above describe the experiment phase.
