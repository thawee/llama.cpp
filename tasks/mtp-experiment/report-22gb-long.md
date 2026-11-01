# Longer-prompt MTP screen with a 22 GB ceiling

The candidate passed the tested histories below 22 decimal GB, but generation remained below 9 tokens/sec. Keep production unchanged: this run shows headroom, not a speed improvement or full-128K validation.

| Prompt tokens | Generated tokens | Prompt tokens/sec | Generated tokens/sec | Request wall time, sec |
| ---: | ---: | ---: | ---: | ---: |
| 3018 | 128 | 68.97 | 8.76 | 58.28 |
| 12118 | 128 | 65.94 | 7.87 | 199.93 |

Peak sampled RSS: 19.879 decimal GB. Peak sampled physical footprint: 7.875 GB. Watchdog checked the larger of both every 0.5 seconds against 22,000,000,000 bytes. Memory pressure remained at level 1, with no watchdog events. The arithmetic smoke response was 42.

The ceiling was a watchdog setting, not an allocation request or enforced OS process cap. The run never used 22 GB. No runtime memory-budget flag was used because its gate rejects native MTP; fixed context and explicit KV settings were retained.

## Method and limits

- Same IQ2_M target plus extracted native MTP head, depth 5; target and draft KV both Q8_0. Metal, one slot, context 131072, five threads, batch/microbatch 64.
- One trial with two synthetic repetitive library-update histories, 80 and 320 numbered entries. These are memory/speed screens, not representative quality evaluation. Both requests generated 128 tokens including reasoning with temperature 1 and seed 1234, uncached prompts and forced token count.
- Neither a complete answer nor useful answer throughput was measured. MTP acceptance was 81/224 and 82/221 respectively.
- Saved short-prompt results remain in report.md and results.json. No baseline rerun. Different prompts and run times prevent attributing throughput differences to the memory ceiling.
- Full occupied 128K speed, quality and memory remain unverified. About 12K input tokens is the longest history tested in this run.
- Background applications remained open. System-wide swap used was 1195.56 MiB before and after; swapouts did not increase, swapins increased by 47 pages, and pageouts increased by 869 pages. These are system-wide counters, not attribution to the server. Flat swap usage does not prove all paging was absent.
- Production was restored successfully with its original ngram settings, verified owned/ready at port 8000 and 131072 tokens. No production memory ceiling or launcher was changed.

Raw evidence: [results-131072-mtp5-q8_0-limit22-long.json](results-131072-mtp5-q8_0-limit22-long.json), benchmark-128k-q8-limit22-long.log and context131072-q8_0-limit22-long-mtp5-1.log.
