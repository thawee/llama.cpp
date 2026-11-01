# M3 Pro CPU thread experiment

Keep the existing five-thread default. Increasing generation threads to eleven reduced median generation throughput by 1.3%; increasing batch threads to eleven changed prefill throughput by only 0.1%. Neither cleared the predeclared 3% generation or 5% prefill screen threshold. No production configuration or runtime binary was changed.

## Hardware and method

This M3 Pro has 5 performance cores and 6 efficiency cores (11 physical/logical cores). Original llama.cpp already selects the performance-core count through `hw.perflevel0.physicalcpu` in `common/common.cpp`. Apple CPU affinity is a no-op in `ggml/src/ggml-cpu/ggml-cpu.c`; a thread count of five does not prove performance-core placement.

The installed runtime and exact IQ2_M model were held constant: Metal all layers, one slot, configured context 131072, Q8 K/V, batch/microbatch 64, ngram-simple 3/3, reasoning off, and 18432 MiB inference estimate budget. Each trial used a fresh server on port 18092, an eight-token warmup, a 50-token coding prompt with 128 generated tokens, then a 1056-token synthetic retrieval prompt with 16 generated tokens. Requests used temperature zero, seed 1234, ignore_eos=true, and cache_prompt=false. These fixed-length microbenchmarks are not natural conversation or occupied-128K tests.

Eight settings were screened once each. Five further trials each compared baseline, eleven generation threads, and eleven batch threads, reversing configuration order on alternate rounds. Active applications stayed open, so small differences can reflect system activity or time drift. The shared server was stopped during measurement and restored afterward. Independent trials used the same outputs: one output hash per workload across all fifteen confirmation runs; arithmetic/docstring and retrieval-needle checks passed.

## Five-run confirmation

| Generation / batch threads | Generation tokens/sec median (range) | Prefill tokens/sec median (range) | Median coding / retrieval wall seconds |
| --- | --- | --- | --- |
| 5 / 5 | 9.336 (9.187-9.444) | 72.719 (72.575-73.437) | 14.491 / 15.891 |
| 11 / 5 | 9.215 (9.184-9.237) | 72.792 (72.726-72.855) | 14.670 / 15.896 |
| 5 / 11 | 9.357 (9.352-9.377) | 72.808 (72.741-72.823) | 14.462 / 15.867 |

## Single-run screen

| Setting | Generation tokens/sec | Prefill tokens/sec |
| --- | --- | --- |
| baseline | 9.072 | 71.648 |
| generation1 | 9.184 | 71.681 |
| generation2 | 8.880 | 70.062 |
| generation11 | 8.838 | 70.469 |
| batch1 | 8.907 | 70.979 |
| batch2 | 8.839 | 69.652 |
| batch11 | 8.676 | 71.127 |
| priority1 | 9.167 | 71.632 |

Single-run screens are exploratory. Priority1 is not a verified process-priority comparison: `setpriority(-5)` failed with EACCES and process nice remained zero. The separate native pthread probe successfully set and read back SCHED_FIFO priority 40, but this does not establish actual server worker placement or a throughput gain. Its QoS attempt occurred after setting FIFO and cannot establish whether QoS works on an otherwise untouched thread.

## Memory, CPU and limits

Peak sampled RSS was 17.102 decimal GB; peak physical footprint was 5.437 GB. The 0.5-second watchdog saw no 20,000,000,000-byte ceiling event or critical memory-pressure event; all sampled pressure levels were 1. Sampling can miss brief peaks. RSS and physical footprint use different macOS accounting and are not interchangeable.

| Generation / batch threads | Median CPU use during coding | Median CPU use during retrieval |
| --- | --- | --- |
| 5 / 5 | 17.30% | 4.46% |
| 11 / 5 | 33.72% | 5.01% |
| 5 / 11 | 15.31% | 7.90% |

100% means one fully occupied CPU core. `proc_pid_rusage` CPU times use Mach clock ticks on this machine. They were converted with mach_timebase_info (125/3 nanoseconds per tick), verified against a short process_time calibration. Initial screen CPU percentages had incorrectly assumed nanoseconds; corrected values preserve the original under cpu_percent_before_timebase_correction. Server token timings and memory readings were unaffected. This low host CPU use and the earlier Metal-wait profile give no evidence of CPU core count being the main limiter; they do not distinguish GPU bandwidth, dequantization, arithmetic, or command latency.

Configured 128K capacity is verified, but occupied-128K quality, peak memory and throughput remain unverified. This experiment does not meet the 25 or 50 generated tokens/sec goal and does not justify a CPU scheduling patch.

## Evidence and reproduction

- `tasks/cpu-core-benchmark.py --self-test` checks the harness without changing services.
- Run `python3 tasks/cpu-core-benchmark.py` for the screen, then `python3 tasks/cpu-core-benchmark.py --confirm-all-core` for the five-run comparison. These commands stop and restart the owned shared server.
- `tasks/cpu-core-results.json` contains exact arguments, runtime hash, outputs, timings and memory samples; logs are in `tasks/cpu-core-benchmark.log`, `tasks/cpu-core-confirmation.log` and per-trial log files referenced by the JSON.
- `tasks/cpu-scheduling-probe.c` and `.log` record the native scheduling probe; `tasks/cpu-clock-calibration.json` records the time-unit calibration.
- `tasks/cpu-core-restored-check.json` verifies restored health, model alias, configured context and arithmetic response. The restored server reports owned=True ready=True.
