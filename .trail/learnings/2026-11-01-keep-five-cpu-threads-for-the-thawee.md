---
title: Keep five CPU threads for the Metal Qwen workload
author: Thawee
date: 2026-11-01
branch: local/memory-budget
id: l-e029fee3
scope: repo
status: active
src: agent
touches:
  - common/common.cpp
  - ggml/src/ggml-cpu/ggml-cpu.c
  - README.md
---

On this M3 Pro (5 performance and 6 efficiency cores), five alternating trials per configuration measured median generation at 9.336 tokens/sec for 5/5 generation/batch threads versus 9.215 for 11/5. Batch threads 11 changed prefill from 72.719 to 72.808 tokens/sec. Keep the existing five-thread default because extra threads did not clear the declared gain thresholds. These are short IQ2_M coding/retrieval workloads with configured 131072 context, not occupied-128K validation. Apple affinity is a no-op, so thread count is not core pinning. In the code: tasks/cpu-core-results.json and tasks/cpu-core-report.md contain evidence; tasks/cpu-core-benchmark.py converts proc_pid_rusage CPU Mach ticks using mach_timebase_info (125/3 ns per tick), calibrated against process_time. RSS peak was 17.102 decimal GB, no 20 GB watchdog events. No production settings changed.
