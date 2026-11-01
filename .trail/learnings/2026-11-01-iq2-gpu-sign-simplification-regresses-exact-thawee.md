---
title: IQ2 GPU sign simplification regresses exact-model decode
author: Thawee
date: 2026-11-01
branch: local/memory-budget
id: l-759caaf9
scope: repo
status: active
src: agent
touches:
  - ggml/src/ggml-metal/kernels/mul_mv.metal
  - ggml/src/ggml-metal/ggml-metal-device.m
---

Actual M3 Pro probe supports encoder-stage timestamp counters but not dispatch-boundary counters. Use temporary stage attachments and a separately labeled matmul-split diagnostic, not unsupported per-dispatch samples or CPU waits, for attribution. Exact Qwen27B IQ2 target: unsplit four decode steps average121.59ms GPU; split two steps124.90ms, IQ2-containing groups86.96ms, BF16 head group19.23ms. Split groups can include preceding work and alter scheduling, so these are not undisturbed kernel shares or bandwidth counters. Integer-domain grid sign before float conversion in IQ2_S passed37 CPU/Metal numerical cases perbuild but regressed five alternating128-token full-model replies: baseline median9.3460 range9.3303-9.3528 tok/s, candidate9.0442 range9.0260-9.0489 (-3.23%). Identical outputs, sampled RSS17.091GB and physicalfootprint5.424GB, no watchdog events. Rejectedpatch and exactsource restoration; retainoriginal. Do not repeat this source-level arithmetic simplification as a presumed optimization; compiler lowering and memory traffic remain unresolved. Artifacts tasks/metal-decode-optimization.md, tasks/metal-decode-experiment/, tasks/metal-iq2-signed-grid-results.json. Shortprompt withconfigured128K is not occupied128K. Serverrestored/APIchecked.
