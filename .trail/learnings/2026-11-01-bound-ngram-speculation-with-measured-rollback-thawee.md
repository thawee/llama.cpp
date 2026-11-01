---
title: Bound ngram speculation with measured rollback buffers
author: Thawee
date: 2026-11-01
branch: local/memory-budget
id: l-7264017b
scope: repo
status: active
src: agent
touches:
  - common/common.h
  - common/common.cpp
  - common/fit.cpp
---

Local/private M3 Qwen35 27B optimization keeps IQ2_M, Q8 KV and 131072 context. At user-approved18432MiB budget, allow only ngram-simple with one slot and1<=N<=M<=4. Request RS snapshots=M only in budget mode; estimator measures their buffers and verifies actual recurrent/hybrid rollback support, rejecting SWA. This avoids the uncounted FULL host checkpoint. With N=M=3 total estimate16743.64MiB, while16384MiB rejects. Five alternating real-model coding pairs improved7.526->8.652tokens/sec (+15%) with identical output. Peak sampledRSS16308.19MiB; full128Kquality untested. --spec-type appends NONE and repeated modes, so ignore NONE in mode classification and use LLAMA_LOCAL_SPECULATIVE=0 to disable launcher defaults. Retain zero-budget behavior. Evidence: tasks/speculative-rs-results.json, speculative-budget-preflight.json, speculative-launcher-check.json.
