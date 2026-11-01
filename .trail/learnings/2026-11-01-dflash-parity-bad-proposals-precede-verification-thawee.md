---
title: "DFlash parity: bad proposals precede verification"
author: Thawee
date: 2026-11-01
branch: local/memory-budget
id: l-b57348c3
scope: repo
status: active
src: agent
touches:
  - common/speculative.cpp
  - src/models/qwen35.cpp
---

On the exact fused Qwen 27B IQ2_M coding prefix, Splash and llama.cpp have identical tested tokenizer output and chat prompt, 17/17 matching target greedy logits, and mean capture cosines 0.99985 or better. HF output layers [5,19,33,47,61] intentionally map to llama layer inputs [6,20,34,48,62]. Native affine4, llama Q8_0 and original BF16 DFlash2 via released MLX reference all propose the wrong first continuation token before verification. Do not port a verifier or blame draft quantization alone for this prefix; first establish a known-good reference fixture for the intended base target to separate shared setup from target mismatch. This does not prove retraining is needed. Monitor both process-group RSS and physical footprint because native Metal and mapped llama weights are accounted differently. Evidence: tasks/dflash-conditioning-parity.md and accompanying JSON. Short-prefix tests do not validate occupied 128K speed or memory.
