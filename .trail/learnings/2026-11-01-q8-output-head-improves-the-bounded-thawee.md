---
title: Q8 output head improves the bounded IQ3 screen
author: Thawee
date: 2026-11-01
branch: local/memory-budget
id: l-38ff1a0f
scope: repo
status: active
src: agent
touches:
  - common/speculative.cpp
  - ggml/src/ggml-metal/ggml-metal-common.cpp
---

The 27B IQ3 model has a 2.543 GB BF16 output.weight as its first tensor. Q8_0 reduces it to 1.351 GB; GGUF requires contiguous offsets, so APFS clone with a hole is invalid and the remaining865 tensors must shift. The compact model passed publisher source SHA and unchanged-tail readback checks. Single candidate screens reached9.56/17.12 tok/s prose/code at64K versus saved7.30/14.17; configured128K qualified at9.51/17.56 and21.127 decimal GB RSS. Full occupied128K and general quality unverified; fixed128token thinking screen15.66tok/s ended beforefinalanswer. Batch256 and bulk host snapshot showed no clear gain. Do not copy grouped IQ3 decode blindly: Metal already groups it. M3Ultra-tuned MMA overrides remain experimental and overlapping model-I/O screens cannot select a winner. Production remains IQ2/128K/ngram/8000.
