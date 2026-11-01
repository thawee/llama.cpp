---
title: Use the native exact-format path before porting speculative kernels
author: Thawee
date: 2026-11-01
branch: local/memory-budget
id: l-6eaccf25
scope: repo
status: active
src: agent
touches:
  - common/arg.cpp
  - common/common.cpp
  - common/common.h
  - common/fit.cpp
  - common/fit.h
  - common/speculative.cpp
  - common/speculative.h
  - tests/CMakeLists.txt
  - tests/test-arg-parser.cpp
---

For dense IQ2 Qwen targets, Splash documents direct IQ block loading while Kapitsa's GGUF importer converts dense IQ2 to 3-bit and IQ3 to 4-bit. Do not treat the converted model as unchanged weights or assume its memory equals the source GGUF. Use native exact-format acceptance as a diagnostic before porting candidate trees or concluding a drafter needs retraining. KFlash explicitly handles Qwen recurrent branches; its small incremental gain over its prior native engine does not explain a whole engine's advantage over llama.cpp. The incoai and z-lab Qwen3.8-27B-DFlash2 model APIs currently expose identical safetensors hashes, so publisher names alone are not evidence of distinct draft weights.
In the code: tasks/50tps-128k-research.md records source URLs and scope. tasks/splash-exact-iq2-header-check.json proves only the local file's Python header/type checks. common/speculative.cpp owns the draft policy; avoid attributing acceptance differences to target fusion without a controlled native comparison.
