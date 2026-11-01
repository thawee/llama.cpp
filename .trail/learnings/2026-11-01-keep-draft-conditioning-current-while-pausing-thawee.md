---
title: Keep draft conditioning current while pausing proposals
author: Thawee
date: 2026-11-01
branch: local/memory-budget
id: l-c3a754d9
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

## Decided
- Acceptance backoff skips only the DFlash2 proposal pass, while target-feature processing continues. A resumed draft needs the target features from the skipped interval; dropping conditioning updates would leave stale KV state. This also means backoff cannot remove all draft overhead.
- Cheap ngram implementations already precede learned drafts in common_speculative_init, independent of CLI type order. Preserve that order and ignore other implementations' acceptance callbacks when updating the learned draft's backoff.

In the code: common/speculative.cpp common_speculative_impl_draft_dflash::process, ::draft, ::accept, and common_speculative_init; common/speculative.h common_speculative_backoff. The local test is tests/test-speculative-backoff.cpp and the real-model harness is tasks/dflash-check.py --adaptive-comparison.
