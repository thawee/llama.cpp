---
title: Size MTP using its independent draft cache settings
author: Thawee
date: 2026-11-01
branch: local/memory-budget
id: l-7b9d3cdd
scope: repo
status: active
src: agent
touches:
  - README.md
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

MTP draft KV does not inherit the target cache type. Using target Q8 parameters for draft preflight underestimates the F16 default and can incorrectly suggest a run will stay below a process-memory limit. Derive draft sizing parameters through the same conversion as the server, and set draft KV flags explicitly when testing Q8. A preflight estimate is still not a process RSS cap.

In the code: common/speculative.cpp common_base_params_to_speculative applies params.speculative.draft.cache_type_k/v independently. tasks/mtp-experiment/preflight.cpp now calls this conversion. tasks/mtp-experiment/benchmark.py exposes --draft-kv and uses --spec-draft-type-k/--spec-draft-type-v, storing separate output files to preserve F16 evidence. The Q8 128K depth-5 short screen fits just below 20 decimal GB but does not establish a useful memory margin or speed improvement; see tasks/mtp-experiment/report.md.
