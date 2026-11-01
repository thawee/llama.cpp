---
title: Measure DFlash with its target context attached
author: Thawee
date: 2026-11-01
branch: local/memory-budget
id: l-3fe8c162
scope: repo
status: active
src: agent
touches:
  - common/arg.cpp
  - common/common.cpp
  - common/common.h
  - common/fit.cpp
  - common/fit.h
  - tests/test-arg-parser.cpp
---

Independent DFlash no-allocation measurement fails because its graph borrows the target's token embeddings through ctx_other. Measure a retained target no-allocation context and attach it to the draft before measuring the draft. Add target and draft buffers once, then account for target logits, extracted feature copies and prompt state. The generic decoder-only output reservation must not be applied to a selector draft as if it emitted vocabulary logits. Keep unsupported production budget modes rejected until all retained state is measured; a successful structural load does not prove useful speculative acceptance on a modified target.

In the code: tasks/dflash-preflight.cpp constructs both no-allocation contexts with dcp.ctx_other=context.get(); common/speculative.cpp attaches the real target the same way; common/fit.cpp common_fit_measure currently rejects decision heads with include_output=true. Evidence: tasks/dflash-preflight.log and tasks/dflash-results.json. The isolated experiment uses its own preflight and watchdog, not an expanded production budget gate.
