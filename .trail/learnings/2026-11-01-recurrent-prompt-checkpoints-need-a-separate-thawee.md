---
title: Recurrent prompt checkpoints need a separate budget
author: Thawee
date: 2026-11-01
branch: local/memory-budget
id: l-f555cfe8
scope: repo
status: active
src: agent
touches:
  - common/fit.cpp
  - tools/server/server-context.cpp
---

Local memory-budget mode must disable optional context checkpoints as well as the prompt cache. n_ctx_checkpoints defaults to 32 and server prompt processing serializes recurrent state independently of cache_ram_mib. In the local Qwen35 27B configuration each recurrent state payload is about 149.62 MiB, so those retained host copies can substantially exceed the buffer estimate. Speculative FULL rollback uses another separate checkpoint; keep speculative modes rejected until that state is accounted for. Source: common/common.h n_ctx_checkpoints; server-context.cpp create_checkpoint and speculative ckpt.update_tgt.
