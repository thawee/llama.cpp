---
title: Native Splash exact IQ2 draft acceptance stays low
author: Thawee
date: 2026-11-01
branch: local/memory-budget
id: l-31246160
scope: repo
status: active
src: agent
---

Exact DavidAU qwen3.8-27b-iq2_m.gguf was tested on M3 Pro in native Splash 031a25a, cached incoai DFlash2 safetensors, GPU-only, int8 KV, no disk cache. Three fresh 50-token coding prompts produced identical correct 97-token answers: median 5.197 tok/s, 18/1890 draft tokens accepted (0.95%), 90 speculative cycles per request. Configured context 131072, occupied context under 150 tokens; no long-context speed or memory claim. An 18 decimal GB Metal ceiling rejected 128K (77817 tokens planned); 19.8 GB Metal allowed it. Short-run aggregate physical footprint peaked 16.006 GB; no watchdog events. RSS alone undercounts native private Metal allocations: track proc_pid_rusage physical footprint as well as RSS for future memory gates. This native result rules out blaming low draft acceptance solely on llama.cpp for this coding workload, but does not prove target/draft mismatch or require retraining: tokenizer, target-feature parity and quantization need a differential check before porting a verifier or training. Original llama.cpp restored, context verified, arithmetic request returned42. Production remains ngram-only; no code port justified by this screen.
