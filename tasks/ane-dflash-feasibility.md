# ANE and DFlash2 on the local M3 Pro

The exact fused 27B IQ2_M target loads with the published Qwen3.8 DFlash2 draft, but the Q4 draft failed the speed screen. ANE runs on this Mac; the tested ANE path accelerates prefill rather than the generation step. Neither result establishes 25 generated tokens/sec, 50 tokens/sec, or full-128K retrieval quality.

The acceptance target is the same model, 131072 context, about 18 GB with up to 20 GB allowed, at least 25 generated tokens/sec and preferably above 50. Existing CLI budgets are expressed in MiB. Reports below use explicit GiB or MiB; sampled RSS does not account for every unified-memory allocation and is not an OS hard cap.

## Hardware proof

Rebuilt the existing `tools/ane_probe.mm` from the sibling Kapitsa checkout without modifying that checkout. Sources at commit `78db07839f5fb6ec3d2bb21521a88cc7447f15af`. Official source snapshots inspected: Splash `031a25a6125af2d7acf7096580afe56eb33d65bc`, Kapitsa `f9791364f895280922a42016fcc3271b726add2e`.

```sh
cmake -DOUT=/tmp/llama-ane-probe-kernels.cpp -DSRC_DIR=/Users/thawee.p/Workspaces/Labs/kapitsa/src/kernels -P /Users/thawee.p/Workspaces/Labs/kapitsa/cmake/embed_kernels.cmake
clang++ -std=c++20 -O3 -fobjc-arc -I/Users/thawee.p/Workspaces/Labs/kapitsa/src /Users/thawee.p/Workspaces/Labs/kapitsa/tools/ane_probe.mm /Users/thawee.p/Workspaces/Labs/kapitsa/src/metal/gpu.mm /tmp/llama-ane-probe-kernels.cpp -framework Foundation -framework Metal -framework IOSurface -framework IOKit -o /tmp/llama-ane-probe
/tmp/llama-ane-probe sync --units 17 --rows 512 --iters 5
/tmp/llama-ane-probe split --units 17 --rows 512 --iters 5
```

Hidden width 5120, intermediate width 17408. The probe assigns 8704 intermediate channels to ANE and uses synthetic weights and inputs. Five timed synchronous evaluations gave median 14.31 ms (13.99-14.68 ms), relative error 0.0103 against the CPU reference. First compilation was 1107 ms and load 26 ms in the split run. The CPU numerical check and split timing are separate runs.

The synthetic split completed five iterations with zero reported execution failures: GPU half alone 39.71/40.45 ms, ANE half alone 13.90 ms, combined 42.05 ms per layer. Peak child RSS was 432963584 bytes. These numbers do not compare a complete GPU FFN with a split FFN, and do not measure the local model or its decode performance. Probe packing and staging are stand-ins, not a verified IQ2_s conversion.

Kapitsa's `src/ane/ffn_split.h` targets dense FFN prefill, defaults to at least 512 rows, and stages rotated int8 weights into two alternating sets. It does not retain a full second copy of all FFN weights. For a 17-unit split, three int8 weight matrices and two staging sets alone require roughly 255 MiB; surfaces, scales, compiler and GPU work add memory. A full int8 copy of the local FFN's 17.11 billion parameters would exceed the available extra memory. The existing MLX affine-weight path is not an IQ2_s GGUF bridge. No ANE backend was added to llama.cpp or enabled in the launcher.

## Q4 DFlash2 screen

Draft: [z-lab Qwen3.8-27B-DFlash2 Q4_K_M](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/tree/main), 1143006816 bytes, SHA256 `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd`.

Target remains `models/local/qwen3.8-27b-iq2_m.gguf`. The draft reads five target-layer inputs 6/20/34/48/62. These intentionally correspond to HF output layers 5/19/33/47/61: `conversion/qwen.py` adds one when exporting. The target has 64 layers and hidden width 5120. Structural compatibility does not establish that this base-model draft predicts the modified target well.

Both contexts report 131072, batch/ubatch 64, one slot, Q8 K/V and all model layers on Metal. DFlash uses seven proposals and seven target rollback snapshots, an eight-row trained block and a 2048-token draft sliding window. The production budget gate remains closed to DFlash; this experiment uses a separate no-allocation preflight and an RSS/pressure watchdog. Target and draft contexts must be attached with `ctx_other` during measurement. Independent draft measurement failed explicitly, so the final preflight uses attached no-allocation contexts.

Combined estimated backend and listed host buffers: 18652.55 MiB (18.21 GiB). This includes worst-case target logits, feature-copy allowances and prompt tokens; metadata, temporary compilation and HTTP allocations are excluded. Real-model watchdog limits are sampled, not enforced allocation limits.

One paired screen, each with warmup and 128 fixed generation tokens:

| Configuration | Generated tokens/sec | Completion wall time | Draft accepted / proposed |
| --- | ---: | ---: | ---: |
| Retained ngram-simple N3/M3, budget 18432 MiB | 8.949 | 15.112 s | 25 / 51 |
| Q4 DFlash2, experimental budget gate off | 4.887 | 26.909 s | 16 / 756 |

Both generated the same output and passed the coding AST check; both servers exited 0. Sampled peak RSS was 18094.94 MiB (17.67 GiB), with no watchdog events. This is a screening result, not a five-pair performance estimate. The poor proposal acceptance is a measured blocker. It does not prove whether draft quantization, target modification, target quantization or runtime behavior causes it.

The mask-token row is nonzero in the target: token 248070 has 5120 nonzero values, L2 norm 0.223455141. This rules out a completely zero mask row, not distortion of the row. The [published draft](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) is trained for the base Qwen model. Splash's results do not establish this fused target's acceptance.

Reproduction: compile `tasks/dflash-preflight.cpp` against the local common/llama/ggml libraries, then run the flags recorded in `tasks/dflash-results.json`. Stop the existing server before running `python3 tasks/dflash-check.py`; the harness refuses occupied ports 8080 or 18092. It does not stop or restore the user's server itself. Run only one full-model server at a time. Raw evidence: `ane-*-results.json`, `dflash-preflight.log`, `dflash-results.json`, `dflash-mask-row.log` and the two inventory JSON files in this directory.

## Higher-precision draft check

The Q8_0 draft from the same repository is 2056414816 bytes; verified SHA256 `c18e800daedc59ca68fd13b6a856d795746af6d399a9279ac6a277d1d422f87e`. Combined no-allocation estimate: 19523.64 MiB (19.07 GiB), with the same exclusions as Q4. This experiment used the same target, context, batch, KV format, seven proposals, deterministic prompt and fixed 128 generated tokens. Its fresh baseline preceded the candidate.

| Configuration | Generated tokens/sec | Accepted / proposed |
| --- | ---: | ---: |
| Retained ngram-simple N3/M3 | 8.943 | 25 / 51 |
| Q8 DFlash2 | 4.545 | 9 / 798 |

Outputs matched exactly and passed the AST check, both server exits were 0, and no watchdog events occurred. Peak sampled RSS was 18972.23 MiB (18.53 GiB, 19.89 decimal GB). The existing experiment watchdog was 20 GiB (21.47 decimal GB); these units must not be confused with a strict 20 decimal GB cap. Full-128K residency and retrieval remain untested, so short-prompt RSS does not establish that cap at full occupancy. `vmmap` physical footprint after timing was 5.0G for baseline and 5.8G for Q8, excluding the file-backed model charge; it is not the complete runtime memory figure.

Increasing draft precision did not recover acceptance. This rules out a straightforward Q4-to-Q8 draft replacement as a useful fix for this workload. It does not prove the ultimate cause: target quantization, the fused target's changed features, or another runtime issue remain possible. A custom draft trained or adapted for this exact target would need a separate design and validation; no such draft has been trained here.

Both precision levels failed screening. No promising gain remained to confirm with five pairs. No DFlash or ANE production setting was enabled, and the 18 GiB budgeted ngram launcher was retained. The approved temporary server stop is recorded in `dflash-stopped-server.json`; restoration and API checks are recorded in `dflash-restored-server.json`. Q4 evidence is captured in its results JSON; the original scratch log names were reused by the Q8 screen. The harness now gives each result file its own scratch log prefix.

The tested path does not meet the user's generation target. A larger memory budget by itself did not fix the measured acceptance bottleneck. The next substantial investigation would need to establish a useful draft for the exact fused/quantized target before implementing an ANE drafting backend or another large integration. The retained five-pair ngram result remains 8.65 tokens/sec; the roughly 8.94 values above are fresh single-screen observations, not a new performance claim. Future harness runs use a stricter 20 decimal GB RSS watchdog; both completed screens' recorded RSS samples also fall below that threshold. The Q8 worst-case preflight estimate exceeds 20 decimal GB, so it is not a proven full-occupancy fit for that stricter limit.
