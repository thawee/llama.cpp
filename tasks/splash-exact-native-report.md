# Native Splash exact-IQ2 screen

Native Splash did not solve the generation bottleneck on the exact selected weights. Three fresh coding requests reached median 5.197 tokens/sec (range 5.181-5.199), with 18 of 1890 proposed draft tokens accepted (0.95%). Every answer was identical and passed the four arithmetic-function AST checks. This is a short feasibility screen, not an interleaved engine comparison or a general ranking of Splash and llama.cpp.

| Condition | Result |
| --- | --- |
| Device | M3 Pro, 14 GPU cores, 36 GiB unified RAM |
| Target | Unchanged `models/local/qwen3.8-27b-iq2_m.gguf`, 11,673,303,648 bytes |
| Runtime | Splash 031a25a, Release C++/Metal build, local offline preparation |
| Draft | Cached incoai/Qwen3.8-27B-DFlash2 snapshot 015e795645c74b1a0eeef3b570031fb62e769bc5 |
| Draft packing | BF16 projection tensors packed into native affine 4-bit groups of 64; other tensors retained as stored |
| Execution | GPU only; int8 KV; SSD cache disabled; one request at a time; greedy; thinking off |
| Capacity | Server reports 131072 maximum context tokens |
| Actual request | 50 prompt tokens, 97 output tokens, cache miss in all three requests |
| TTFT | 1.018-1.023 seconds |
| Speculation | 6 of 630 proposed tokens accepted in each request, 90 cycles |
| Decode cycle | Approximately 205-206 ms; native GPU-command wall time is 99.8% of native cycle time |
| Sampled physical footprint | 16.006 decimal GB peak for frontend plus engine during short requests |
| Errors | Zero request errors, zero watchdog events; 3/3 AST checks passed |
| Thermal state | Nominal in captured status |

The first launch refused 128K at an 18,000,000,000-byte Metal budget: its planner allowed 77817 tokens. A 19,800,000,000-byte Metal budget admitted 128K. The Metal budget is an allocation-planning limit, not a process-memory cap. Separate monitors sampled aggregate RSS and physical footprint every 0.5 seconds and terminated the experimental process group on a 20,000,000,000-byte limit; RSS monitoring also checked critical system memory pressure. Physical-footprint monitoring was added shortly after the first request started after discovering that RSS undercounted native Metal allocations. These samples are not an OS hard cap and do not prove startup's footprint or full occupied-128K memory fits. `proc_pid_rusage` RUSAGE_INFO_V2 supplies physical footprint, verified against the local SDK structure.

The request rate excludes prefill and the first emission, as documented by Splash. All three requests reported nonzero decode work and cache misses. Native command timing establishes execution dominates the native cycle; it does not separate bandwidth, dequantization and GPU arithmetic. No profiler ran during the reported requests. Other user apps remained running. No supported public or per-request target-only switch was found, so no native speculation-off claim is made.

Long occupied-context testing was deliberately skipped: short-context generation already misses both 25 and 50 tokens/sec. The previous installed llama.cpp ngram screen measured 8.936 tokens/sec on a coding workload, but these tests are not paired, do not emit the same length or necessarily the same answer, and do not prove a general speed ratio.

The native result shows that low acceptance on this prompt is not specific to llama.cpp. It does not prove that retraining is necessary. The next discriminating experiment is a teacher-forced comparison of tokenization, selected target hidden states and draft predictions on the unchanged fused target. Check implementation and conditioning parity before attributing low acceptance to the target/draft pairing or choosing calibration/training. Porting a verifier or adding ANE prefill would not address the demonstrated rejection rate by itself.

Splash was stopped and the original launcher restored as PID 60971. Health returned `ok`; `/v1/models` succeeded; `/slots` reports context 131072; chat arithmetic returned `42`. The production launcher still uses ngram-only speculation. No production code was changed in this experiment.

Evidence: `splash-exact-native-results.json` contains command, per-request responses, status snapshots, RSS/pressure samples and restoration. `splash-exact-native-footprint.json` contains physical-footprint samples. `splash-exact-native-summary.json` contains derived rates and AST checks. `splash-exact-native-18gb-preflight.json` records the refused first launch. `splash-exact-restored-check.json` records restored API checks. Scratch orchestration scripts are `/tmp/splash-exact-screen.py` and `/tmp/splash-footprint-watch.py`; build/start logs are `/tmp/splash-build.log`, `/tmp/splash-environment.log` and `/tmp/splash-native-screen.log`. Prepared files are links under the isolated Splash checkout; original weights were not modified.
