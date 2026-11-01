# Exact-target DFlash conditioning diagnostic

The tested failure starts in draft generation before verification. No tokenizer or target feature alignment fix was identified in llama.cpp. The original BF16 draft also misses the first expected continuation token when driven by captures from either engine. Keep the verified ngram launcher as the production default.

Scope: exact local Qwen 27B IQ2_M target, one fixed 50-token coding prompt, 16 teacher-forced continuation tokens. llama.cpp diagnostics configured 131072 context with Q8 KV. Native diagnostic tools used a small test window. These results do not measure generation at an occupied 128K context or demonstrate 25/50 tokens per second.

## Tokenization and target conditioning

Eleven raw text cases cover code, English, numbers, Thai, Chinese, Unicode, whitespace and special tokens. Every tested token sequence matches. Both engines render the benchmark chat prompt identically with reasoning disabled, producing the same 50 token IDs. A tokenizer library warning is not evidence of a mismatch in these tested inputs.

HF output layers [5,19,33,47,61] correctly correspond to llama.cpp layer inputs [6,20,34,48,62]. The offset reflects capture placement rather than an off-by-one error: llama.cpp captures the input of the following layer. Source tracing also checked ordering, normalization and draft position handling.

| HF output layer | llama input layer | Mean feature cosine | Relative RMSE |
| --- | --- | --- | --- |
| 5 | 6 | 0.9999969 | 0.00475 |
| 19 | 20 | 0.9999788 | 0.00885 |
| 33 | 34 | 0.9999543 | 0.00885 |
| 47 | 48 | 0.9999220 | 0.01213 |
| 61 | 62 | 0.9998477 | 0.01598 |

All 17 compared target greedy next-token predictions agree. Mean logit cosine is 0.999943 and mean native-to-llama KL divergence is 0.00001476. Numerics are close, not bit-identical; draft outputs remain sensitive to those differences.

## Proposals before verification

Both engines agree on anchor token 71093. The expected first speculative token is 12305, `python`, beginning `python\ndef add(a, b`. Native affine-4 draft generation instead begins with token 67; llama.cpp Q8_0 draft generation begins with token 310. Both proposals already fail at the first token. Changing verification cannot correctly accept a proposal that disagrees with the target.

To check whether draft quantization alone explains this, run the released MLX reference DFlash2 implementation at commit 07ebd93db9f472af339b644bb70221ad8428328a with the original BF16 draft safetensors. Use exact GGUF target embedding rows, the exact BF16 target output head, and each engine's saved conditioning features. Both reference runs propose token 310 (` to`) first, also incorrect. The unary first-token prediction is incorrect as well, so the candidate selector alone does not explain this prefix's failure.

This rules out a failure confined to llama.cpp and draft quantization as the sole cause on this prefix. It does not rule out a shared configuration issue, establish general draft incompatibility, or prove retraining is necessary.

## Memory, restoration and evidence

All five diagnostic processes exited zero without watchdog events. Sampled aggregate process-group RSS and physical footprint remained below 20 decimal GB. Monitoring both matters: native Metal allocations are poorly represented by RSS, while mapped llama.cpp weights can be poorly represented by physical footprint. The highest observed llama draft diagnostic RSS was 18.679 GB; native teacher physical footprint reached 14.931 GB. The BF16 reference peaked at 7.392 GB physical footprint. Sampling cannot establish an absolute instantaneous allocation cap.

The local server was restored with the existing 128K launcher. Health, model discovery, slot context 131072 and a chat response of 42 passed. No production source changes were made in this diagnostic pass.

Evidence files:

- `tasks/dflash-tokenizer-parity.json`: raw text and rendered chat token comparison.
- `tasks/dflash-numeric-parity-summary.json`: feature and target-logit comparison.
- `tasks/dflash-first-block-summary.json`: native and llama draft proposals.
- `tasks/dflash-bf16-reference-results.json`: original BF16 reference proposals.
- `tasks/dflash-numeric-parity-run.json`, `tasks/dflash-first-block-run.json`, `tasks/dflash-bf16-reference-run.json`: actual commands, process samples, exits and restoration.
- `tasks/dflash-parity-restored-check.json`: restored API checks.

The next useful experiment is to establish a known-good original-BF16 reference fixture for this draft and its intended base target, then compare the exact fused IQ2 target against it. That separates shared setup errors from target/draft mismatch before spending effort on draft adaptation. Porting a verifier or adding ANE execution has no demonstrated path to fix the observed bad proposals. The requested 25/50 tokens per second remains unachieved.
