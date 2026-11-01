# Selective Q8 output-head candidate

Candidate: `tasks/iq3-head-experiment/qwen3.8-27b-iq3-mtp-q8-head.gguf`. Source weights and production runtime were not modified. No inference or production-server operation was run by the creator.

The publisher-verified IQ3 model has `output.weight` first, BF16 `[5120, 248320]`, followed by 865 other tensors including MTP tensors. Q8_0 would reduce this head from 2,542,796,800 to 1,350,860,800 bytes, saving 1,191,936,000 bytes. This changes target-head precision and requires quality screening; no speed or quality claim is available.

The compact model occupies 13,340,112,416 bytes. Initial preflight lacked 1,310,443,040 bytes with a 512 MiB reserve. The main task subsequently reclaimed an abandoned download with user approval. The creator then passed its disk-space check and completed without an APFS clone or source modification.

APFS whole-file cloning does not solve this layout: `ggml/src/gguf.cpp:795` requires each tensor offset to equal the accumulated aligned payload size. Since the head is first, shrinking it requires shifting all 11,978,256,384 subsequent payload bytes. Retaining a BF16-sized gap would produce an invalid GGUF. Native parser checks in `preflight.py` prove this with BF16 contiguous, Q8 contiguous and Q8 with unchanged later offsets. The last case emits an expected rejection and all three checks pass.

`COPY` is also unsuitable for selective quantization: `src/llama-quant.cpp:304` disables quantization before considering tensor overrides. `create_candidate.py` instead preserves all header bytes except the first tensor type and subsequent tensor offsets, quantizes only the first tensor using exported `ggml_bf16_to_fp32_row` and `ggml_quantize_chunk` in 128-row CPU chunks, then streams and hashes the unchanged tail. It refuses existing output files and retains incomplete outputs with `.partial` suffix. It requires enough space for the complete compact file plus 512 MiB.

Validation in `candidate.json`:

- Complete source SHA256 matches publisher: `19ee0a94937a3b028f7f2565f9b210c2a15f32738b237ea466b0a96cb61717b0`.
- Candidate SHA256: `f87099a29462d9270314f720e137aa477f21c0d4075ccf9b77cfdfc14a874e53`.
- All 865 other tensor payloads, including MTP, are covered by an unchanged-tail readback comparison: 11,978,256,384 bytes, SHA256 `3aaa4000d3bfb1800c6f258ebbe7ca545f11ad679e2ccda707b3439acf84b916`.
- All tensor descriptors match expected names, shapes, types and shifted offsets. Native GGUF parser accepts the complete candidate.
- Synthetic self-test checks Q8 reconstruction error below 0.01, unchanged source, unchanged norm/MTP payloads, destination overwrite refusal and native GGUF parser behavior. Run `PYTHONDONTWRITEBYTECODE=1 python3 tasks/iq3-head-experiment/create_candidate.py --self-test`.

Only target output-head precision changed. Model quality and throughput must be screened by the main task. The creator completed while the main task was beginning GPU screening; main was warned to discard any overlapping measurement because file hashing can contend for memory bandwidth. No further bulk I/O is performed by this helper.
