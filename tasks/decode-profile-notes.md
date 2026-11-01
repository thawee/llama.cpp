# Decode profile

An actual 128-token generation was sampled for five seconds with macOS sample. This profiled run is excluded from throughput comparisons. The dominant decode-worker samples wait for GPU completion. CPU sampling cannot separate GPU bandwidth from GPU arithmetic or dequantization.

```text
    +         !   : | 4078 ggml_backend_sched_synchronize  (in libggml-base.0.dylib) + 84  [0x10112866c]
    +         !   : |   4078 ggml_metal_synchronize  (in libggml-metal.0.dylib) + 40  [0x1035ce1d4]
    +         !   : |     4078 -[_MTLCommandBuffer waitUntilCompleted]  (in Metal) + 76  [0x195bd193c]
    +         !   :   + 54 llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&)  (in libllama.0.dylib) + 516  [0x102dd6004]
```

Metal initialization reported fusion, concurrency and graph optimization enabled. All model layers are offloaded. The model contains 408 IQ2_s tensors, 64 Q4_K tensors, 25 IQ3_s tensors and one BF16 tensor, plus F32 parameters. IQ2_M is the container quantization label, not a standalone Metal kernel type.
