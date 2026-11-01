# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Asanmai%20OR%20author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Athelittlefireman%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3AmasterFoad%20OR%20author%3Aallozaur%20OR%20author%3AAndreas-Krebbel%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Achaxu01%20OR%20author%3Alhez%20OR%20author%3AAlekseiNikiforovIBM%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming%20OR%20author%3Aynankani)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Local M3 Pro project: goals and results

This private fork targets local, private general tasks and coding assistance on an Apple M3 Pro with 36 GB unified memory. Production uses the selected DavidAU Qwen 27B IQ3_M MTP model with only `output.weight` requantized to Q8_0. The original IQ2_M model remains available for rollback.

The active profile is 131072 context tokens, native MTP depth 5, Q8_0 target/draft KV caches, batch and microbatch 64, five CPU threads and full Metal offload. It serves `qwen3.8-27b-local` at `http://127.0.0.1:8000/v1` through `llama-local`, Pi and Copilot. Short tests sampled 21.127 GB peak RSS; this is not a memory cap. Full occupied-128K performance remains unverified.

| Goal | Verified result |
| --- | --- |
| Keep the selected 27B model | IQ3_M weights with a Q8_0 output head run through Metal. All other 865 tensors were preserved; the original IQ2_M remains available. |
| 128K context | Server exposes 131072 tokens in one slot. Full-128K occupied-context quality, speed, and peak memory remain unverified. |
| About 18-20 GB memory; up to 22 GB for experiments | CPU-thread trials sampled 17.102 GB peak RSS; the experimental IQ3 Q8-head 128K screen peaked at 21.127 GB. The default launcher uses an 18432 MiB inference estimate budget, which is not a process memory cap. |
| At least 25 generated tokens/sec; preferably above 50 | Not achieved. Best IQ3 Q8-head short code screen reached 17.56 tokens/sec. The earlier IQ2 coding baseline was a median 9.336 across five 128-token trials; these are different workloads. |
| Faster generation without a separate draft model | Native MTP depth 5 is enabled using the model's embedded draft head. The older IQ2 profile used bounded ngram speculation. |
| Safe performance experiments | The tested Metal integer-sign optimization was 3.23% slower and was reverted. No experimental kernel or profiler is installed. |
| Earlier IQ2 native MTP experiment | Depth 5 at 128K with Q8 draft cache: 8.643-11.424 tokens/sec, 19.920 GB peak sampled RSS. Short prompts only; no demonstrated speed improvement. See the [MTP screen](tasks/mtp-experiment/report.md). |
| IQ3 comparison at 64K | Native MTP depth 5 measured 7.30-14.17 tokens/sec, 20.238 GB peak sampled RSS. Saved Kapitsa was faster on all four matched workloads. See the [IQ3 comparison](tasks/mtp-experiment/report-iq3-kapitsa.md). |
| Production IQ3 with Q8 output head | Head-only requantization reached 9.51 tokens/sec prose and 17.56 code at configured 128K, with 21.127 GB peak sampled RSS. A fixed 128-token general-thinking screen reached 15.66 tokens/sec, including reasoning and truncating before the final answer. Single trials; full occupied 128K and general quality remain unverified. See the [optimization screen](tasks/mtp-experiment/report-fast-optimization.md). |
| M3 CPU threads | Five performance cores plus six efficiency cores. Five-thread default retained: eleven generation threads were 1.3% slower; eleven batch threads changed prefill by only 0.1%. See the [CPU thread experiment](tasks/cpu-core-report.md). |
| Local coding clients | Installed commands start one shared server on demand. Earlier IQ2 Pi and GitHub Copilot CLI file-read checks passed at about 17.11 GB peak RSS. The promoted IQ3 API and completed-response smoke check passed; client alias and endpoint are unchanged. |

The [22 GB longer-prompt screen](tasks/mtp-experiment/report-22gb-long.md) measured 8.76 tokens/sec at 3018 input tokens and 7.87 at 12118, with 19.879 GB peak sampled RSS and normal memory pressure. Full occupied 128K remains unverified.

ANE acceleration is not enabled. The native Splash screen on these exact weights did not meet the throughput target; it does not establish a general ranking of engines. See the [memory-budget and runtime measurements](docs/memory-budget.md), [Metal experiment](tasks/metal-decode-optimization.md), and [exact-model Splash screen](tasks/splash-exact-native-report.md) for methods and limitations.

The default serves general thinking at `http://127.0.0.1:8000/v1`, with temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 0 and repetition penalty 1.0. The coding/thread measurements above used the earlier non-thinking configuration; the MTP screen used general thinking.

For installation under `~/.local/bin`, on-demand startup, Pi, and GitHub Copilot CLI, see [local installation](docs/local-install.md).

After installing:

```sh
llama-pi             # Start or reuse the local server, then open Pi
llama-copilot        # Start or reuse it, then open Copilot offline
llama-local status   # Check the shared server
llama-local stop     # Release its memory when finished
```

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
