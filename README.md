<div align="center">
  <h1>Awesome Apple Silicon LLM</h1>
  <p>Repositories, docs, papers and talks for running and optimizing model inference on Apple Silicon Macs</p>
  <p><b>English</b> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ko.md">한국어</a></p>
  <p>
    <a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="Awesome"></a>
    <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
    <img src="https://img.shields.io/badge/entries-114-blue" alt="114 entries">
    <img src="https://img.shields.io/badge/links%20checked-2026--10--07-green" alt="Links checked 2026-10-07">
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-CC0%201.0-lightgrey" alt="CC0 1.0"></a>
    <a href="https://github.com/yuxuanfanOrion/awesome-apple-silicon-llm/stargazers"><img src="https://img.shields.io/github/stars/yuxuanfanOrion/awesome-apple-silicon-llm?style=social" alt="GitHub stars"></a>
  </p>
</div>

This list is about the inference stack itself: how weights move through unified memory, which runtime uses which compute unit, and how the kernels underneath are written. Sections run from the bottom up: first principles (unified memory, the bandwidth ceiling, prefill versus decode, GPU versus ANE), then the MLX family, serving, the Metal backends of cross-platform engines, the Neural Engine, quantization, kernels, multi-Mac setups, benchmarks and papers.

> [!TIP]
> New to the area? Pick a row in [Where to start](#where-to-start), or follow the [Learning path](#learning-path) from first contact to your own Metal kernel. For a complete port of a CUDA engine to MLX with a measurement at every step, read the [token-rush case study](#case-study-token-rush-on-apple-silicon).

> [!NOTE]
> Star counts are not listed. On 2026-10-07 every link was checked to resolve, and every GitHub link to be the canonical repository rather than a fork. Archived projects are left out.

### Updates

- 2026-10-08: The README now comes in English, Simplified Chinese and Korean. Added a "Where to start" table.
- 2026-10-07: Reorganized into numbered sections. New sections: quantization and speculative decoding, monitoring, speech and image. 30 new entries. The token-rush case study now covers llama.cpp and SGLang.
- 2026-10-06: First version, 84 entries.

### Contents

- [Where to start](#where-to-start)
- [Fundamentals](#fundamentals)
- [MLX family](#mlx-family)
- [Inference engines and serving](#inference-engines-and-serving)
- [Engines with a Metal backend](#engines-with-a-metal-backend)
- [Apps and on-device SDKs](#apps-and-on-device-sdks)
- [Neural Engine](#neural-engine)
- [Quantization and speculative decoding](#quantization-and-speculative-decoding)
- [Kernels and low level](#kernels-and-low-level)
- [Distributed](#distributed)
- [Speech and image](#speech-and-image)
- [Monitoring](#monitoring)
- [Benchmarks](#benchmarks)
- [Papers](#papers)
- [Talks and tutorials](#talks-and-tutorials)
- [Chinese-language resources](#chinese-language-resources)
- [Related lists](#related-lists)
- [Learning path](#learning-path)
- [Case study: token-rush on Apple Silicon](#case-study-token-rush-on-apple-silicon)

## Where to start

| You want to | Start with |
|---|---|
| Chat with a local model without writing code | [Ollama](https://github.com/ollama/ollama), [Jan](https://github.com/janhq/jan), [Llama-macOS](https://github.com/ggml-org/Llama-macOS) |
| Generate text from Python | [mlx-lm](https://github.com/ml-explore/mlx-lm) |
| Serve an OpenAI-compatible API | [`mlx_lm.server`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md), [oMLX](https://github.com/jundot/omlx) for continuous batching, [vllm-metal](https://github.com/vllm-project/vllm-metal) |
| Ship a model inside a macOS or iOS app | [MLX Swift](https://github.com/ml-explore/mlx-swift), [swift-transformers](https://github.com/huggingface/swift-transformers), [Core ML](https://developer.apple.com/documentation/coreml) |
| Run a model on the Neural Engine | [Anemll](https://github.com/Anemll/Anemll), [coremltools](https://github.com/apple/coremltools) |
| Write your own Metal kernel | [MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html), [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) |
| Run one model across several Macs | [exo](https://github.com/exo-explore/exo), [MLX distributed](https://ml-explore.github.io/mlx/build/html/usage/distributed.html) |
| Compare decode speed across M-series chips | [llama.cpp discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167) |

## Fundamentals

> [!NOTE]
> CPU, GPU and Neural Engine share one pool of unified memory, so there is no host-to-device copy and a 64 GB Mac can hold a model a 32 GB discrete card cannot. Single-stream decode reads every weight once per token, so its ceiling is roughly memory bandwidth divided by bytes read per token; that is why quantization moves tokens per second more than anything else. Prefill processes many tokens per weight read and is bound by compute instead, which is the part the M5 GPU's Neural Accelerators (matrix-multiply units inside each GPU core) speed up. The ANE is a separate fixed-function unit reached through Core ML or private APIs; it is power-efficient but constrained in layout, shapes and on-chip SRAM, so most LLM runtimes use the GPU.

1. [MLX docs: unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html): How MLX arrays live in shared memory and how a stream picks CPU or GPU without copying.
2. [MLX docs: lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html) and [compile](https://ml-explore.github.io/mlx/build/html/usage/compile.html): Why MLX builds a graph before running it and how `mx.compile` fuses ops; the closest MLX analog to capturing a CUDA graph.
3. [Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5): Apple ML Research post measuring time-to-first-token and decode on M5 versus M4, and showing which phase the new matrix units help.
4. [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks): Measured M1/M2 GPU microarchitecture: ALU latencies, cache sizes, pipelines, with comparisons to AMD and Nvidia. The best single source for reasoning about Apple GPU bottlenecks.
5. [hollance/neural-engine](https://github.com/hollance/neural-engine): Community notes on what the ANE is, which layers it supports and why a model falls back to GPU or CPU.
6. [PyTorch MPS backend notes](https://docs.pytorch.org/docs/stable/notes/mps.html): What the PyTorch Metal backend supports; useful when deciding whether PyTorch is enough or a port to MLX is needed.

<div align="right"><a href="#contents">back to top</a></div>

## MLX family

1. [ml-explore/mlx](https://github.com/ml-explore/mlx): Apple's array framework for Apple Silicon with Python, C++, C and Swift APIs, lazy evaluation and Metal kernels. The base for most Mac-native inference work.
2. [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm): LLM generation, quantization, LoRA fine-tuning and an OpenAI-compatible server on MLX. Read its model files to see how each architecture is written for MLX.
3. [mlx-lm server docs](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md): The flags and endpoints of `mlx_lm.server`, the quickest way to put an MLX model behind an HTTP API.
4. [ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples): Reference implementations (Whisper, Stable Diffusion, LLaMA and more) that are short enough to read end to end.
5. [ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift): Swift bindings for MLX, for shipping models inside macOS and iOS apps.
6. [ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm): LLM and VLM model implementations and generation in MLX Swift.
7. [ml-explore/mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples): Example apps and command-line tools built on MLX Swift.
8. [ml-explore/mlx-c](https://github.com/ml-explore/mlx-c): C API for MLX, the layer to bind from other languages.
9. [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm): Vision-language model inference and fine-tuning on MLX.
10. [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio): Text-to-speech, speech-to-text and speech-to-speech models on MLX.
11. [mflux-community/mflux](https://github.com/mflux-community/mflux): MLX-native image and video generation models such as FLUX.
12. [Goekdeniz-Guelmez/mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora): Local LLM training on MLX: SFT, DPO, ORPO and quantization-aware training on top of mlx-lm models.
13. [frost-beta/node-mlx](https://github.com/frost-beta/node-mlx): MLX bindings for Node.js.
14. [mlx-community on Hugging Face](https://huggingface.co/mlx-community): Thousands of models already converted and quantized for MLX.
15. [Hugging Face Hub docs: MLX](https://huggingface.co/docs/hub/en/mlx): How MLX models are found, downloaded and uploaded on the Hub.

<div align="right"><a href="#contents">back to top</a></div>

## Inference engines and serving

1. [jundot/omlx](https://github.com/jundot/omlx): Apple Silicon inference server with continuous batching and an SSD-backed KV cache, for serving several requests from one Mac.
2. [vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal): Community-maintained vLLM hardware plugin that runs the vLLM engine and API on MLX, with experimental paged attention. [Docs](https://docs.vllm.ai/projects/vllm-metal/en/latest/).
3. [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx): MLX-backed server with OpenAI- and Anthropic-compatible APIs and multimodal input; accompanies [arXiv 2601.19139](https://arxiv.org/abs/2601.19139).
4. [lmstudio-ai/mlx-engine](https://github.com/lmstudio-ai/mlx-engine): The MLX engine inside LM Studio; a readable example of wrapping mlx-lm for a production app.
5. [lmstudio-ai/lms](https://github.com/lmstudio-ai/lms): LM Studio's command-line tool for loading models and running its local server from scripts.
6. [cubist38/mlx-openai-server](https://github.com/cubist38/mlx-openai-server): Lightweight OpenAI-compatible server for MLX models.
7. [madroidmaq/mlx-omni-server](https://github.com/madroidmaq/mlx-omni-server): Local MLX server with OpenAI- and Anthropic-compatible APIs for chat, audio, image generation and embeddings.
8. [PicoMLX/PicoMLXServer](https://github.com/PicoMLX/PicoMLXServer): Menu bar app that starts and stops MLX servers with an OpenAI-compatible API; last updated in 2024.
9. [trymirai/uzu](https://github.com/trymirai/uzu): Rust inference engine with a Metal backend that uses unified memory on Apple devices, aimed at embedding models in apps.
10. [skyzh/tiny-llm](https://github.com/skyzh/tiny-llm): Course that builds a small LLM serving system on MLX from scratch: attention, KV cache, batching, quantized matmul. [Book](https://skyzh.github.io/tiny-llm/).

<div align="right"><a href="#contents">back to top</a></div>

## Engines with a Metal backend

1. [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp): C/C++ LLM inference with a mature Metal backend and the GGUF format; the usual reference point for Mac decode speed.
2. [ggml-org/ggml](https://github.com/ggml-org/ggml): The tensor library under llama.cpp and whisper.cpp; its Metal source is the place to read hand-written quantized kernels.
3. [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp): Whisper speech recognition on ggml, with Metal and Core ML encoder support.
4. [ollama/ollama](https://github.com/ollama/ollama): Model runner and local API built on llama.cpp; third-party write-ups report an optional MLX backend added in 2026.
5. [mozilla-ai/llamafile](https://github.com/mozilla-ai/llamafile): Single-file executables bundling llama.cpp and weights, with Metal support on macOS.
6. [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm): Compiler-based deployment engine (TVM) with a Metal target, covering Mac, iOS and the browser.
7. [EricLBuehler/mistral.rs](https://github.com/EricLBuehler/mistral.rs): Rust inference engine with a Metal backend, quantization and an OpenAI-compatible server.
8. [huggingface/candle](https://github.com/huggingface/candle): Minimal Rust ML framework with Metal kernels; good for embedding inference in Rust programs.
9. [tinygrad/tinygrad](https://github.com/tinygrad/tinygrad): Small framework with a Metal backend; its codegen is a readable example of generating Metal from a tensor graph.
10. [pytorch/executorch](https://github.com/pytorch/executorch): PyTorch's on-device runtime, with MPS and Core ML delegates for Apple hardware.
11. [mudler/LocalAI](https://github.com/mudler/LocalAI): Local API server whose backends wrap llama.cpp, MLX, whisper.cpp and others, with Apple Silicon among the supported targets.

<div align="right"><a href="#contents">back to top</a></div>

## Apps and on-device SDKs

1. [ggml-org/Llama-macOS](https://github.com/ggml-org/Llama-macOS): Menu bar app from the ggml team for running local LLMs on macOS.
2. [johnmai-dev/ChatMLX](https://github.com/johnmai-dev/ChatMLX): Open-source macOS chat app on MLX Swift.
3. [janhq/jan](https://github.com/janhq/jan): Offline desktop chat app with a llama.cpp engine that has a Metal variant for Apple Silicon.
4. [gluonfield/enchanted](https://github.com/gluonfield/enchanted): iOS and macOS chat app for self-hosted models such as an Ollama server.
5. [kevinhermawan/Ollamac](https://github.com/kevinhermawan/Ollamac): Native Mac app for Ollama; last updated in 2025.
6. [psugihara/FreeChat](https://github.com/psugihara/FreeChat): llama.cpp-based chat app for macOS; last updated in 2024.
7. [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers): Tokenizers, Hub download and generation utilities for Swift apps.
8. [huggingface/swift-chat](https://github.com/huggingface/swift-chat): Small app showing how to integrate swift-transformers into a Swift app.
9. [argmaxinc/argmax-oss-swift](https://github.com/argmaxinc/argmax-oss-swift): On-device speech recognition (WhisperKit) for Apple Silicon, using Core ML and the ANE.
10. [argmaxinc/DiffusionKit](https://github.com/argmaxinc/DiffusionKit): On-device image generation with Core ML and MLX.
11. [drawthingsai/draw-things-community](https://github.com/drawthingsai/draw-things-community): Community source of the Draw Things image-generation app, with a Swift package and `draw-things-cli` for macOS.

<div align="right"><a href="#contents">back to top</a></div>

## Neural Engine

1. [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers): Apple's explanation of the ANE-friendly rewrite: channels-first (B, C, 1, S) layout, convolutions in place of linear layers, chunked attention.
2. [apple-aiml-research/ml-ane-transformers](https://github.com/apple-aiml-research/ml-ane-transformers): Reference Transformer implementation that goes with the post above.
3. [On Device Llama 3.1 with Core ML](https://machinelearning.apple.com/research/core-ml-on-device-llama): Apple's walkthrough of converting Llama 3.1 8B to Core ML with a stateful KV cache and int4 weights.
4. [Core ML documentation](https://developer.apple.com/documentation/coreml): Apple's reference for the framework that schedules a model across CPU, GPU and ANE.
5. [apple/coremltools](https://github.com/apple/coremltools): Converts PyTorch models to Core ML and applies palettization, pruning and quantization.
6. [huggingface/exporters](https://github.com/huggingface/exporters): Exports Hugging Face models to Core ML; last updated in 2024.
7. [huggingface/coreml-examples](https://github.com/huggingface/coreml-examples): Swift examples that run Core ML models in apps.
8. [Anemll/Anemll](https://github.com/Anemll/Anemll): Pipeline that converts Hugging Face LLMs (Llama, Qwen, Gemma and others) into ANE-targeted Core ML models and runs them.
9. [smpanaro/more-ane-transformers](https://github.com/smpanaro/more-ane-transformers): Experiments running transformers, LLMs included, on the ANE; last updated in 2023.
10. [smpanaro/coreml-llm-cli](https://github.com/smpanaro/coreml-llm-cli): Command-line demo of an LLM running on the ANE through Core ML.
11. [maderix/ANE](https://github.com/maderix/ANE): Reverse-engineered private APIs to dispatch work to the ANE directly, with measurements of real fp16 throughput and its on-chip SRAM limit.
12. [mechramc/Orion](https://github.com/mechramc/Orion): Runtime that trains and runs small LLMs on the ANE without Core ML; paper [arXiv 2603.06728](https://arxiv.org/abs/2603.06728).
13. [apple-aiml-research/ml-stable-diffusion](https://github.com/apple-aiml-research/ml-stable-diffusion): Stable Diffusion on Core ML; a worked example of splitting a large model across ANE and GPU.
14. [apple-aiml-research/ml-fastvlm](https://github.com/apple-aiml-research/ml-fastvlm): Apple's efficient vision encoder for VLMs, with a demo iOS app.
15. [ANE guide](https://ane-guide.readthedocs.io/): Community documentation on programming the ANE and its constraints.

<div align="right"><a href="#contents">back to top</a></div>

## Quantization and speculative decoding

> [!NOTE]
> Decode on a Mac is bandwidth-bound, so the bits per weight set the ceiling and speculative decoding is the only way past it. Verifying several drafted tokens is not free on Apple GPUs (see the [case study](#case-study-token-rush-on-apple-silicon)), so draft depth has to be tuned per machine.

1. [MLX docs: `mx.quantize`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html): The affine group-wise format MLX uses (codes, a scale and a bias per group) and its group sizes and bit widths.
2. [mlx-lm: learned quantization](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md): DWQ, AWQ, dynamic mixed precision and GPTQ in mlx-lm, with the calibration each needs.
3. [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md): The file format llama.cpp and ollama load: metadata, tensor layout and quantization types.
4. [llama.cpp quantize tool](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md): How GGUF quantizations such as Q4_K_M are produced, with the importance-matrix option.
5. [llama.cpp speculative decoding](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md): The draft-model and draft-free speculation modes `llama-server` supports, and their flags.

<div align="right"><a href="#contents">back to top</a></div>

## Kernels and low level

1. [MLX docs: custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html): How to write a kernel body in Metal and call it from Python with `mx.fast.metal_kernel`, including grid and threadgroup setup.
2. [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf): The language reference: address spaces, SIMD-group functions, threadgroup memory, matrix types.
3. [Metal `MTLTensor`](https://developer.apple.com/documentation/metal/mtltensor): The Metal 4 tensor type that shader-side `matmul2d` operates on, the route to the M5 Neural Accelerators from your own kernel.
4. [Performing calculations on a GPU](https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu): Apple's minimal compute-pipeline sample: device, command queue, buffers, dispatch.
5. [philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention): FlashAttention ported to Metal, with notes on register pressure and SIMD-group matrix ops on Apple GPUs.
6. [dougallj/applegpu](https://github.com/dougallj/applegpu): Reverse-engineered Apple G13 GPU instruction set documentation and a disassembler.
7. [AsahiLinux/gpu](https://github.com/AsahiLinux/gpu): Asahi Linux's research notes on the M1 GPU from the driver side.
8. [manishklach/mlx-metal-kernels](https://github.com/manishklach/mlx-metal-kernels): Small experimental collection of MLX custom kernels (attention, quantized matvec, RMSNorm, RoPE, SwiGLU) to read as worked examples.
9. [BaseRT (arXiv 2607.00501)](https://arxiv.org/abs/2607.00501): Native Metal LLM runtime that issues command buffers directly with chip-specific kernel fusion; reports higher decode throughput than llama.cpp and MLX on M4 Max.

<div align="right"><a href="#contents">back to top</a></div>

## Distributed

1. [MLX docs: distributed communication](https://ml-explore.github.io/mlx/build/html/usage/distributed.html): MLX collectives over MPI, ring and JACCL (RDMA over Thunderbolt 5), and how to launch multi-Mac jobs.
2. [Explore distributed inference and training with MLX (WWDC26)](https://developer.apple.com/videos/play/wwdc2026/233/): Apple session on JACCL, RDMA over Thunderbolt 5 (macOS 26.2 and later) and sharding models across Macs.
3. [exo-explore/exo](https://github.com/exo-explore/exo): Runs one model across several Macs with topology-aware tensor parallelism on MLX distributed, including RDMA over Thunderbolt 5.

<div align="right"><a href="#contents">back to top</a></div>

## Speech and image

1. [mustafaaljadery/lightning-whisper-mlx](https://github.com/mustafaaljadery/lightning-whisper-mlx): Whisper on MLX with batched decoding and quantized models; last updated in 2024.
2. [senstella/parakeet-mlx](https://github.com/senstella/parakeet-mlx): Nvidia's Parakeet speech-recognition models ported to MLX.
3. [lucasnewman/f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx): F5-TTS text-to-speech implemented in MLX.
4. [riccardomusmeci/mlx-image](https://github.com/riccardomusmeci/mlx-image): Image classification and embedding models for MLX.

<div align="right"><a href="#contents">back to top</a></div>

## Monitoring

> [!NOTE]
> Benchmarks on a laptop drift as it heats. Watch GPU power, frequency and temperature while measuring.

1. [tlkh/asitop](https://github.com/tlkh/asitop): Terminal monitor for Apple Silicon CPU, GPU and ANE utilization and power, built on `powermetrics`; last updated in 2024.
2. [context-labs/mactop](https://github.com/context-labs/mactop): `top`-style monitor for Apple Silicon CPU, GPU, memory and power.
3. [vladkens/macmon](https://github.com/vladkens/macmon): Real-time Apple Silicon monitor that needs no sudo, with TUI, JSON and Prometheus output.

<div align="right"><a href="#contents">back to top</a></div>

## Benchmarks

1. [llama.cpp discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167): Community llama-bench results for every M-series chip at pinned commits; the most reproducible cross-chip comparison.
2. [TristanBilot/mlx-benchmark](https://github.com/TristanBilot/mlx-benchmark): Op-level timings for MLX on Apple Silicon CPU and GPU against PyTorch MPS and CUDA.
3. [john-rocky/apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench): Same-harness LLM benchmarks across Core ML, MLX and llama.cpp on iPhone and M4 Max, with quantization recorded for every number.
4. [XiongjieDai/GPU-Benchmarks-on-LLM-Inference](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference): Nvidia GPUs versus Apple Silicon for llama.cpp inference; last updated in 2024.
5. [Tom's Hardware: Mac Studio M4 Max local AI performance](https://www.tomshardware.com/desktops/exploring-apple-silicons-local-ai-performance-with-the-mac-studio-and-m4-max-m4-max-beats-gb10-and-strix-halo-in-decode-throughput-but-memory-bandwidth-isnt-everything/2): M4 Max against GB10 and Strix Halo on decode and prefill.

<div align="right"><a href="#contents">back to top</a></div>

## Papers

1. [Production-Grade Local LLM Inference on Apple Silicon (arXiv 2511.05502)](https://arxiv.org/abs/2511.05502): Compares MLX, MLC-LLM, Ollama, llama.cpp and PyTorch MPS on time-to-first-token, throughput and long context.
2. [Native LLM and MLLM Inference at Scale on Apple Silicon (arXiv 2601.19139)](https://arxiv.org/abs/2601.19139): The design of vllm-mlx: batching and multimodal serving on MLX.
3. [Orion (arXiv 2603.06728)](https://arxiv.org/abs/2603.06728): Characterizes the ANE and programs it directly for LLM training and inference.
4. [BaseRT (arXiv 2607.00501)](https://arxiv.org/abs/2607.00501): Native Metal runtime with per-chip kernel fusion.
5. [Recurrent Drafter (arXiv 2403.09919)](https://arxiv.org/abs/2403.09919): Apple's RNN draft model for speculative decoding, with results on Apple Silicon GPUs through MLX. [Blog](https://machinelearning.apple.com/research/recurrent-drafter), [code](https://github.com/apple-aiml-research/ml-recurrent-drafter).

<div align="right"><a href="#contents">back to top</a></div>

## Talks and tutorials

1. [WWDC25 298: Explore large language models on Apple silicon with MLX](https://developer.apple.com/videos/play/wwdc2025/298/): Text generation, quantization, fine-tuning and MLX Swift integration. [Written notes](https://wwdcnotes.com/documentation/wwdc25-298-explore-large-language-models-on-apple-silicon-with-mlx/).
2. [WWDC25 315: Get started with MLX for Apple silicon](https://developer.apple.com/videos/play/wwdc2025/315/): Introduction to MLX: arrays, lazy evaluation, unified memory and the Python and Swift APIs.
3. [WWDC25 262: Combine Metal 4 machine learning and graphics](https://developer.apple.com/videos/play/wwdc2025/262/): Metal 4's tensor resources and running ML inside your own Metal pipeline.
4. [WWDC24 10161: Deploy machine learning and AI models on-device with Core ML](https://developer.apple.com/videos/play/wwdc2024/10161/): Core ML's stateful models, KV caches and compression for on-device LLMs.
5. [WWDC26 233: Explore distributed inference and training with MLX](https://developer.apple.com/videos/play/wwdc2026/233/): Multi-Mac clusters with JACCL.
6. [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/): Apple's setup page for the PyTorch MPS backend.
7. [MLX vs llama.cpp on Apple Silicon](https://yage.ai/share/mlx-apple-silicon-en-20260331.html): Long-form comparison covering benchmarks, M5 Neural Accelerators and Ollama's move to MLX.
8. [tiny-llm book](https://skyzh.github.io/tiny-llm/): Week-by-week chapters for building an LLM serving stack on MLX.

<div align="right"><a href="#contents">back to top</a></div>

## Chinese-language resources

> [!NOTE]
> Few Chinese-language resources cover this area yet. These are the ones that could be verified.

1. [苹果官方发布大模型框架 MLX (DataLearner)](https://www.datalearner.com/blog/apple-mlx-machine-learning-framework-apple-silicon): Introduction written at MLX's release, covering unified memory and how its interfaces map to PyTorch and NumPy.
2. [MLX vs llama.cpp (2026)：功能对比 (Ertas AI)](https://www.ertas.ai/zh/compare/mlx-vs-llama-cpp): Feature and use-case comparison of the two engines.
3. [Apple Silicon 上的 MLX：当你需要自己的模型，而非 Apple 的模型](https://blakecrosley.com/zh-Hans/blog/mlx-on-device-ml-apple-silicon): When to run your own model with MLX instead of the system's built-in Foundation Models.

<div align="right"><a href="#contents">back to top</a></div>

## Related lists

1. [antranapp/awesome-mlx](https://github.com/antranapp/awesome-mlx): MLX projects only; this list also covers the ANE, Metal kernels, serving and other engines.

<div align="right"><a href="#contents">back to top</a></div>

## Learning path

A reading order from first contact to writing your own Metal kernel.

1. Build the mental model. Read the [Fundamentals](#fundamentals) note and the [MLX unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html) page, then compute your own Mac's ceiling: bandwidth divided by model bytes.
2. Run something. Watch [WWDC25 315](https://developer.apple.com/videos/play/wwdc2025/315/) and [298](https://developer.apple.com/videos/play/wwdc2025/298/), then generate with [mlx-lm](https://github.com/ml-explore/mlx-lm) and with [llama.cpp](https://github.com/ggml-org/llama.cpp) on the same model and compare against your ceiling.
3. Read a model implementation. Pick one architecture in mlx-lm's `models/` directory and follow a token from embedding to sampling, including the KV cache.
4. Build a small engine. Work through [tiny-llm](https://skyzh.github.io/tiny-llm/): attention, KV cache, quantized matmul, batching.
5. Learn how MLX executes. Read the [compile](https://ml-explore.github.io/mlx/build/html/usage/compile.html) docs and measure an eager decode step against a compiled one.
6. Learn the hardware. Read [metal-benchmarks](https://github.com/philipturner/metal-benchmarks) and the parts of the [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) on SIMD groups and threadgroup memory.
7. Write a kernel. Follow [custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) to write a fused op (RMSNorm or a quantized matvec), check it against the reference op, and measure achieved GB/s.
8. Study production kernels. Read the Metal sources in [mlx](https://github.com/ml-explore/mlx) and [ggml](https://github.com/ggml-org/ggml), and [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) for attention.

<div align="right"><a href="#contents">back to top</a></div>

## Case study: token-rush on Apple Silicon

[token-rush](https://github.com/zyhector/token-rush) is a single-stream inference engine for Qwen3.8-27B built for one RTX 5090, ported here to MLX on a MacBook Pro M5 Pro (64 GB). The port lives on the `mac-mlx` branch of the fork [yuxuanfanOrion/token-rush-apple-silicon](https://github.com/yuxuanfanOrion/token-rush-apple-silicon/tree/mac-mlx) and is proposed upstream as a pull request; the full write-up with every measurement is its `docs/mac.md`. The short version:

| Engine on the M5 Pro, same int4 bytes | Essay | Code | Math |
|---|---|---|---|
| token-rush, DFlash2 draft (server) | 32.0 | **49.2** | **51.1** |
| token-rush, MTP chain (server) | **34.2** | 38.4 | 38.2 |
| vllm-metal 0.31, raw (no draft runs on this model) | 16.9 | 17.0 | 16.9 |
| SGLang, MLX backend | does not run | | |
| llama.cpp b11429 and b11474, unsloth GGUF | wrong output | | |

Decode tok/s, single stream, greedy, 256 tokens, one OpenAI-compatible client for every server.

1. Measure the wall first. A streaming read gives 291 GB/s on this machine, so 13.65 GB of int4 weights per token caps raw decode at 21.3 tok/s. MLX's own int4 matmul already streams at 95-97% of that for one row, so hand-written GEMV is not where the time goes.
2. Check the format before converting. The engine's int4 g128 asymmetric packing is byte-identical to MLX's affine 4-bit layout, so the published checkpoint loads with a `view(uint32)` and keeps its GPTQ quality.
3. Use an independent implementation as the reference. mlx-lm's Qwen3.5 model on the same bytes agrees with the port to a KL of about 3e-4, two orders below the quantization's own error.
4. Raw decode reaches 18.4 tok/s in process, 86.2% of the wall, against mlx-lm's 17.2 (80.6%) on the same bytes.
5. Speculative verify is not free on Apple GPUs. MLX's int4 matmul reads the weights once per row for 2 to 15 rows, so an 8-row verify costs 2.2-3.6x a decode step (1.13x on the 5090). A Metal 4 `matmul2d` kernel on the M5 Neural Accelerators brings 8 rows to 1.45-1.85x.
6. The drafts are the whole margin over the rivals. Raw decode is a tie with vllm-metal. vllm-metal rejects MTP, DFlash2's candidate head and even ngram verification for hybrid models on Metal. SGLang's MLX backend does not start on v0.5.16-v0.5.21, and v0.5.15 sends the GDN layers to a torch backend it never initializes. llama.cpp has both drafts for this model, but on this Mac it produces wrong text for unsloth's Qwen3.8-27B GGUF on Metal and on the CPU, so its speeds (16 raw, 8-11 with drafts that accept 1.2-1.5 tokens a step) are not a valid comparison.
7. Grouped-query attention needs folding on MLX. MLX's multi-row attention kernel reads K/V once per query head; folding the 6 query heads of each KV head into the row dimension with an explicit causal mask made the 8-row verify 3.2x cheaper at 60k context. The needle is found at 8k, 32k and 60k, and DFlash2 still gives 17-35 tok/s there.
8. Check fused kernels bit for bit. `mx.compile` of the GDN gating changed fp32 results by 3.8e-6, which the recurrence turned into a 2x larger KL to the reference on one prompt; it stays uncompiled.
9. Benchmark on a laptop in one interleaved run. The same configuration measured 74 ms and later 115 ms per step as the machine heated.

<div align="right"><a href="#contents">back to top</a></div>

## Contributing

Pull requests that add resources are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the entry format and what gets accepted.

## Star history

<a href="https://star-history.com/#yuxuanfanOrion/awesome-apple-silicon-llm&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=yuxuanfanOrion/awesome-apple-silicon-llm&type=Date&theme=dark">
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=yuxuanfanOrion/awesome-apple-silicon-llm&type=Date">
  </picture>
</a>

## License

[CC0 1.0](LICENSE)
