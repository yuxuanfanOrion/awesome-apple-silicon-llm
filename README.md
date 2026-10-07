# Awesome Mac-adapted infra

A curated list of repositories, docs, papers and talks for running and optimizing model inference on Apple Silicon (M-series Macs). The focus is the inference stack itself: how weights move through unified memory, which runtime uses which compute unit, and how to write the kernels underneath.

中文导读：这份列表收录让模型在 Apple Silicon（M 系列 Mac）上高效推理的仓库、文档、论文和讲解。按层次组织：先是基本原理（统一内存、带宽上限、prefill 与 decode 的区别、GPU 与 ANE），再到 MLX 官方生态、推理服务、跨平台引擎的 Metal 后端、Neural Engine、底层 kernel、多机分布式、benchmark 和论文。想系统学习可以直接看 [Learning path](#learning-path)。

Star counts are not listed. Every GitHub link was checked to exist and to be the canonical repository (not a fork) as of 2026-10-06.

## Contents

- [Fundamentals](#fundamentals)
- [Official frameworks: MLX family](#official-frameworks-mlx-family)
- [Inference engines and serving](#inference-engines-and-serving)
- [Cross-platform engines with a Metal backend](#cross-platform-engines-with-a-metal-backend)
- [Apps and on-device SDKs](#apps-and-on-device-sdks)
- [Neural Engine (ANE)](#neural-engine-ane)
- [Kernels and low level](#kernels-and-low-level)
- [Distributed](#distributed)
- [Benchmarks](#benchmarks)
- [Papers](#papers)
- [Talks and tutorials](#talks-and-tutorials)
- [Chinese resources](#chinese-resources)
- [Related lists](#related-lists)
- [Learning path](#learning-path)
- [Case study: token-rush on Apple Silicon](#case-study-token-rush-on-apple-silicon)

## Fundamentals

The short version, before the links. CPU, GPU and Neural Engine share one pool of unified memory, so there is no host-to-device copy and a 64 GB Mac can hold a model a 32 GB discrete card cannot. Single-stream decode reads every weight once per token, so its ceiling is roughly memory bandwidth divided by bytes read per token; that is why quantization moves tokens per second more than anything else. Prefill processes many tokens per weight read and is bound by compute instead, which is the part the M5 GPU's Neural Accelerators (matrix-multiply units inside each GPU core) speed up. The ANE is a separate fixed-function unit reached through Core ML or private APIs; it is power-efficient but constrained in layout, shapes and on-chip SRAM, so most LLM runtimes use the GPU.

- [MLX docs: unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html) - How MLX arrays live in shared memory and how a stream picks CPU or GPU without copying.
- [MLX docs: lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html) and [compile](https://ml-explore.github.io/mlx/build/html/usage/compile.html) - Why MLX builds a graph before running it and how `mx.compile` fuses ops; the closest MLX analog to capturing a CUDA graph.
- [Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) - Apple ML Research post measuring time-to-first-token and decode on M5 versus M4, and showing which phase the new matrix units help.
- [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks) - Measured M1/M2 GPU microarchitecture: ALU latencies, cache sizes, pipelines, with comparisons to AMD and Nvidia. The best single source for reasoning about Apple GPU bottlenecks.
- [hollance/neural-engine](https://github.com/hollance/neural-engine) - Community notes on what the ANE is, which layers it supports and why a model falls back to GPU or CPU.
- [PyTorch MPS backend notes](https://docs.pytorch.org/docs/stable/notes/mps.html) - What the PyTorch Metal backend supports; useful when deciding whether PyTorch is enough or a port to MLX is needed.

## Official frameworks: MLX family

- [ml-explore/mlx](https://github.com/ml-explore/mlx) - Apple's array framework for Apple Silicon with Python, C++, C and Swift APIs, lazy evaluation and Metal kernels. The base for most Mac-native inference work.
- [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) - LLM generation, quantization, LoRA fine-tuning and an OpenAI-compatible server on MLX. Read its model files to see how each architecture is written for MLX.
- [ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples) - Reference implementations (Whisper, Stable Diffusion, LLaMA and more) that are short enough to read end to end.
- [ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift) - Swift bindings for MLX, for shipping models inside macOS and iOS apps.
- [ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) - LLM and VLM model implementations and generation in MLX Swift.
- [ml-explore/mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) - Example apps and command-line tools built on MLX Swift.
- [ml-explore/mlx-c](https://github.com/ml-explore/mlx-c) - C API for MLX, the layer to bind from other languages.
- [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) - Vision-language model inference and fine-tuning on MLX.
- [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) - Text-to-speech, speech-to-text and speech-to-speech models on MLX.
- [mflux-community/mflux](https://github.com/mflux-community/mflux) - MLX-native image and video generation models such as FLUX.
- [Goekdeniz-Guelmez/mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora) - Local LLM training on MLX: SFT, DPO, ORPO and quantization-aware training on top of mlx-lm models.
- [frost-beta/node-mlx](https://github.com/frost-beta/node-mlx) - MLX bindings for Node.js.
- [mlx-community on Hugging Face](https://huggingface.co/mlx-community) - Thousands of models already converted and quantized for MLX.

## Inference engines and serving

- [jundot/omlx](https://github.com/jundot/omlx) - Apple Silicon inference server with continuous batching and an SSD-backed KV cache, for serving several requests from one Mac.
- [vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal) - Community-maintained vLLM hardware plugin that runs the vLLM engine and API on MLX, with experimental paged attention. [Docs](https://docs.vllm.ai/projects/vllm-metal/en/latest/).
- [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx) - MLX-backed server with OpenAI- and Anthropic-compatible APIs and multimodal input; accompanies [arXiv 2601.19139](https://arxiv.org/abs/2601.19139).
- [lmstudio-ai/mlx-engine](https://github.com/lmstudio-ai/mlx-engine) - The MLX engine inside LM Studio; a readable example of wrapping mlx-lm for a production app.
- [lmstudio-ai/lms](https://github.com/lmstudio-ai/lms) - LM Studio's command-line tool for loading models and running its local server from scripts.
- [cubist38/mlx-openai-server](https://github.com/cubist38/mlx-openai-server) - Lightweight OpenAI-compatible server for MLX models.
- [madroidmaq/mlx-omni-server](https://github.com/madroidmaq/mlx-omni-server) - Local MLX server with OpenAI- and Anthropic-compatible APIs for chat, audio, image generation and embeddings.
- [skyzh/tiny-llm](https://github.com/skyzh/tiny-llm) - Course that builds a small LLM serving system on MLX from scratch: attention, KV cache, batching, quantized matmul. [Book](https://skyzh.github.io/tiny-llm/).

## Cross-platform engines with a Metal backend

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) - C/C++ LLM inference with a mature Metal backend and the GGUF format; the usual reference point for Mac decode speed.
- [ggml-org/ggml](https://github.com/ggml-org/ggml) - The tensor library under llama.cpp and whisper.cpp; its Metal source is the place to read hand-written quantized kernels.
- [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp) - Whisper speech recognition on ggml, with Metal and Core ML encoder support.
- [ollama/ollama](https://github.com/ollama/ollama) - Model runner and local API built on llama.cpp; third-party write-ups report an optional MLX backend added in 2026.
- [mozilla-ai/llamafile](https://github.com/mozilla-ai/llamafile) - Single-file executables bundling llama.cpp and weights, with Metal support on macOS.
- [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm) - Compiler-based deployment engine (TVM) with a Metal target, covering Mac, iOS and the browser.
- [EricLBuehler/mistral.rs](https://github.com/EricLBuehler/mistral.rs) - Rust inference engine with a Metal backend, quantization and an OpenAI-compatible server.
- [huggingface/candle](https://github.com/huggingface/candle) - Minimal Rust ML framework with Metal kernels; good for embedding inference in Rust programs.
- [tinygrad/tinygrad](https://github.com/tinygrad/tinygrad) - Small framework with a Metal backend; its codegen is a readable example of generating Metal from a tensor graph.
- [pytorch/executorch](https://github.com/pytorch/executorch) - PyTorch's on-device runtime, with MPS and Core ML delegates for Apple hardware.

## Apps and on-device SDKs

- [ggml-org/Llama-macOS](https://github.com/ggml-org/Llama-macOS) - Menu bar app from the ggml team for running local LLMs on macOS.
- [johnmai-dev/ChatMLX](https://github.com/johnmai-dev/ChatMLX) - Open-source macOS chat app on MLX Swift.
- [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers) - Tokenizers, Hub download and generation utilities for Swift apps.
- [huggingface/swift-chat](https://github.com/huggingface/swift-chat) - Small app showing how to integrate swift-transformers into a Swift app.
- [argmaxinc/argmax-oss-swift](https://github.com/argmaxinc/argmax-oss-swift) - On-device speech recognition (WhisperKit) for Apple Silicon, using Core ML and the ANE.
- [argmaxinc/DiffusionKit](https://github.com/argmaxinc/DiffusionKit) - On-device image generation with Core ML and MLX.

## Neural Engine (ANE)

- [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers) - Apple's explanation of the ANE-friendly rewrite: channels-first (B, C, 1, S) layout, convolutions in place of linear layers, chunked attention.
- [apple-aiml-research/ml-ane-transformers](https://github.com/apple-aiml-research/ml-ane-transformers) - Reference Transformer implementation that goes with the post above.
- [On Device Llama 3.1 with Core ML](https://machinelearning.apple.com/research/core-ml-on-device-llama) - Apple's walkthrough of converting Llama 3.1 8B to Core ML with a stateful KV cache and int4 weights.
- [apple/coremltools](https://github.com/apple/coremltools) - Converts PyTorch models to Core ML and applies palettization, pruning and quantization.
- [Anemll/Anemll](https://github.com/Anemll/Anemll) - Pipeline that converts Hugging Face LLMs (Llama, Qwen, Gemma and others) into ANE-targeted Core ML models and runs them.
- [maderix/ANE](https://github.com/maderix/ANE) - Reverse-engineered private APIs to dispatch work to the ANE directly, with measurements of real fp16 throughput and its on-chip SRAM limit.
- [mechramc/Orion](https://github.com/mechramc/Orion) - Runtime that trains and runs small LLMs on the ANE without Core ML; paper [arXiv 2603.06728](https://arxiv.org/abs/2603.06728).
- [apple-aiml-research/ml-stable-diffusion](https://github.com/apple-aiml-research/ml-stable-diffusion) - Stable Diffusion on Core ML; a worked example of splitting a large model across ANE and GPU.
- [apple-aiml-research/ml-fastvlm](https://github.com/apple-aiml-research/ml-fastvlm) - Apple's efficient vision encoder for VLMs, with a demo iOS app.
- [ANE guide](https://ane-guide.readthedocs.io/) - Community documentation on programming the ANE and its constraints.

## Kernels and low level

- [MLX docs: custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) - How to write a kernel body in Metal and call it from Python with `mx.fast.metal_kernel`, including grid and threadgroup setup.
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) - The language reference: address spaces, SIMD-group functions, threadgroup memory, matrix types.
- [Performing calculations on a GPU](https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu) - Apple's minimal compute-pipeline sample: device, command queue, buffers, dispatch.
- [philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention) - FlashAttention ported to Metal, with notes on register pressure and SIMD-group matrix ops on Apple GPUs.
- [dougallj/applegpu](https://github.com/dougallj/applegpu) - Reverse-engineered Apple G13 GPU instruction set documentation and a disassembler.
- [AsahiLinux/gpu](https://github.com/AsahiLinux/gpu) - Asahi Linux's research notes on the M1 GPU from the driver side.
- [manishklach/mlx-metal-kernels](https://github.com/manishklach/mlx-metal-kernels) - Small experimental collection of MLX custom kernels (attention, quantized matvec, RMSNorm, RoPE, SwiGLU) to read as worked examples.
- [BaseRT (arXiv 2607.00501)](https://arxiv.org/abs/2607.00501) - Native Metal LLM runtime that issues command buffers directly with chip-specific kernel fusion; reports higher decode throughput than llama.cpp and MLX on M4 Max.

## Distributed

- [MLX docs: distributed communication](https://ml-explore.github.io/mlx/build/html/usage/distributed.html) - MLX collectives over MPI, ring and JACCL (RDMA over Thunderbolt 5), and how to launch multi-Mac jobs.
- [Explore distributed inference and training with MLX (WWDC26)](https://developer.apple.com/videos/play/wwdc2026/233/) - Apple session on JACCL, RDMA over Thunderbolt 5 (macOS 26.2 and later) and sharding models across Macs.
- [exo-explore/exo](https://github.com/exo-explore/exo) - Runs one model across several Macs with topology-aware tensor parallelism on MLX distributed, including RDMA over Thunderbolt 5.

## Benchmarks

- [llama.cpp discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167) - Community llama-bench results for every M-series chip at pinned commits; the most reproducible cross-chip comparison.
- [TristanBilot/mlx-benchmark](https://github.com/TristanBilot/mlx-benchmark) - Op-level timings for MLX on Apple Silicon CPU and GPU against PyTorch MPS and CUDA.
- [john-rocky/apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench) - Same-harness LLM benchmarks across Core ML, MLX and llama.cpp on iPhone and M4 Max, with quantization recorded for every number.
- [XiongjieDai/GPU-Benchmarks-on-LLM-Inference](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference) - Nvidia GPUs versus Apple Silicon for llama.cpp inference; last updated in 2024.
- [Tom's Hardware: Mac Studio M4 Max local AI performance](https://www.tomshardware.com/desktops/exploring-apple-silicons-local-ai-performance-with-the-mac-studio-and-m4-max-m4-max-beats-gb10-and-strix-halo-in-decode-throughput-but-memory-bandwidth-isnt-everything/2) - M4 Max against GB10 and Strix Halo on decode and prefill.

## Papers

- [Production-Grade Local LLM Inference on Apple Silicon (arXiv 2511.05502)](https://arxiv.org/abs/2511.05502) - Compares MLX, MLC-LLM, Ollama, llama.cpp and PyTorch MPS on time-to-first-token, throughput and long context.
- [Native LLM and MLLM Inference at Scale on Apple Silicon (arXiv 2601.19139)](https://arxiv.org/abs/2601.19139) - The design of vllm-mlx: batching and multimodal serving on MLX.
- [Orion (arXiv 2603.06728)](https://arxiv.org/abs/2603.06728) - Characterizes the ANE and programs it directly for LLM training and inference.
- [BaseRT (arXiv 2607.00501)](https://arxiv.org/abs/2607.00501) - Native Metal runtime with per-chip kernel fusion.
- [Recurrent Drafter (arXiv 2403.09919)](https://arxiv.org/abs/2403.09919) - Apple's RNN draft model for speculative decoding, with results on Apple Silicon GPUs through MLX. [Blog](https://machinelearning.apple.com/research/recurrent-drafter), [code](https://github.com/apple-aiml-research/ml-recurrent-drafter).

## Talks and tutorials

- [WWDC25 298: Explore large language models on Apple silicon with MLX](https://developer.apple.com/videos/play/wwdc2025/298/) - Text generation, quantization, fine-tuning and MLX Swift integration. [Written notes](https://wwdcnotes.com/documentation/wwdc25-298-explore-large-language-models-on-apple-silicon-with-mlx/).
- [WWDC25 315: Get started with MLX for Apple silicon](https://developer.apple.com/videos/play/wwdc2025/315/) - Introduction to MLX: arrays, lazy evaluation, unified memory and the Python and Swift APIs.
- [WWDC26 233: Explore distributed inference and training with MLX](https://developer.apple.com/videos/play/wwdc2026/233/) - Multi-Mac clusters with JACCL.
- [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/) - Apple's setup page for the PyTorch MPS backend.
- [MLX vs llama.cpp on Apple Silicon](https://yage.ai/share/mlx-apple-silicon-en-20260331.html) - Long-form comparison covering benchmarks, M5 Neural Accelerators and Ollama's move to MLX.
- [tiny-llm book](https://skyzh.github.io/tiny-llm/) - Week-by-week chapters for building an LLM serving stack on MLX.

## Chinese resources

Chinese material on this topic is still thin; the list below is what could be verified.

- [苹果官方发布大模型框架 MLX（DataLearner）](https://www.datalearner.com/blog/apple-mlx-machine-learning-framework-apple-silicon) - MLX 发布时的中文介绍，讲统一内存和与 PyTorch、NumPy 的接口对应。
- [MLX vs llama.cpp (2026)：功能对比（Ertas AI）](https://www.ertas.ai/zh/compare/mlx-vs-llama-cpp) - 两个引擎的功能和适用场景对比。
- [Apple Silicon 上的 MLX：当你需要自己的模型，而非 Apple 的模型](https://blakecrosley.com/zh-Hans/blog/mlx-on-device-ml-apple-silicon) - 讲什么时候该用 MLX 自己跑模型，而不是用系统自带的 Foundation Models。

## Related lists

- [antranapp/awesome-mlx](https://github.com/antranapp/awesome-mlx) - MLX projects only; this list also covers ANE, Metal kernels, serving and other engines.

## Learning path

A reading order from first contact to writing your own Metal kernel.

1. Build the mental model. Read the [Fundamentals](#fundamentals) paragraph and the [MLX unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html) page, then compute your own Mac's ceiling: bandwidth divided by model bytes.
2. Run something. Watch [WWDC25 315](https://developer.apple.com/videos/play/wwdc2025/315/) and [298](https://developer.apple.com/videos/play/wwdc2025/298/), then generate with [mlx-lm](https://github.com/ml-explore/mlx-lm) and with [llama.cpp](https://github.com/ggml-org/llama.cpp) on the same model and compare against your ceiling.
3. Read a model implementation. Pick one architecture in mlx-lm's `models/` directory and follow a token from embedding to sampling, including the KV cache.
4. Build a small engine. Work through [tiny-llm](https://skyzh.github.io/tiny-llm/): attention, KV cache, quantized matmul, batching.
5. Learn how MLX executes. Read the [compile](https://ml-explore.github.io/mlx/build/html/usage/compile.html) docs and measure an eager decode step against a compiled one.
6. Learn the hardware. Read [metal-benchmarks](https://github.com/philipturner/metal-benchmarks) and the parts of the [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) on SIMD groups and threadgroup memory.
7. Write a kernel. Follow [custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) to write a fused op (RMSNorm or a quantized matvec), check it against the reference op, and measure achieved GB/s.
8. Study production kernels. Read the Metal sources in [mlx](https://github.com/ml-explore/mlx) and [ggml](https://github.com/ggml-org/ggml), and [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) for attention.

## Case study: token-rush on Apple Silicon

[token-rush](https://github.com/zyhector/token-rush) is a single-stream inference engine for Qwen3.8-27B built for one RTX 5090, ported here to MLX on a MacBook Pro M5 Pro (64 GB). The full write-up with every measurement is `docs/mac.md` on token-rush's [`mac-mlx` branch](https://github.com/zyhector/token-rush/tree/mac-mlx); the short version:

- **Measure the wall first.** A streaming read gives 291 GB/s on this machine, so 13.65 GB of int4 weights per token caps raw decode at 21.3 tok/s. MLX's own int4 matmul already streams at 95-97% of that for one row, so hand-written GEMV is not where the time goes.
- **Check the format before converting.** The engine's int4 g128 asymmetric packing is byte-identical to MLX's affine 4-bit layout, so the published checkpoint loads with a `view(uint32)` and keeps its GPTQ quality.
- **Use an independent implementation as the reference.** mlx-lm's Qwen3.5 model on the same bytes agrees with the port to a KL of about 3e-4, two orders below the quantization's own error.
- **Raw decode:** 18.5 tok/s, 86.7% of the wall, against mlx-lm's 17.8 on the same bytes.
- **Speculative verify is not free on Apple GPUs.** MLX's int4 matmul reads the weights once per row for 2 to 15 rows, so an 8-row verify costs 2.2-3.6x a decode step (1.13x on the 5090). A Metal 4 `matmul2d` kernel on the M5 Neural Accelerators brings 8 rows to 1.45-1.85x.
- **With the drafts:** 28-33 tok/s on essays (MTP head), 40-42 on code and 47-50 on math (DFlash2), and 61 tok/s on a code request through the OpenAI-compatible server.
- **Benchmark on a laptop in one interleaved run.** The same configuration measured 74 ms and later 115 ms per step as the machine heated.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[CC0 1.0](LICENSE)
