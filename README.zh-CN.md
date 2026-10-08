<div align="center">
  <h1>Awesome Apple Silicon LLM</h1>
  <p>在 Apple Silicon（M 系列 Mac）上运行和优化模型推理的仓库、文档、论文与演讲</p>
  <p><a href="README.md">English</a> · <b>简体中文</b> · <a href="README.ko.md">한국어</a></p>
  <p>
    <a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="Awesome"></a>
    <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
    <img src="https://img.shields.io/badge/entries-114-blue" alt="114 entries">
    <img src="https://img.shields.io/badge/links%20checked-2026--10--07-green" alt="Links checked 2026-10-07">
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-CC0%201.0-lightgrey" alt="CC0 1.0"></a>
    <a href="https://github.com/yuxuanfanOrion/awesome-apple-silicon-llm/stargazers"><img src="https://img.shields.io/github/stars/yuxuanfanOrion/awesome-apple-silicon-llm?style=social" alt="GitHub stars"></a>
  </p>
</div>

这份列表关注推理栈本身：权重怎样在统一内存里流动，哪个运行时用哪块计算单元，底层 kernel 怎么写。章节从下往上排：先是基本原理（统一内存、带宽上限、prefill 与 decode 的区别、GPU 与 ANE），再到 MLX 官方生态、推理服务、跨平台引擎的 Metal 后端、神经引擎、量化、底层 kernel、多机分布式、评测和论文。

> [!TIP]
> 第一次接触这个方向，先在 [从哪里开始](#从哪里开始) 里找一行对应你需求的，或者按 [学习路线](#学习路线) 从上手一路读到自己写 Metal kernel。想看一个完整的移植案例（CUDA 引擎搬到 MLX，每一步都有测量），看 [token-rush 案例](#案例-token-rush-移植到-apple-silicon)。

> [!NOTE]
> 不列 star 数。2026-10-07 逐个检查过：每个链接都能打开，每个 GitHub 链接都指向原始仓库而不是 fork。已归档（archived）的项目不收录。

### 更新

- 2026-10-08：README 拆成英文、简体中文、韩文三个版本，新增“从哪里开始”速查表。
- 2026-10-07：按编号重新整理章节。新增量化与投机解码、监控、语音与图像三个章节，共 30 条新资源。token-rush 案例补上了 llama.cpp 和 SGLang 的对比。
- 2026-10-06：第一版，84 条。

### 目录

- [从哪里开始](#从哪里开始)
- [基础原理](#基础原理)
- [MLX 生态](#mlx-生态)
- [推理服务](#推理服务)
- [跨平台引擎的 Metal 后端](#跨平台引擎的-metal-后端)
- [应用与端侧 SDK](#应用与端侧-sdk)
- [神经引擎](#神经引擎)
- [量化与投机解码](#量化与投机解码)
- [底层 kernel](#底层-kernel)
- [多机分布式](#多机分布式)
- [语音与图像](#语音与图像)
- [监控](#监控)
- [评测](#评测)
- [论文](#论文)
- [演讲与教程](#演讲与教程)
- [中文资源](#中文资源)
- [相关列表](#相关列表)
- [学习路线](#学习路线)
- [案例 token-rush 移植到 Apple Silicon](#案例-token-rush-移植到-apple-silicon)

## 从哪里开始

| 你想做的事 | 从这里开始 |
|---|---|
| 不写代码，直接和本地模型对话 | [Ollama](https://github.com/ollama/ollama)、[Jan](https://github.com/janhq/jan)、[Llama-macOS](https://github.com/ggml-org/Llama-macOS) |
| 在 Python 里生成文本 | [mlx-lm](https://github.com/ml-explore/mlx-lm) |
| 提供 OpenAI 兼容的 API | [`mlx_lm.server`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md)、[oMLX](https://github.com/jundot/omlx)（continuous batching）、[vllm-metal](https://github.com/vllm-project/vllm-metal) |
| 把模型打包进 macOS 或 iOS 应用 | [MLX Swift](https://github.com/ml-explore/mlx-swift)、[swift-transformers](https://github.com/huggingface/swift-transformers)、[Core ML](https://developer.apple.com/documentation/coreml) |
| 在神经引擎上跑模型 | [Anemll](https://github.com/Anemll/Anemll)、[coremltools](https://github.com/apple/coremltools) |
| 自己写 Metal kernel | [MLX 自定义 Metal kernel](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)、[metal-flash-attention](https://github.com/philipturner/metal-flash-attention) |
| 用几台 Mac 跑同一个模型 | [exo](https://github.com/exo-explore/exo)、[MLX distributed](https://ml-explore.github.io/mlx/build/html/usage/distributed.html) |
| 横向比较各款 M 系列芯片的 decode 速度 | [llama.cpp discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167) |

## 基础原理

> [!NOTE]
> CPU、GPU 和神经引擎共用一块统一内存，没有 host 到 device 的拷贝，所以 64 GB 的 Mac 能装下 32 GB 独立显卡装不下的模型。单流 decode 每生成一个 token 都要把全部权重读一遍，上限大约是内存带宽除以每个 token 读取的字节数，这也是为什么量化对 tokens/s 的影响比其他任何手段都大。prefill 每读一次权重处理很多 token，瓶颈在算力，M5 GPU 的 Neural Accelerators（每个 GPU 核心里的矩阵乘单元）加速的正是这一段。ANE 是独立的固定功能单元，要通过 Core ML 或私有 API 调用；它省电，但对数据布局、shape 和片上 SRAM 限制很多，所以大多数 LLM 运行时用的是 GPU。

1. [MLX 文档：统一内存](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)：MLX 数组怎样放在共享内存里，stream 怎样不经拷贝就选 CPU 或 GPU。
2. [MLX 文档：惰性求值](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html) 和 [compile](https://ml-explore.github.io/mlx/build/html/usage/compile.html)：MLX 为什么先建图再执行，`mx.compile` 怎样融合算子；在 MLX 里最接近 CUDA graph capture 的东西。
3. [Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)：Apple ML Research 的文章，在 M5 和 M4 上测首 token 延迟和 decode，说明新的矩阵单元加速的是哪个阶段。
4. [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks)：实测的 M1/M2 GPU 微架构：ALU 延迟、缓存大小、流水线，并和 AMD、Nvidia 对比。分析 Apple GPU 瓶颈时最值得看的单一资料。
5. [hollance/neural-engine](https://github.com/hollance/neural-engine)：社区整理的 ANE 笔记：它是什么，支持哪些层，模型为什么会退回 GPU 或 CPU。
6. [PyTorch MPS 后端说明](https://docs.pytorch.org/docs/stable/notes/mps.html)：PyTorch 的 Metal 后端支持什么；判断 PyTorch 够不够用、要不要移植到 MLX 时有用。

<div align="right"><a href="#目录">回到顶部</a></div>

## MLX 生态

1. [ml-explore/mlx](https://github.com/ml-explore/mlx)：Apple 为 Apple Silicon 做的数组框架，有 Python、C++、C 和 Swift API，惰性求值，底层是 Metal kernel。大多数 Mac 原生推理工作的基础。
2. [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)：基于 MLX 的 LLM 生成、量化、LoRA 微调和 OpenAI 兼容服务。读它的模型文件能看到每种架构在 MLX 里怎么写。
3. [mlx-lm server 文档](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md)：`mlx_lm.server` 的参数和接口，把 MLX 模型挂到 HTTP API 后面最快的办法。
4. [ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples)：参考实现（Whisper、Stable Diffusion、LLaMA 等），代码短到能从头读到尾。
5. [ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift)：MLX 的 Swift 绑定，用来把模型放进 macOS 和 iOS 应用。
6. [ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)：MLX Swift 里的 LLM、VLM 模型实现和生成。
7. [ml-explore/mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples)：基于 MLX Swift 的示例应用和命令行工具。
8. [ml-explore/mlx-c](https://github.com/ml-explore/mlx-c)：MLX 的 C API，其他语言做绑定时接这一层。
9. [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm)：MLX 上的视觉语言模型推理与微调。
10. [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio)：MLX 上的语音合成、语音识别和语音到语音模型。
11. [mflux-community/mflux](https://github.com/mflux-community/mflux)：MLX 原生的图像和视频生成模型，例如 FLUX。
12. [Goekdeniz-Guelmez/mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora)：在 MLX 上本地训练 LLM：基于 mlx-lm 模型做 SFT、DPO、ORPO 和量化感知训练。
13. [frost-beta/node-mlx](https://github.com/frost-beta/node-mlx)：MLX 的 Node.js 绑定。
14. [Hugging Face 上的 mlx-community](https://huggingface.co/mlx-community)：几千个已经转换并量化好的 MLX 模型。
15. [Hugging Face Hub 文档：MLX](https://huggingface.co/docs/hub/en/mlx)：在 Hub 上查找、下载和上传 MLX 模型的方法。

<div align="right"><a href="#目录">回到顶部</a></div>

## 推理服务

1. [jundot/omlx](https://github.com/jundot/omlx)：Apple Silicon 推理服务，支持 continuous batching 和落盘到 SSD 的 KV cache，一台 Mac 同时服务多个请求。
2. [vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal)：社区维护的 vLLM 硬件插件，让 vLLM 的引擎和 API 跑在 MLX 上，带实验性的 paged attention。[文档](https://docs.vllm.ai/projects/vllm-metal/en/latest/)。
3. [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx)：以 MLX 为后端的服务，提供 OpenAI 和 Anthropic 兼容 API，支持多模态输入；对应论文 [arXiv 2601.19139](https://arxiv.org/abs/2601.19139)。
4. [lmstudio-ai/mlx-engine](https://github.com/lmstudio-ai/mlx-engine)：LM Studio 内置的 MLX 引擎；看生产级应用怎样包装 mlx-lm 的好例子。
5. [lmstudio-ai/lms](https://github.com/lmstudio-ai/lms)：LM Studio 的命令行工具，可以在脚本里加载模型、启动本地服务。
6. [cubist38/mlx-openai-server](https://github.com/cubist38/mlx-openai-server)：轻量的 OpenAI 兼容 MLX 模型服务。
7. [madroidmaq/mlx-omni-server](https://github.com/madroidmaq/mlx-omni-server)：本地 MLX 服务，提供 OpenAI 和 Anthropic 兼容 API，覆盖对话、音频、图像生成和 embedding。
8. [PicoMLX/PicoMLXServer](https://github.com/PicoMLX/PicoMLXServer)：菜单栏应用，用来启停带 OpenAI 兼容 API 的 MLX 服务；最后更新于 2024 年。
9. [trymirai/uzu](https://github.com/trymirai/uzu)：Rust 写的推理引擎，Metal 后端利用 Apple 设备的统一内存，面向把模型嵌入应用的场景。
10. [skyzh/tiny-llm](https://github.com/skyzh/tiny-llm)：从零在 MLX 上搭一个小型 LLM 服务系统的课程：attention、KV cache、batching、量化 matmul。[电子书](https://skyzh.github.io/tiny-llm/)。

<div align="right"><a href="#目录">回到顶部</a></div>

## 跨平台引擎的 Metal 后端

1. [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)：C/C++ 写的 LLM 推理，Metal 后端成熟，使用 GGUF 格式；衡量 Mac decode 速度的常用参照。
2. [ggml-org/ggml](https://github.com/ggml-org/ggml)：llama.cpp 和 whisper.cpp 底下的张量库；想读手写的量化 kernel，就看它的 Metal 源码。
3. [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp)：基于 ggml 的 Whisper 语音识别，encoder 支持 Metal 和 Core ML。
4. [ollama/ollama](https://github.com/ollama/ollama)：基于 llama.cpp 的模型运行器和本地 API；据第三方文章，2026 年加入了可选的 MLX 后端。
5. [mozilla-ai/llamafile](https://github.com/mozilla-ai/llamafile)：把 llama.cpp 和权重打包成单个可执行文件，macOS 上支持 Metal。
6. [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm)：基于编译器（TVM）的部署引擎，有 Metal 目标，覆盖 Mac、iOS 和浏览器。
7. [EricLBuehler/mistral.rs](https://github.com/EricLBuehler/mistral.rs)：Rust 推理引擎，有 Metal 后端、量化和 OpenAI 兼容服务。
8. [huggingface/candle](https://github.com/huggingface/candle)：极简的 Rust ML 框架，带 Metal kernel；适合在 Rust 程序里嵌入推理。
9. [tinygrad/tinygrad](https://github.com/tinygrad/tinygrad)：带 Metal 后端的小型框架；它的 codegen 是从张量图生成 Metal 代码的易读范例。
10. [pytorch/executorch](https://github.com/pytorch/executorch)：PyTorch 的端侧运行时，针对 Apple 硬件有 MPS 和 Core ML delegate。
11. [mudler/LocalAI](https://github.com/mudler/LocalAI)：本地 API 服务，后端封装了 llama.cpp、MLX、whisper.cpp 等，支持 Apple Silicon。

<div align="right"><a href="#目录">回到顶部</a></div>

## 应用与端侧 SDK

1. [ggml-org/Llama-macOS](https://github.com/ggml-org/Llama-macOS)：ggml 团队做的菜单栏应用，在 macOS 上跑本地 LLM。
2. [johnmai-dev/ChatMLX](https://github.com/johnmai-dev/ChatMLX)：基于 MLX Swift 的开源 macOS 聊天应用。
3. [janhq/jan](https://github.com/janhq/jan)：离线桌面聊天应用，llama.cpp 引擎有面向 Apple Silicon 的 Metal 版本。
4. [gluonfield/enchanted](https://github.com/gluonfield/enchanted)：iOS 和 macOS 聊天应用，连接自托管模型，例如 Ollama 服务。
5. [kevinhermawan/Ollamac](https://github.com/kevinhermawan/Ollamac)：Ollama 的原生 Mac 客户端；最后更新于 2025 年。
6. [psugihara/FreeChat](https://github.com/psugihara/FreeChat)：基于 llama.cpp 的 macOS 聊天应用；最后更新于 2024 年。
7. [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers)：给 Swift 应用用的 tokenizer、Hub 下载和生成工具。
8. [huggingface/swift-chat](https://github.com/huggingface/swift-chat)：演示怎样把 swift-transformers 接进 Swift 应用的小程序。
9. [argmaxinc/argmax-oss-swift](https://github.com/argmaxinc/argmax-oss-swift)：Apple Silicon 上的端侧语音识别（WhisperKit），使用 Core ML 和 ANE。
10. [argmaxinc/DiffusionKit](https://github.com/argmaxinc/DiffusionKit)：用 Core ML 和 MLX 做端侧图像生成。
11. [drawthingsai/draw-things-community](https://github.com/drawthingsai/draw-things-community)：图像生成应用 Draw Things 的社区版源码，包含 Swift package 和 macOS 上的 `draw-things-cli`。

<div align="right"><a href="#目录">回到顶部</a></div>

## 神经引擎

1. [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)：Apple 讲怎样把模型改写成适合 ANE 的形式：channels-first 的 (B, C, 1, S) 布局，用卷积代替线性层，分块 attention。
2. [apple-aiml-research/ml-ane-transformers](https://github.com/apple-aiml-research/ml-ane-transformers)：配合上面那篇文章的 Transformer 参考实现。
3. [On Device Llama 3.1 with Core ML](https://machinelearning.apple.com/research/core-ml-on-device-llama)：Apple 演示把 Llama 3.1 8B 转成 Core ML，带有状态的 KV cache 和 int4 权重。
4. [Core ML 文档](https://developer.apple.com/documentation/coreml)：Core ML 框架的官方参考，这个框架负责把模型调度到 CPU、GPU 和 ANE 上。
5. [apple/coremltools](https://github.com/apple/coremltools)：把 PyTorch 模型转成 Core ML，并做调色板化（palettization）、剪枝和量化。
6. [huggingface/exporters](https://github.com/huggingface/exporters)：把 Hugging Face 模型导出为 Core ML；最后更新于 2024 年。
7. [huggingface/coreml-examples](https://github.com/huggingface/coreml-examples)：在应用里运行 Core ML 模型的 Swift 示例。
8. [Anemll/Anemll](https://github.com/Anemll/Anemll)：把 Hugging Face 上的 LLM（Llama、Qwen、Gemma 等）转换成面向 ANE 的 Core ML 模型并运行的工具链。
9. [smpanaro/more-ane-transformers](https://github.com/smpanaro/more-ane-transformers)：在 ANE 上跑 transformer（包括 LLM）的实验；最后更新于 2023 年。
10. [smpanaro/coreml-llm-cli](https://github.com/smpanaro/coreml-llm-cli)：命令行演示，通过 Core ML 在 ANE 上运行 LLM。
11. [maderix/ANE](https://github.com/maderix/ANE)：逆向出的私有 API，直接把计算派发给 ANE，并实测了真实的 fp16 吞吐和片上 SRAM 上限。
12. [mechramc/Orion](https://github.com/mechramc/Orion)：不经过 Core ML、直接在 ANE 上训练和运行小型 LLM 的运行时；论文 [arXiv 2603.06728](https://arxiv.org/abs/2603.06728)。
13. [apple-aiml-research/ml-stable-diffusion](https://github.com/apple-aiml-research/ml-stable-diffusion)：Core ML 上的 Stable Diffusion；把大模型拆到 ANE 和 GPU 上运行的完整例子。
14. [apple-aiml-research/ml-fastvlm](https://github.com/apple-aiml-research/ml-fastvlm)：Apple 为 VLM 做的高效视觉编码器，附 iOS 演示应用。
15. [ANE guide](https://ane-guide.readthedocs.io/)：社区写的 ANE 编程文档及其限制。

<div align="right"><a href="#目录">回到顶部</a></div>

## 量化与投机解码

> [!NOTE]
> Mac 上的 decode 受带宽限制，每个权重占几 bit 决定了上限，投机解码是越过这个上限的唯一办法。在 Apple GPU 上一次验证多个草稿 token 并不免费（见 [案例](#案例-token-rush-移植到-apple-silicon)），所以草稿深度要针对每台机器调。

1. [MLX 文档：`mx.quantize`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)：MLX 使用的仿射分组量化格式（每组一份编码、一个 scale 和一个 bias），以及可选的组大小和位宽。
2. [mlx-lm：学习型量化](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md)：mlx-lm 里的 DWQ、AWQ、动态混合精度和 GPTQ，以及各自需要的校准。
3. [GGUF 规范](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)：llama.cpp 和 ollama 加载的文件格式：元数据、张量布局和量化类型。
4. [llama.cpp quantize 工具](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)：Q4_K_M 这类 GGUF 量化是怎么生成的，以及 importance matrix 选项。
5. [llama.cpp 投机解码](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)：`llama-server` 支持的草稿模型和无草稿模型两类投机模式，以及对应参数。

<div align="right"><a href="#目录">回到顶部</a></div>

## 底层 kernel

1. [MLX 文档：自定义 Metal kernel](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)：怎样用 Metal 写 kernel 主体、用 `mx.fast.metal_kernel` 从 Python 调用，包括 grid 和 threadgroup 的设置。
2. [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)：语言参考：地址空间、SIMD-group 函数、threadgroup memory、矩阵类型。
3. [Metal `MTLTensor`](https://developer.apple.com/documentation/metal/mtltensor)：Metal 4 的张量类型，shader 里的 `matmul2d` 作用在它上面，是从自己的 kernel 调用 M5 Neural Accelerators 的途径。
4. [Performing calculations on a GPU](https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu)：Apple 最小的计算管线示例：device、command queue、buffer、dispatch。
5. [philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention)：移植到 Metal 的 FlashAttention，附有关于 Apple GPU 寄存器压力和 SIMD-group 矩阵运算的笔记。
6. [dougallj/applegpu](https://github.com/dougallj/applegpu)：逆向得到的 Apple G13 GPU 指令集文档和反汇编器。
7. [AsahiLinux/gpu](https://github.com/AsahiLinux/gpu)：Asahi Linux 从驱动角度研究 M1 GPU 的笔记。
8. [manishklach/mlx-metal-kernels](https://github.com/manishklach/mlx-metal-kernels)：一小组实验性的 MLX 自定义 kernel（attention、量化 matvec、RMSNorm、RoPE、SwiGLU），可当作例题来读。
9. [BaseRT (arXiv 2607.00501)](https://arxiv.org/abs/2607.00501)：原生 Metal LLM 运行时，直接提交 command buffer，按芯片做 kernel 融合；报告在 M4 Max 上 decode 吞吐高于 llama.cpp 和 MLX。

<div align="right"><a href="#目录">回到顶部</a></div>

## 多机分布式

1. [MLX 文档：分布式通信](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)：MLX 在 MPI、ring 和 JACCL（基于 Thunderbolt 5 的 RDMA）上的集合通信，以及怎样启动多台 Mac 的任务。
2. [Explore distributed inference and training with MLX (WWDC26)](https://developer.apple.com/videos/play/wwdc2026/233/)：Apple 的讲座，介绍 JACCL、Thunderbolt 5 上的 RDMA（macOS 26.2 及以上）和跨 Mac 切分模型。
3. [exo-explore/exo](https://github.com/exo-explore/exo)：用几台 Mac 跑同一个模型，基于 MLX distributed 做拓扑感知的张量并行，支持 Thunderbolt 5 上的 RDMA。

<div align="right"><a href="#目录">回到顶部</a></div>

## 语音与图像

1. [mustafaaljadery/lightning-whisper-mlx](https://github.com/mustafaaljadery/lightning-whisper-mlx)：MLX 上的 Whisper，支持批量解码和量化模型；最后更新于 2024 年。
2. [senstella/parakeet-mlx](https://github.com/senstella/parakeet-mlx)：移植到 MLX 的 Nvidia Parakeet 语音识别模型。
3. [lucasnewman/f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx)：用 MLX 实现的 F5-TTS 语音合成。
4. [riccardomusmeci/mlx-image](https://github.com/riccardomusmeci/mlx-image)：MLX 上的图像分类和 embedding 模型。

<div align="right"><a href="#目录">回到顶部</a></div>

## 监控

> [!NOTE]
> 笔记本发热后跑分会漂。测量时盯着 GPU 功耗、频率和温度。

1. [tlkh/asitop](https://github.com/tlkh/asitop)：基于 `powermetrics` 的终端监控，显示 Apple Silicon CPU、GPU、ANE 的占用和功耗；最后更新于 2024 年。
2. [context-labs/mactop](https://github.com/context-labs/mactop)：`top` 风格的 Apple Silicon 监控，看 CPU、GPU、内存和功耗。
3. [vladkens/macmon](https://github.com/vladkens/macmon)：不需要 sudo 的 Apple Silicon 实时监控，支持 TUI、JSON 和 Prometheus 输出。

<div align="right"><a href="#目录">回到顶部</a></div>

## 评测

1. [llama.cpp discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167)：社区在固定 commit 上用 llama-bench 测的各款 M 系列芯片结果；跨芯片对比里可复现性最好的一份。
2. [TristanBilot/mlx-benchmark](https://github.com/TristanBilot/mlx-benchmark)：MLX 在 Apple Silicon CPU 和 GPU 上的算子级耗时，与 PyTorch MPS 和 CUDA 对比。
3. [john-rocky/apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench)：同一套测试框架下 Core ML、MLX、llama.cpp 在 iPhone 和 M4 Max 上的 LLM 评测，每个数字都注明了量化方式。
4. [XiongjieDai/GPU-Benchmarks-on-LLM-Inference](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference)：Nvidia GPU 与 Apple Silicon 的 llama.cpp 推理对比；最后更新于 2024 年。
5. [Tom's Hardware：Mac Studio M4 Max 本地 AI 性能](https://www.tomshardware.com/desktops/exploring-apple-silicons-local-ai-performance-with-the-mac-studio-and-m4-max-m4-max-beats-gb10-and-strix-halo-in-decode-throughput-but-memory-bandwidth-isnt-everything/2)：M4 Max 与 GB10、Strix Halo 在 decode 和 prefill 上的对比。

<div align="right"><a href="#目录">回到顶部</a></div>

## 论文

1. [Production-Grade Local LLM Inference on Apple Silicon (arXiv 2511.05502)](https://arxiv.org/abs/2511.05502)：比较 MLX、MLC-LLM、Ollama、llama.cpp 和 PyTorch MPS 的首 token 延迟、吞吐和长上下文表现。
2. [Native LLM and MLLM Inference at Scale on Apple Silicon (arXiv 2601.19139)](https://arxiv.org/abs/2601.19139)：vllm-mlx 的设计：MLX 上的 batching 和多模态服务。
3. [Orion (arXiv 2603.06728)](https://arxiv.org/abs/2603.06728)：刻画 ANE 的特性，并直接对它编程来做 LLM 训练和推理。
4. [BaseRT (arXiv 2607.00501)](https://arxiv.org/abs/2607.00501)：按芯片做 kernel 融合的原生 Metal 运行时。
5. [Recurrent Drafter (arXiv 2403.09919)](https://arxiv.org/abs/2403.09919)：Apple 用 RNN 做投机解码的草稿模型，附有通过 MLX 在 Apple Silicon GPU 上的结果。[博客](https://machinelearning.apple.com/research/recurrent-drafter)，[代码](https://github.com/apple-aiml-research/ml-recurrent-drafter)。

<div align="right"><a href="#目录">回到顶部</a></div>

## 演讲与教程

1. [WWDC25 298: Explore large language models on Apple silicon with MLX](https://developer.apple.com/videos/play/wwdc2025/298/)：文本生成、量化、微调和 MLX Swift 集成。[文字笔记](https://wwdcnotes.com/documentation/wwdc25-298-explore-large-language-models-on-apple-silicon-with-mlx/)。
2. [WWDC25 315: Get started with MLX for Apple silicon](https://developer.apple.com/videos/play/wwdc2025/315/)：MLX 入门：数组、惰性求值、统一内存，以及 Python 和 Swift API。
3. [WWDC25 262: Combine Metal 4 machine learning and graphics](https://developer.apple.com/videos/play/wwdc2025/262/)：Metal 4 的张量资源，以及在自己的 Metal 管线里跑 ML。
4. [WWDC24 10161: Deploy machine learning and AI models on-device with Core ML](https://developer.apple.com/videos/play/wwdc2024/10161/)：Core ML 的有状态模型、KV cache 和压缩，用于端侧 LLM。
5. [WWDC26 233: Explore distributed inference and training with MLX](https://developer.apple.com/videos/play/wwdc2026/233/)：用 JACCL 组多台 Mac 的集群。
6. [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/)：Apple 的 PyTorch MPS 后端安装说明页。
7. [MLX vs llama.cpp on Apple Silicon](https://yage.ai/share/mlx-apple-silicon-en-20260331.html)：长篇对比，涵盖跑分、M5 Neural Accelerators 和 Ollama 转向 MLX。
8. [tiny-llm 电子书](https://skyzh.github.io/tiny-llm/)：按周分章，在 MLX 上搭一套 LLM 服务栈。

<div align="right"><a href="#目录">回到顶部</a></div>

## 中文资源

> [!NOTE]
> 这个方向的中文资料还不多，下面是能核实的部分。

1. [苹果官方发布大模型框架 MLX（DataLearner）](https://www.datalearner.com/blog/apple-mlx-machine-learning-framework-apple-silicon)：MLX 发布时的中文介绍，讲统一内存和与 PyTorch、NumPy 的接口对应。
2. [MLX vs llama.cpp (2026)：功能对比（Ertas AI）](https://www.ertas.ai/zh/compare/mlx-vs-llama-cpp)：两个引擎的功能和适用场景对比。
3. [Apple Silicon 上的 MLX：当你需要自己的模型，而非 Apple 的模型](https://blakecrosley.com/zh-Hans/blog/mlx-on-device-ml-apple-silicon)：讲什么时候该用 MLX 自己跑模型，而不是用系统自带的 Foundation Models。

<div align="right"><a href="#目录">回到顶部</a></div>

## 相关列表

1. [antranapp/awesome-mlx](https://github.com/antranapp/awesome-mlx)：只收 MLX 项目；本列表还覆盖 ANE、Metal kernel、推理服务和其他引擎。

<div align="right"><a href="#目录">回到顶部</a></div>

## 学习路线

从第一次上手到自己写 Metal kernel 的阅读顺序。

1. 建立直觉。读 [基础原理](#基础原理) 里的说明和 [MLX 统一内存](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html) 文档，然后算出自己这台 Mac 的上限：带宽除以模型字节数。
2. 先跑起来。看 [WWDC25 315](https://developer.apple.com/videos/play/wwdc2025/315/) 和 [298](https://developer.apple.com/videos/play/wwdc2025/298/)，再用 [mlx-lm](https://github.com/ml-explore/mlx-lm) 和 [llama.cpp](https://github.com/ggml-org/llama.cpp) 跑同一个模型，和你算出的上限对比。
3. 读一个模型实现。在 mlx-lm 的 `models/` 目录里挑一个架构，跟着一个 token 从 embedding 走到 sampling，KV cache 也要看。
4. 搭一个小引擎。跟着 [tiny-llm](https://skyzh.github.io/tiny-llm/) 做一遍：attention、KV cache、量化 matmul、batching。
5. 弄清 MLX 怎么执行。读 [compile](https://ml-explore.github.io/mlx/build/html/usage/compile.html) 文档，测一下 eager 的 decode step 和 compile 后的差多少。
6. 了解硬件。读 [metal-benchmarks](https://github.com/philipturner/metal-benchmarks)，以及 [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) 中关于 SIMD group 和 threadgroup memory 的部分。
7. 写一个 kernel。照着 [自定义 Metal kernel](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) 写一个融合算子（RMSNorm 或量化 matvec），和参考算子对结果，再测实际达到的 GB/s。
8. 研究生产级 kernel。读 [mlx](https://github.com/ml-explore/mlx) 和 [ggml](https://github.com/ggml-org/ggml) 里的 Metal 源码，attention 部分看 [metal-flash-attention](https://github.com/philipturner/metal-flash-attention)。

<div align="right"><a href="#目录">回到顶部</a></div>

## 案例 token-rush 移植到 Apple Silicon

[token-rush](https://github.com/zyhector/token-rush) 是一个为单张 RTX 5090 写的 Qwen3.8-27B 单流推理引擎，这里把它移植到了 MacBook Pro M5 Pro（64 GB）上的 MLX。移植代码在 fork [yuxuanfanOrion/token-rush-apple-silicon](https://github.com/yuxuanfanOrion/token-rush-apple-silicon/tree/mac-mlx) 的 `mac-mlx` 分支，已经向上游提了 pull request；带全部测量数据的完整记录在它的 `docs/mac.md`。简要结论如下：

| M5 Pro 上的引擎，相同的 int4 权重 | 写作 | 代码 | 数学 |
|---|---|---|---|
| token-rush，DFlash2 草稿（服务模式） | 32.0 | **49.2** | **51.1** |
| token-rush，MTP 链（服务模式） | **34.2** | 38.4 | 38.2 |
| vllm-metal 0.31，无草稿（这个模型上跑不了任何草稿） | 16.9 | 17.0 | 16.9 |
| SGLang，MLX 后端 | 无法运行 | | |
| llama.cpp b11429 和 b11474，unsloth GGUF | 输出错误 | | |

单位是 decode tok/s，单流，greedy，256 个 token，所有服务都用同一个 OpenAI 兼容客户端测。

1. 先测上限。这台机器流式读取带宽是 291 GB/s，每个 token 要读 13.65 GB 的 int4 权重，所以无草稿 decode 最多 21.3 tok/s。MLX 自带的 int4 matmul 在单行时已经达到这个带宽的 95-97%，手写 GEMV 省不出时间。
2. 转换前先核对格式。引擎的 int4 g128 非对称打包和 MLX 的仿射 4-bit 布局逐字节一致，所以发布的 checkpoint 用一个 `view(uint32)` 就能加载，GPTQ 的精度原样保留。
3. 用独立实现做参照。mlx-lm 的 Qwen3.5 模型在同一份权重上和移植版的 KL 约为 3e-4，比量化本身的误差低两个数量级。
4. 无草稿 decode 在进程内达到 18.4 tok/s，是上限的 86.2%；mlx-lm 在同一份权重上是 17.2（80.6%）。
5. 在 Apple GPU 上，投机解码的验证并不免费。MLX 的 int4 matmul 在 2 到 15 行时每一行都要把权重读一遍，所以验证 8 行的开销是一个 decode step 的 2.2-3.6 倍（5090 上是 1.13 倍）。用 Metal 4 的 `matmul2d` kernel 跑在 M5 Neural Accelerators 上，8 行降到 1.45-1.85 倍。
6. 领先对手的全部差距都来自草稿。无草稿 decode 和 vllm-metal 打平。vllm-metal 在 Metal 上对混合架构模型拒绝 MTP、DFlash2 的候选头，连 ngram 验证都不支持。SGLang 的 MLX 后端在 v0.5.16-v0.5.21 上启动不了，v0.5.15 则把 GDN 层交给一个从未初始化的 torch 后端。llama.cpp 对这个模型两种草稿都支持，但在这台 Mac 上，unsloth 的 Qwen3.8-27B GGUF 无论走 Metal 还是 CPU 都输出错误文本，所以它的速度（无草稿 16，开草稿 8-11，每步接受 1.2-1.5 个 token）不能拿来比较。
7. 在 MLX 上，grouped-query attention 需要折叠。MLX 的多行 attention kernel 每个 query head 都要读一遍 K/V；把每个 KV head 对应的 6 个 query head 折叠进行维度、再配上显式的 causal mask，60k 上下文下验证 8 行的开销降到原来的 1/3.2。8k、32k、60k 下都能找到 needle，DFlash2 在这些长度下仍有 17-35 tok/s。
8. 融合 kernel 要逐位核对。对 GDN gating 做 `mx.compile` 让 fp32 结果偏差 3.8e-6，经过递推放大后，在一个 prompt 上与参照的 KL 变成原来的 2 倍；所以这部分保持不编译。
9. 在笔记本上跑分，要在一次交错运行里完成。同一配置随着机器发热，先测出每步 74 ms，后来变成 115 ms。

<div align="right"><a href="#目录">回到顶部</a></div>

## 贡献

欢迎提交 PR 补充资源。收录标准和条目格式见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## Star 历史

<a href="https://star-history.com/#yuxuanfanOrion/awesome-apple-silicon-llm&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=yuxuanfanOrion/awesome-apple-silicon-llm&type=Date&theme=dark">
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=yuxuanfanOrion/awesome-apple-silicon-llm&type=Date">
  </picture>
</a>

## 许可

[CC0 1.0](LICENSE)
