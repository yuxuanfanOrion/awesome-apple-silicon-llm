<div align="center">
  <h1>Awesome Apple Silicon LLM</h1>
  <p>Apple Silicon(M 시리즈 Mac)에서 모델 추론을 실행하고 최적화하기 위한 저장소, 문서, 논문, 강연 모음</p>
  <p><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <b>한국어</b></p>
  <p>
    <a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="Awesome"></a>
    <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
    <img src="https://img.shields.io/badge/entries-114-blue" alt="114 entries">
    <img src="https://img.shields.io/badge/links%20checked-2026--10--07-green" alt="Links checked 2026-10-07">
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-CC0%201.0-lightgrey" alt="CC0 1.0"></a>
    <a href="https://github.com/yuxuanfanOrion/awesome-apple-silicon-llm/stargazers"><img src="https://img.shields.io/github/stars/yuxuanfanOrion/awesome-apple-silicon-llm?style=social" alt="GitHub stars"></a>
  </p>
</div>

이 목록은 추론 스택 자체를 다룹니다. 가중치가 통합 메모리(unified memory) 안에서 어떻게 이동하는지, 어떤 런타임이 어떤 연산 장치를 쓰는지, 그 아래의 kernel은 어떻게 작성하는지가 관심사입니다. 섹션은 아래층부터 위로 쌓입니다. 먼저 기본 원리(통합 메모리, 대역폭 상한, prefill과 decode의 차이, GPU와 ANE)를 다루고, 이어서 MLX 생태계, 추론 서버, 크로스플랫폼 엔진의 Metal 백엔드, Neural Engine, 양자화, kernel, 여러 대의 Mac을 묶는 분산 구성, 벤치마크와 논문 순서로 이어집니다.

> [!TIP]
> 이 분야가 처음이라면 [시작하기](#시작하기) 표에서 자신의 목적에 맞는 행을 고르거나, [학습 경로](#학습-경로)를 따라 첫 실행부터 직접 Metal kernel을 작성하는 단계까지 읽어 보세요. CUDA 엔진을 MLX로 옮기면서 단계마다 측정한 완전한 이식 사례는 [token-rush 사례 연구](#사례-연구-token-rush의-apple-silicon-이식)에 있습니다.

> [!NOTE]
> star 수는 싣지 않습니다. 2026-10-07에 모든 링크가 열리는지, 모든 GitHub 링크가 fork가 아닌 원본 저장소인지 확인했습니다. archived 상태의 프로젝트는 제외합니다.

### 업데이트

- 2026-10-08: README를 영어, 중국어 간체, 한국어 세 가지 버전으로 나누고 "시작하기" 표를 추가했습니다.
- 2026-10-07: 번호를 붙인 섹션으로 재구성했습니다. 양자화와 추측 디코딩, 모니터링, 음성과 이미지 섹션을 새로 만들고 30개 항목을 추가했습니다. token-rush 사례 연구에 llama.cpp와 SGLang 비교를 넣었습니다.
- 2026-10-06: 첫 버전, 84개 항목.

### 목차

- [시작하기](#시작하기)
- [기본 원리](#기본-원리)
- [MLX 생태계](#mlx-생태계)
- [추론 서버](#추론-서버)
- [Metal 백엔드 엔진](#metal-백엔드-엔진)
- [앱과 온디바이스 SDK](#앱과-온디바이스-sdk)
- [뉴럴 엔진](#뉴럴-엔진)
- [양자화와 추측 디코딩](#양자화와-추측-디코딩)
- [커널과 저수준](#커널과-저수준)
- [분산 추론](#분산-추론)
- [음성과 이미지](#음성과-이미지)
- [모니터링](#모니터링)
- [벤치마크](#벤치마크)
- [논문](#논문)
- [강연과 튜토리얼](#강연과-튜토리얼)
- [중국어 자료](#중국어-자료)
- [관련 목록](#관련-목록)
- [학습 경로](#학습-경로)
- [사례 연구 token-rush의 Apple Silicon 이식](#사례-연구-token-rush의-apple-silicon-이식)

## 시작하기

| 하고 싶은 일 | 시작점 |
|---|---|
| 코드 없이 로컬 모델과 대화하기 | [Ollama](https://github.com/ollama/ollama), [Jan](https://github.com/janhq/jan), [Llama-macOS](https://github.com/ggml-org/Llama-macOS) |
| Python에서 텍스트 생성하기 | [mlx-lm](https://github.com/ml-explore/mlx-lm) |
| OpenAI 호환 API 서버 띄우기 | [`mlx_lm.server`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md), [oMLX](https://github.com/jundot/omlx)(continuous batching), [vllm-metal](https://github.com/vllm-project/vllm-metal) |
| macOS나 iOS 앱에 모델 넣기 | [MLX Swift](https://github.com/ml-explore/mlx-swift), [swift-transformers](https://github.com/huggingface/swift-transformers), [Core ML](https://developer.apple.com/documentation/coreml) |
| Neural Engine에서 모델 실행하기 | [Anemll](https://github.com/Anemll/Anemll), [coremltools](https://github.com/apple/coremltools) |
| 직접 Metal kernel 작성하기 | [MLX 커스텀 Metal kernel](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html), [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) |
| 여러 대의 Mac에서 하나의 모델 실행하기 | [exo](https://github.com/exo-explore/exo), [MLX distributed](https://ml-explore.github.io/mlx/build/html/usage/distributed.html) |
| M 시리즈 칩별 decode 속도 비교하기 | [llama.cpp discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167) |

## 기본 원리

> [!NOTE]
> CPU, GPU, Neural Engine은 하나의 통합 메모리를 공유합니다. host에서 device로 복사할 필요가 없고, 64 GB Mac은 32 GB 외장 그래픽카드에 들어가지 않는 모델도 올릴 수 있습니다. 단일 스트림 decode는 토큰 하나마다 모든 가중치를 한 번씩 읽으므로, 속도 상한은 대략 메모리 대역폭을 토큰당 읽는 바이트 수로 나눈 값입니다. 양자화가 초당 토큰 수를 가장 크게 바꾸는 이유가 여기에 있습니다. prefill은 가중치를 한 번 읽을 때 여러 토큰을 처리하므로 연산량이 병목이 되고, M5 GPU의 Neural Accelerators(각 GPU 코어 안의 행렬곱 유닛)가 빠르게 해 주는 부분이 바로 이 단계입니다. ANE는 Core ML이나 비공개 API로만 접근하는 별도의 고정 기능 유닛입니다. 전력 효율은 좋지만 데이터 레이아웃, shape, 온칩 SRAM 제약이 많아서 대부분의 LLM 런타임은 GPU를 씁니다.

1. [MLX 문서: 통합 메모리](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html): MLX 배열이 공유 메모리에 어떻게 놓이는지, stream이 복사 없이 CPU나 GPU를 어떻게 고르는지 설명합니다.
2. [MLX 문서: 지연 평가](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)와 [compile](https://ml-explore.github.io/mlx/build/html/usage/compile.html): MLX가 실행 전에 그래프를 먼저 만드는 이유와 `mx.compile`이 연산을 융합하는 방식을 다룹니다. MLX에서 CUDA graph capture에 가장 가까운 기능입니다.
3. [Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5): M5와 M4에서 첫 토큰까지의 시간과 decode를 측정하고, 새 행렬 유닛이 어느 단계를 빠르게 하는지 보여 주는 Apple ML Research 글입니다.
4. [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks): M1/M2 GPU 마이크로아키텍처 실측 자료입니다. ALU 지연 시간, 캐시 크기, 파이프라인을 AMD, Nvidia와 비교합니다. Apple GPU 병목을 분석할 때 가장 먼저 볼 자료입니다.
5. [hollance/neural-engine](https://github.com/hollance/neural-engine): ANE가 무엇인지, 어떤 레이어를 지원하는지, 모델이 왜 GPU나 CPU로 되돌아가는지 정리한 커뮤니티 노트입니다.
6. [PyTorch MPS 백엔드 노트](https://docs.pytorch.org/docs/stable/notes/mps.html): PyTorch의 Metal 백엔드가 무엇을 지원하는지 정리한 문서입니다. PyTorch로 충분한지, MLX로 이식해야 하는지 판단할 때 유용합니다.

<div align="right"><a href="#목차">맨 위로</a></div>

## MLX 생태계

1. [ml-explore/mlx](https://github.com/ml-explore/mlx): Apple이 Apple Silicon용으로 만든 배열 프레임워크입니다. Python, C++, C, Swift API와 지연 평가를 제공하고 Metal kernel 위에서 동작합니다. Mac 네이티브 추론 작업 대부분의 기반입니다.
2. [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm): MLX 기반의 LLM 생성, 양자화, LoRA 파인튜닝, OpenAI 호환 서버입니다. 모델 파일을 읽으면 각 아키텍처가 MLX로 어떻게 작성되는지 볼 수 있습니다.
3. [mlx-lm server 문서](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md): `mlx_lm.server`의 옵션과 엔드포인트입니다. MLX 모델을 HTTP API 뒤에 두는 가장 빠른 방법입니다.
4. [ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples): 처음부터 끝까지 읽을 수 있을 만큼 짧은 참조 구현 모음입니다(Whisper, Stable Diffusion, LLaMA 등).
5. [ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift): MLX의 Swift 바인딩으로, macOS와 iOS 앱에 모델을 넣을 때 씁니다.
6. [ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm): MLX Swift로 작성한 LLM, VLM 모델 구현과 생성 코드입니다.
7. [ml-explore/mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples): MLX Swift로 만든 예제 앱과 명령줄 도구입니다.
8. [ml-explore/mlx-c](https://github.com/ml-explore/mlx-c): MLX의 C API로, 다른 언어에서 바인딩할 때 연결하는 계층입니다.
9. [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm): MLX에서 비전-언어 모델을 추론하고 파인튜닝합니다.
10. [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio): MLX용 TTS, STT, speech-to-speech 모델입니다.
11. [mflux-community/mflux](https://github.com/mflux-community/mflux): FLUX 같은 MLX 네이티브 이미지, 비디오 생성 모델입니다.
12. [Goekdeniz-Guelmez/mlx-lm-lora](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora): MLX에서 로컬로 LLM을 학습합니다. mlx-lm 모델 위에서 SFT, DPO, ORPO, 양자화 인식 학습을 지원합니다.
13. [frost-beta/node-mlx](https://github.com/frost-beta/node-mlx): MLX의 Node.js 바인딩입니다.
14. [Hugging Face의 mlx-community](https://huggingface.co/mlx-community): MLX용으로 이미 변환하고 양자화한 모델 수천 개가 있습니다.
15. [Hugging Face Hub 문서: MLX](https://huggingface.co/docs/hub/en/mlx): Hub에서 MLX 모델을 찾고, 내려받고, 올리는 방법입니다.

<div align="right"><a href="#목차">맨 위로</a></div>

## 추론 서버

1. [jundot/omlx](https://github.com/jundot/omlx): continuous batching과 SSD 기반 KV cache를 갖춘 Apple Silicon 추론 서버로, Mac 한 대에서 여러 요청을 처리할 때 씁니다.
2. [vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal): vLLM 엔진과 API를 MLX 위에서 돌리는 커뮤니티 관리 vLLM 하드웨어 플러그인으로, 실험적인 paged attention을 포함합니다. [문서](https://docs.vllm.ai/projects/vllm-metal/en/latest/).
3. [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx): OpenAI, Anthropic 호환 API와 멀티모달 입력을 지원하는 MLX 기반 서버입니다. 논문 [arXiv 2601.19139](https://arxiv.org/abs/2601.19139)와 짝을 이룹니다.
4. [lmstudio-ai/mlx-engine](https://github.com/lmstudio-ai/mlx-engine): LM Studio에 들어 있는 MLX 엔진입니다. 실제 제품에서 mlx-lm을 감싸는 방법을 읽기 쉽게 보여 줍니다.
5. [lmstudio-ai/lms](https://github.com/lmstudio-ai/lms): 스크립트에서 모델을 로드하고 로컬 서버를 실행하는 LM Studio의 명령줄 도구입니다.
6. [cubist38/mlx-openai-server](https://github.com/cubist38/mlx-openai-server): MLX 모델용 경량 OpenAI 호환 서버입니다.
7. [madroidmaq/mlx-omni-server](https://github.com/madroidmaq/mlx-omni-server): 채팅, 오디오, 이미지 생성, 임베딩을 OpenAI, Anthropic 호환 API로 제공하는 로컬 MLX 서버입니다.
8. [PicoMLX/PicoMLXServer](https://github.com/PicoMLX/PicoMLXServer): OpenAI 호환 API를 가진 MLX 서버를 켜고 끄는 메뉴 막대 앱입니다. 마지막 업데이트는 2024년입니다.
9. [trymirai/uzu](https://github.com/trymirai/uzu): Apple 기기의 통합 메모리를 활용하는 Metal 백엔드를 갖춘 Rust 추론 엔진으로, 앱에 모델을 내장하는 용도입니다.
10. [skyzh/tiny-llm](https://github.com/skyzh/tiny-llm): MLX 위에서 작은 LLM 서빙 시스템을 처음부터 만드는 강좌입니다. attention, KV cache, batching, 양자화 matmul을 다룹니다. [책](https://skyzh.github.io/tiny-llm/).

<div align="right"><a href="#목차">맨 위로</a></div>

## Metal 백엔드 엔진

1. [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp): 성숙한 Metal 백엔드와 GGUF 포맷을 갖춘 C/C++ LLM 추론 엔진입니다. Mac decode 속도를 잴 때 흔히 기준으로 삼습니다.
2. [ggml-org/ggml](https://github.com/ggml-org/ggml): llama.cpp와 whisper.cpp 아래의 텐서 라이브러리입니다. 손으로 작성한 양자화 kernel을 읽으려면 이 저장소의 Metal 소스를 보면 됩니다.
3. [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp): ggml 기반 Whisper 음성 인식으로, 인코더에서 Metal과 Core ML을 지원합니다.
4. [ollama/ollama](https://github.com/ollama/ollama): llama.cpp 기반의 모델 실행기이자 로컬 API입니다. 서드파티 글에 따르면 2026년에 선택형 MLX 백엔드가 추가되었습니다.
5. [mozilla-ai/llamafile](https://github.com/mozilla-ai/llamafile): llama.cpp와 가중치를 실행 파일 하나로 묶습니다. macOS에서 Metal을 지원합니다.
6. [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm): 컴파일러(TVM) 기반 배포 엔진으로, Metal 타깃을 통해 Mac, iOS, 브라우저를 지원합니다.
7. [EricLBuehler/mistral.rs](https://github.com/EricLBuehler/mistral.rs): Metal 백엔드, 양자화, OpenAI 호환 서버를 갖춘 Rust 추론 엔진입니다.
8. [huggingface/candle](https://github.com/huggingface/candle): Metal kernel을 포함한 최소한의 Rust ML 프레임워크로, Rust 프로그램에 추론을 내장할 때 좋습니다.
9. [tinygrad/tinygrad](https://github.com/tinygrad/tinygrad): Metal 백엔드를 가진 작은 프레임워크입니다. 텐서 그래프에서 Metal 코드를 생성하는 codegen이 읽기 쉬운 예제가 됩니다.
10. [pytorch/executorch](https://github.com/pytorch/executorch): PyTorch의 온디바이스 런타임으로, Apple 하드웨어용 MPS, Core ML delegate가 있습니다.
11. [mudler/LocalAI](https://github.com/mudler/LocalAI): llama.cpp, MLX, whisper.cpp 등을 백엔드로 감싼 로컬 API 서버로, Apple Silicon을 지원합니다.

<div align="right"><a href="#목차">맨 위로</a></div>

## 앱과 온디바이스 SDK

1. [ggml-org/Llama-macOS](https://github.com/ggml-org/Llama-macOS): ggml 팀이 만든, macOS에서 로컬 LLM을 실행하는 메뉴 막대 앱입니다.
2. [johnmai-dev/ChatMLX](https://github.com/johnmai-dev/ChatMLX): MLX Swift 기반의 오픈소스 macOS 채팅 앱입니다.
3. [janhq/jan](https://github.com/janhq/jan): 오프라인 데스크톱 채팅 앱으로, llama.cpp 엔진에 Apple Silicon용 Metal 버전이 있습니다.
4. [gluonfield/enchanted](https://github.com/gluonfield/enchanted): Ollama 서버 같은 자체 호스팅 모델에 연결하는 iOS, macOS 채팅 앱입니다.
5. [kevinhermawan/Ollamac](https://github.com/kevinhermawan/Ollamac): Ollama용 네이티브 Mac 앱입니다. 마지막 업데이트는 2025년입니다.
6. [psugihara/FreeChat](https://github.com/psugihara/FreeChat): llama.cpp 기반 macOS 채팅 앱입니다. 마지막 업데이트는 2024년입니다.
7. [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers): Swift 앱용 tokenizer, Hub 다운로드, 생성 유틸리티입니다.
8. [huggingface/swift-chat](https://github.com/huggingface/swift-chat): swift-transformers를 Swift 앱에 통합하는 방법을 보여 주는 작은 앱입니다.
9. [argmaxinc/argmax-oss-swift](https://github.com/argmaxinc/argmax-oss-swift): Core ML과 ANE를 사용하는 Apple Silicon용 온디바이스 음성 인식(WhisperKit)입니다.
10. [argmaxinc/DiffusionKit](https://github.com/argmaxinc/DiffusionKit): Core ML과 MLX를 이용한 온디바이스 이미지 생성입니다.
11. [drawthingsai/draw-things-community](https://github.com/drawthingsai/draw-things-community): 이미지 생성 앱 Draw Things의 커뮤니티 소스로, Swift 패키지와 macOS용 `draw-things-cli`를 포함합니다.

<div align="right"><a href="#목차">맨 위로</a></div>

## 뉴럴 엔진

1. [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers): ANE에 맞게 모델을 다시 쓰는 방법을 Apple이 설명한 글입니다. channels-first (B, C, 1, S) 레이아웃, 선형 레이어 대신 합성곱, 분할 attention을 다룹니다.
2. [apple-aiml-research/ml-ane-transformers](https://github.com/apple-aiml-research/ml-ane-transformers): 위 글과 함께 공개된 Transformer 참조 구현입니다.
3. [On Device Llama 3.1 with Core ML](https://machinelearning.apple.com/research/core-ml-on-device-llama): Llama 3.1 8B를 상태를 가진 KV cache와 int4 가중치로 Core ML에 변환하는 Apple의 안내입니다.
4. [Core ML 문서](https://developer.apple.com/documentation/coreml): 모델을 CPU, GPU, ANE에 나눠 배치하는 프레임워크의 Apple 공식 레퍼런스입니다.
5. [apple/coremltools](https://github.com/apple/coremltools): PyTorch 모델을 Core ML로 변환하고 palettization, 가지치기, 양자화를 적용합니다.
6. [huggingface/exporters](https://github.com/huggingface/exporters): Hugging Face 모델을 Core ML로 내보냅니다. 마지막 업데이트는 2024년입니다.
7. [huggingface/coreml-examples](https://github.com/huggingface/coreml-examples): 앱에서 Core ML 모델을 실행하는 Swift 예제입니다.
8. [Anemll/Anemll](https://github.com/Anemll/Anemll): Hugging Face LLM(Llama, Qwen, Gemma 등)을 ANE용 Core ML 모델로 변환하고 실행하는 파이프라인입니다.
9. [smpanaro/more-ane-transformers](https://github.com/smpanaro/more-ane-transformers): LLM을 포함한 transformer를 ANE에서 돌려 본 실험입니다. 마지막 업데이트는 2023년입니다.
10. [smpanaro/coreml-llm-cli](https://github.com/smpanaro/coreml-llm-cli): Core ML을 통해 ANE에서 LLM을 실행하는 명령줄 데모입니다.
11. [maderix/ANE](https://github.com/maderix/ANE): 리버스 엔지니어링한 비공개 API로 ANE에 작업을 직접 보내며, 실제 fp16 처리량과 온칩 SRAM 한계를 측정했습니다.
12. [mechramc/Orion](https://github.com/mechramc/Orion): Core ML 없이 ANE에서 작은 LLM을 학습하고 실행하는 런타임입니다. 논문 [arXiv 2603.06728](https://arxiv.org/abs/2603.06728).
13. [apple-aiml-research/ml-stable-diffusion](https://github.com/apple-aiml-research/ml-stable-diffusion): Core ML 위의 Stable Diffusion으로, 큰 모델을 ANE와 GPU에 나눠 돌리는 실전 예제입니다.
14. [apple-aiml-research/ml-fastvlm](https://github.com/apple-aiml-research/ml-fastvlm): Apple이 만든 VLM용 효율적인 비전 인코더로, iOS 데모 앱이 함께 있습니다.
15. [ANE guide](https://ane-guide.readthedocs.io/): ANE 프로그래밍과 그 제약을 정리한 커뮤니티 문서입니다.

<div align="right"><a href="#목차">맨 위로</a></div>

## 양자화와 추측 디코딩

> [!NOTE]
> Mac의 decode는 대역폭에 묶여 있어서, 가중치당 비트 수가 상한을 정하고 그 상한을 넘는 방법은 추측 디코딩(speculative decoding)뿐입니다. Apple GPU에서는 초안 토큰 여러 개를 한 번에 검증하는 비용이 공짜가 아니므로([사례 연구](#사례-연구-token-rush의-apple-silicon-이식) 참고), 초안 깊이는 기기마다 따로 맞춰야 합니다.

1. [MLX 문서: `mx.quantize`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html): MLX가 쓰는 affine 그룹 단위 양자화 포맷(그룹마다 코드, scale, bias)과 지원하는 그룹 크기, 비트 폭입니다.
2. [mlx-lm: 학습형 양자화](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md): mlx-lm의 DWQ, AWQ, 동적 혼합 정밀도, GPTQ와 각각에 필요한 calibration을 설명합니다.
3. [GGUF 명세](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md): llama.cpp와 ollama가 읽는 파일 포맷입니다. 메타데이터, 텐서 레이아웃, 양자화 타입을 다룹니다.
4. [llama.cpp quantize 도구](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md): Q4_K_M 같은 GGUF 양자화를 만드는 방법과 importance matrix 옵션입니다.
5. [llama.cpp 추측 디코딩](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md): `llama-server`가 지원하는 초안 모델 방식과 초안 모델 없는 방식의 추측 디코딩, 그리고 관련 옵션입니다.

<div align="right"><a href="#목차">맨 위로</a></div>

## 커널과 저수준

1. [MLX 문서: 커스텀 Metal kernel](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html): Metal로 kernel 본문을 작성하고 Python에서 `mx.fast.metal_kernel`로 호출하는 방법입니다. grid와 threadgroup 설정도 다룹니다.
2. [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf): 언어 레퍼런스입니다. 주소 공간, SIMD-group 함수, threadgroup memory, 행렬 타입을 다룹니다.
3. [Metal `MTLTensor`](https://developer.apple.com/documentation/metal/mtltensor): shader 쪽 `matmul2d`가 다루는 Metal 4 텐서 타입으로, 직접 작성한 kernel에서 M5 Neural Accelerators를 쓰는 통로입니다.
4. [Performing calculations on a GPU](https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu): device, command queue, buffer, dispatch로 이루어진 Apple의 최소 compute 파이프라인 예제입니다.
5. [philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention): Metal로 옮긴 FlashAttention입니다. Apple GPU의 레지스터 압박과 SIMD-group 행렬 연산에 대한 노트가 있습니다.
6. [dougallj/applegpu](https://github.com/dougallj/applegpu): 리버스 엔지니어링한 Apple G13 GPU 명령어 집합 문서와 디스어셈블러입니다.
7. [AsahiLinux/gpu](https://github.com/AsahiLinux/gpu): Asahi Linux가 드라이버 관점에서 M1 GPU를 분석한 연구 노트입니다.
8. [manishklach/mlx-metal-kernels](https://github.com/manishklach/mlx-metal-kernels): 실험적인 MLX 커스텀 kernel 모음(attention, 양자화 matvec, RMSNorm, RoPE, SwiGLU)으로, 예제로 읽기 좋습니다.
9. [BaseRT (arXiv 2607.00501)](https://arxiv.org/abs/2607.00501): command buffer를 직접 제출하고 칩별로 kernel을 융합하는 네이티브 Metal LLM 런타임입니다. M4 Max에서 llama.cpp와 MLX보다 높은 decode 처리량을 보고합니다.

<div align="right"><a href="#목차">맨 위로</a></div>

## 분산 추론

1. [MLX 문서: 분산 통신](https://ml-explore.github.io/mlx/build/html/usage/distributed.html): MPI, ring, JACCL(Thunderbolt 5 위의 RDMA)을 이용한 MLX 집합 통신과 여러 Mac에서 작업을 띄우는 방법입니다.
2. [Explore distributed inference and training with MLX (WWDC26)](https://developer.apple.com/videos/play/wwdc2026/233/): JACCL, Thunderbolt 5 RDMA(macOS 26.2 이상), 여러 Mac에 모델을 나누는 방법을 다룬 Apple 세션입니다.
3. [exo-explore/exo](https://github.com/exo-explore/exo): MLX distributed 위에서 토폴로지를 고려한 텐서 병렬화로 여러 Mac에 하나의 모델을 나눠 실행합니다. Thunderbolt 5 RDMA도 지원합니다.

<div align="right"><a href="#목차">맨 위로</a></div>

## 음성과 이미지

1. [mustafaaljadery/lightning-whisper-mlx](https://github.com/mustafaaljadery/lightning-whisper-mlx): 배치 디코딩과 양자화 모델을 지원하는 MLX용 Whisper입니다. 마지막 업데이트는 2024년입니다.
2. [senstella/parakeet-mlx](https://github.com/senstella/parakeet-mlx): Nvidia의 Parakeet 음성 인식 모델을 MLX로 옮겼습니다.
3. [lucasnewman/f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx): MLX로 구현한 F5-TTS 음성 합성입니다.
4. [riccardomusmeci/mlx-image](https://github.com/riccardomusmeci/mlx-image): MLX용 이미지 분류, 임베딩 모델입니다.

<div align="right"><a href="#목차">맨 위로</a></div>

## 모니터링

> [!NOTE]
> 노트북은 열이 오르면 벤치마크 결과가 달라집니다. 측정하는 동안 GPU 전력, 주파수, 온도를 지켜보세요.

1. [tlkh/asitop](https://github.com/tlkh/asitop): `powermetrics` 기반 터미널 모니터로, Apple Silicon CPU, GPU, ANE 사용률과 전력을 보여 줍니다. 마지막 업데이트는 2024년입니다.
2. [context-labs/mactop](https://github.com/context-labs/mactop): Apple Silicon CPU, GPU, 메모리, 전력을 보여 주는 `top` 스타일 모니터입니다.
3. [vladkens/macmon](https://github.com/vladkens/macmon): sudo 없이 동작하는 Apple Silicon 실시간 모니터로, TUI, JSON, Prometheus 출력을 지원합니다.

<div align="right"><a href="#목차">맨 위로</a></div>

## 벤치마크

1. [llama.cpp discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167): 커뮤니티가 고정된 commit에서 llama-bench로 측정한 M 시리즈 전 칩의 결과입니다. 칩 간 비교 중 재현성이 가장 높습니다.
2. [TristanBilot/mlx-benchmark](https://github.com/TristanBilot/mlx-benchmark): Apple Silicon CPU, GPU에서 MLX 연산 단위 시간을 PyTorch MPS, CUDA와 비교합니다.
3. [john-rocky/apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench): 같은 하네스로 iPhone과 M4 Max에서 Core ML, MLX, llama.cpp의 LLM 성능을 측정했고, 모든 수치에 양자화 방식을 기록했습니다.
4. [XiongjieDai/GPU-Benchmarks-on-LLM-Inference](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference): llama.cpp 추론에서 Nvidia GPU와 Apple Silicon을 비교합니다. 마지막 업데이트는 2024년입니다.
5. [Tom's Hardware: Mac Studio M4 Max 로컬 AI 성능](https://www.tomshardware.com/desktops/exploring-apple-silicons-local-ai-performance-with-the-mac-studio-and-m4-max-m4-max-beats-gb10-and-strix-halo-in-decode-throughput-but-memory-bandwidth-isnt-everything/2): decode와 prefill에서 M4 Max를 GB10, Strix Halo와 비교합니다.

<div align="right"><a href="#목차">맨 위로</a></div>

## 논문

1. [Production-Grade Local LLM Inference on Apple Silicon (arXiv 2511.05502)](https://arxiv.org/abs/2511.05502): MLX, MLC-LLM, Ollama, llama.cpp, PyTorch MPS를 첫 토큰까지의 시간, 처리량, 긴 컨텍스트 기준으로 비교합니다.
2. [Native LLM and MLLM Inference at Scale on Apple Silicon (arXiv 2601.19139)](https://arxiv.org/abs/2601.19139): vllm-mlx의 설계를 다룹니다. MLX 위의 batching과 멀티모달 서빙입니다.
3. [Orion (arXiv 2603.06728)](https://arxiv.org/abs/2603.06728): ANE의 특성을 분석하고, LLM 학습과 추론을 위해 ANE를 직접 프로그래밍합니다.
4. [BaseRT (arXiv 2607.00501)](https://arxiv.org/abs/2607.00501): 칩별 kernel 융합을 하는 네이티브 Metal 런타임입니다.
5. [Recurrent Drafter (arXiv 2403.09919)](https://arxiv.org/abs/2403.09919): Apple이 추측 디코딩용으로 만든 RNN 초안 모델로, MLX를 통한 Apple Silicon GPU 결과를 포함합니다. [블로그](https://machinelearning.apple.com/research/recurrent-drafter), [코드](https://github.com/apple-aiml-research/ml-recurrent-drafter).

<div align="right"><a href="#목차">맨 위로</a></div>

## 강연과 튜토리얼

1. [WWDC25 298: Explore large language models on Apple silicon with MLX](https://developer.apple.com/videos/play/wwdc2025/298/): 텍스트 생성, 양자화, 파인튜닝, MLX Swift 통합을 다룹니다. [글로 정리한 노트](https://wwdcnotes.com/documentation/wwdc25-298-explore-large-language-models-on-apple-silicon-with-mlx/).
2. [WWDC25 315: Get started with MLX for Apple silicon](https://developer.apple.com/videos/play/wwdc2025/315/): MLX 입문입니다. 배열, 지연 평가, 통합 메모리, Python과 Swift API를 다룹니다.
3. [WWDC25 262: Combine Metal 4 machine learning and graphics](https://developer.apple.com/videos/play/wwdc2025/262/): Metal 4의 텐서 리소스와 직접 만든 Metal 파이프라인 안에서 ML을 실행하는 방법입니다.
4. [WWDC24 10161: Deploy machine learning and AI models on-device with Core ML](https://developer.apple.com/videos/play/wwdc2024/10161/): 온디바이스 LLM을 위한 Core ML의 상태 저장 모델, KV cache, 압축을 다룹니다.
5. [WWDC26 233: Explore distributed inference and training with MLX](https://developer.apple.com/videos/play/wwdc2026/233/): JACCL로 여러 Mac을 클러스터로 묶습니다.
6. [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/): PyTorch MPS 백엔드 설정을 안내하는 Apple 페이지입니다.
7. [MLX vs llama.cpp on Apple Silicon](https://yage.ai/share/mlx-apple-silicon-en-20260331.html): 벤치마크, M5 Neural Accelerators, Ollama의 MLX 전환까지 다룬 긴 비교 글입니다.
8. [tiny-llm 책](https://skyzh.github.io/tiny-llm/): MLX 위에 LLM 서빙 스택을 만드는 주차별 강의 자료입니다.

<div align="right"><a href="#목차">맨 위로</a></div>

## 중국어 자료

> [!NOTE]
> 이 분야를 다룬 중국어 자료는 아직 많지 않습니다. 아래는 확인할 수 있었던 것들입니다. 한국어 자료를 알고 있다면 PR로 알려 주세요.

1. [苹果官方发布大模型框架 MLX (DataLearner)](https://www.datalearner.com/blog/apple-mlx-machine-learning-framework-apple-silicon): MLX 공개 당시의 중국어 소개 글로, 통합 메모리와 PyTorch, NumPy 인터페이스와의 대응을 설명합니다.
2. [MLX vs llama.cpp (2026)：功能对比 (Ertas AI)](https://www.ertas.ai/zh/compare/mlx-vs-llama-cpp): 두 엔진의 기능과 적합한 사용처를 비교합니다.
3. [Apple Silicon 上的 MLX：当你需要自己的模型，而非 Apple 的模型](https://blakecrosley.com/zh-Hans/blog/mlx-on-device-ml-apple-silicon): 시스템 내장 Foundation Models 대신 MLX로 직접 모델을 돌려야 할 때를 설명합니다.

<div align="right"><a href="#목차">맨 위로</a></div>

## 관련 목록

1. [antranapp/awesome-mlx](https://github.com/antranapp/awesome-mlx): MLX 프로젝트만 다룹니다. 이 목록은 ANE, Metal kernel, 추론 서버, 다른 엔진까지 포함합니다.

<div align="right"><a href="#목차">맨 위로</a></div>

## 학습 경로

처음 실행해 보는 단계부터 직접 Metal kernel을 작성하는 단계까지의 읽기 순서입니다.

1. 기본 감각을 잡습니다. [기본 원리](#기본-원리)의 설명과 [MLX 통합 메모리](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html) 문서를 읽고, 내 Mac의 상한(대역폭 ÷ 모델 바이트 수)을 직접 계산해 봅니다.
2. 일단 실행해 봅니다. [WWDC25 315](https://developer.apple.com/videos/play/wwdc2025/315/)와 [298](https://developer.apple.com/videos/play/wwdc2025/298/)을 본 다음, 같은 모델을 [mlx-lm](https://github.com/ml-explore/mlx-lm)과 [llama.cpp](https://github.com/ggml-org/llama.cpp)로 생성해 보고 앞서 계산한 상한과 비교합니다.
3. 모델 구현 하나를 읽습니다. mlx-lm의 `models/` 디렉터리에서 아키텍처 하나를 골라, 토큰 하나가 embedding에서 sampling까지 가는 길을 KV cache까지 포함해 따라갑니다.
4. 작은 엔진을 만듭니다. [tiny-llm](https://skyzh.github.io/tiny-llm/)을 따라 attention, KV cache, 양자화 matmul, batching을 구현합니다.
5. MLX의 실행 방식을 익힙니다. [compile](https://ml-explore.github.io/mlx/build/html/usage/compile.html) 문서를 읽고, eager로 실행한 decode step과 compile한 decode step을 측정해 비교합니다.
6. 하드웨어를 익힙니다. [metal-benchmarks](https://github.com/philipturner/metal-benchmarks)와 [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)에서 SIMD group과 threadgroup memory 부분을 읽습니다.
7. kernel을 작성합니다. [커스텀 Metal kernel](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) 문서를 따라 융합 연산(RMSNorm이나 양자화 matvec)을 만들고, 참조 연산과 결과를 대조한 뒤 실제 GB/s를 측정합니다.
8. 실전 kernel을 공부합니다. [mlx](https://github.com/ml-explore/mlx)와 [ggml](https://github.com/ggml-org/ggml)의 Metal 소스를 읽고, attention은 [metal-flash-attention](https://github.com/philipturner/metal-flash-attention)을 봅니다.

<div align="right"><a href="#목차">맨 위로</a></div>

## 사례 연구 token-rush의 Apple Silicon 이식

[token-rush](https://github.com/zyhector/token-rush)는 RTX 5090 한 장을 위해 만든 Qwen3.8-27B 단일 스트림 추론 엔진이고, 여기서는 이를 MacBook Pro M5 Pro(64 GB)의 MLX로 이식했습니다. 이식 코드는 fork [yuxuanfanOrion/token-rush-apple-silicon](https://github.com/yuxuanfanOrion/token-rush-apple-silicon/tree/mac-mlx)의 `mac-mlx` 브랜치에 있으며 upstream에 pull request로 제안했습니다. 모든 측정값을 담은 전체 기록은 그 저장소의 `docs/mac.md`에 있습니다. 요약은 다음과 같습니다.

| M5 Pro의 엔진, 동일한 int4 가중치 | 에세이 | 코드 | 수학 |
|---|---|---|---|
| token-rush, DFlash2 초안(서버) | 32.0 | **49.2** | **51.1** |
| token-rush, MTP 체인(서버) | **34.2** | 38.4 | 38.2 |
| vllm-metal 0.31, 초안 없음(이 모델에서 동작하는 초안이 없음) | 16.9 | 17.0 | 16.9 |
| SGLang, MLX 백엔드 | 실행 안 됨 | | |
| llama.cpp b11429 및 b11474, unsloth GGUF | 잘못된 출력 | | |

단위는 decode tok/s이며 단일 스트림, greedy, 256 토큰 조건이고, 모든 서버를 같은 OpenAI 호환 클라이언트로 측정했습니다.

1. 상한부터 측정합니다. 이 기기의 스트리밍 읽기 대역폭은 291 GB/s이고 토큰마다 13.65 GB의 int4 가중치를 읽으므로, 초안 없는 decode의 상한은 21.3 tok/s입니다. MLX의 int4 matmul은 한 행일 때 이미 이 대역폭의 95-97%를 내므로, GEMV를 직접 작성해도 시간이 줄지 않습니다.
2. 변환하기 전에 포맷을 확인합니다. 엔진의 int4 g128 비대칭 패킹은 MLX의 affine 4-bit 레이아웃과 바이트 단위로 같습니다. 그래서 공개된 checkpoint를 `view(uint32)` 하나로 불러올 수 있고 GPTQ 품질도 그대로 유지됩니다.
3. 독립적인 구현을 기준으로 삼습니다. 같은 가중치에서 mlx-lm의 Qwen3.5 모델과 이식본의 KL은 약 3e-4로, 양자화 자체의 오차보다 두 자릿수 낮습니다.
4. 초안 없는 decode는 프로세스 안에서 18.4 tok/s로 상한의 86.2%입니다. 같은 가중치에서 mlx-lm은 17.2(80.6%)입니다.
5. Apple GPU에서 추측 디코딩의 검증은 공짜가 아닙니다. MLX의 int4 matmul은 2행부터 15행까지 행마다 가중치를 한 번씩 읽기 때문에, 8행 검증 비용이 decode step 하나의 2.2-3.6배입니다(5090에서는 1.13배). M5 Neural Accelerators에서 도는 Metal 4 `matmul2d` kernel을 쓰면 8행 비용이 1.45-1.85배로 내려갑니다.
6. 경쟁 엔진보다 앞서는 차이는 전부 초안에서 나옵니다. 초안 없는 decode는 vllm-metal과 비슷합니다. vllm-metal은 Metal에서 하이브리드 모델에 대해 MTP, DFlash2의 후보 헤드, 심지어 ngram 검증까지 거부합니다. SGLang의 MLX 백엔드는 v0.5.16-v0.5.21에서 시작되지 않고, v0.5.15는 GDN 레이어를 초기화되지 않은 torch 백엔드로 보냅니다. llama.cpp는 이 모델의 두 가지 초안을 모두 지원하지만, 이 Mac에서는 unsloth의 Qwen3.8-27B GGUF로 Metal과 CPU 모두 잘못된 텍스트를 출력합니다. 따라서 그 속도(초안 없이 16, 초안 사용 시 8-11, 한 step에 1.2-1.5 토큰 수락)는 비교 대상이 될 수 없습니다.
7. MLX에서는 grouped-query attention을 접어야 합니다. MLX의 다중 행 attention kernel은 query head마다 K/V를 한 번씩 읽습니다. KV head 하나에 대응하는 query head 6개를 행 차원으로 접고 명시적인 causal mask를 붙였더니, 60k 컨텍스트에서 8행 검증 비용이 3.2분의 1로 줄었습니다. 8k, 32k, 60k에서 모두 needle을 찾았고, 그 길이에서도 DFlash2는 17-35 tok/s를 냅니다.
8. 융합 kernel은 비트 단위로 확인합니다. GDN gating을 `mx.compile`하자 fp32 결과가 3.8e-6만큼 달라졌고, 순환 구조를 거치면서 한 프롬프트에서 기준 대비 KL이 2배로 커졌습니다. 그래서 이 부분은 compile하지 않습니다.
9. 노트북에서의 벤치마크는 한 번의 교차 실행으로 끝냅니다. 같은 설정이 기기가 달아오르면서 step당 74 ms에서 나중에는 115 ms로 측정되었습니다.

<div align="right"><a href="#목차">맨 위로</a></div>

## 기여

자료를 추가하는 PR을 환영합니다. 항목 형식과 수록 기준은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고하세요.

## Star 기록

<a href="https://star-history.com/#yuxuanfanOrion/awesome-apple-silicon-llm&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=yuxuanfanOrion/awesome-apple-silicon-llm&type=Date&theme=dark">
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=yuxuanfanOrion/awesome-apple-silicon-llm&type=Date">
  </picture>
</a>

## 라이선스

[CC0 1.0](LICENSE)
