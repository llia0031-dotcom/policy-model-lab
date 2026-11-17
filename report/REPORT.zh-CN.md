# 视觉、多模态理解与内容安全识别：2025 年 11 月至 2026 年 11 月进展综述

[English](REPORT.md)

范围：2025 年 11 月至 2026 年 11 月公开的计算机视觉、多模态理解和内容安全识别工作，以及它们对短视频与广告内容识别、自动化数据生产的意义。

## 0. 摘要

- **视觉底座同时在“看清细节”和“少算冗余”上前进。** SigLIP-HD 让中等分辨率的特征去模仿高分辨率的特征，推理预算不变，光学字符识别（OCR）类任务提升最明显[[A5]](#a5)；OneVision-Encoder 只处理信息量高的 3.1%–25% 区域[[A4]](#a4)；SAM 3 用名词短语或示例图做提示，就能在图像和视频里分割并跟踪所有匹配的目标[[A2]](#a2)。这对文字密集的广告和预标注都直接有用。
- **开放多模态模型的小尺寸版本已能直接处理视频，但推理链越长不一定越准。** Qwen3-VL 有 2B 和 4B 稠密版本[[B2]](#b2)，Gemma 4 最小 2.3B[[B4]](#b4)，本文列出的 2B–4B 检查点大多采用 Apache-2.0、不用申请即可下载[[F1]](#f1)。长推理会让模型逐渐忽略图像[[B11]](#b11)，带思考的推理还明显依赖字幕等文字线索[[B6]](#b6)。
- **安全识别正从固定类别的守卫模型（guard model）转向“政策即输入”。** SingGuard 在运行时读入自然语言规则并逐条核对，指出触发了哪一条[[C4]](#c4)；FlexGuard 输出连续风险分，让各平台按自己的严格程度设阈值[[C3]](#c3)；ResponseGuard 的 2B 单次前向守卫在回复有害性检测上胜过较新的 3B 推理守卫，耗时约低 150 倍[[C5]](#c5)。短视频广告审核已有专门方法[[C9]](#c9)[[C10]](#c10)，但截至 2026-11-17，还没有找到可公开下载、专门针对短视频广告政策的基准[[C20]](#c20)[[F1]](#f1)。
- **数据生产的主流做法是“模型先标，人来决定”。** 在商品视频广告的评审实验中，看得到模型判官（judge）提示的评审在 60 条广告上把 58% 的镜头标为商品有误，隐藏提示后看同样广告的两位评审只标出 26%[[D4]](#d4)；在 278K 条 TikTok 评论上，大模型标注训练出的分类器以约十分之一的成本胜过人工标注监督的分类器，评估仍要靠人[[D3]](#d3)。只改训练数据、不改架构和算力，视觉语言模型（vision-language model，VLM）在 20 个公开基准上平均提高 11.7 个百分点[[D2]](#d2)。
- **据各公司报告，自动化已承担大部分执法，误判也被当作要压低的指标。** Google 称超过 99% 的违规广告在投放前被拦下，错误的广告主封禁减少 80%[[E3]](#e3)；TikTok 称自动化系统（automated systems）在无人工复核的情况下处理了 94.1% 的违规内容[[E4]](#e4)；Meta 称大模型执法平均比人少犯 13% 的错误[[E2]](#e2)。

## 1. 范围与方法

除这一时间窗口内的工作，本文还引用少量更早的奠基工作交代来龙去脉。资料包括论文（大多发布在 arXiv）、模型卡和代码仓库、公司发布的透明度报告和博客。所有 arXiv 论文的编号、首版日期和摘要都在 2026-11-17 通过 arXiv 官方接口（export.arxiv.org API）取回，正文只写摘要里有的内容。网页逐一打开并摘录原句，模型和数据集的许可证与门控（gated，即下载前要申请或登录）读自 Hugging Face 接口[[F1]](#f1)。公司数字一律照原文转述，不做推算。

## 2. 视觉基础

### 2.1 来龙去脉

2025 年 2 月发布的 SigLIP 2 是常用的参照：它在 SigLIP 的图文对比目标上加入基于描述生成的预训练、自监督损失和在线数据筛选，定位和稠密预测明显变好，并发布 ViT-B（86M）、L（303M）、So400m（400M）和 g（1B）四档视觉编码器（vision encoder）[[A1]](#a1)。同期的 Perception Encoder 发现，只靠对比式图文训练就能得到适合各种下游任务的通用特征，只是最好的特征藏在网络中间层，要额外对齐才能取出[[A9]](#a9)；DINOv3 走纯自监督路线，用 Gram anchoring 解决长时间训练后稠密特征退化的问题[[A8]](#a8)。

最近的工作主要沿这几条线推进：

- **同样的分辨率，看得更细。** SigLIP-HD 在 SigLIP 2 上做“由细到粗”的监督：让中等分辨率图像的粗特征模仿同一张图高分辨率版本的细特征，推理预算完全不变，在 OCR 相关任务上提升最明显[[A5]](#a5)。
- **视频只算有变化的部分。** OneVision-Encoder 借用视频编解码器（codec）的思路，只处理信息熵高的 3.1%–25% 区域；接入大模型后，它用更少的视觉 token（模型处理的最小单位）和预训练数据，在 16 个图像、视频和文档基准上胜过 Qwen3-ViT 和 SigLIP2，视频任务平均比 Qwen3-ViT 高 4.1%[[A4]](#a4)。
- **重要的是“专门的视觉容量”。** Unbundling 在相同数据和算力下从头对比几种设计，发现预训练得来的视觉先验并非必需，和主干一起从头训练的编码器一样好；真正起作用的是给视觉留出专门的参数，这部分参数放进主干的早期层效果相当，还能缩短首个 token 的等待时间[[A6]](#a6)。

### 2.2 分割与检测

SAM 3 把“分割一切”推进到“按概念分割”（promptable concept segmentation，PCS）：提示可以是名词短语、示例图，或名词短语加示例图，模型返回所有匹配实例的掩码和各自的身份编号，图像和视频共用主干[[A2]](#a2)。它的数据引擎产出了 4M 个独特概念标签，其中包括困难负例（hard negative，看着像但不是的样本），在图像和视频的概念分割上把现有系统的准确率翻倍[[A2]](#a2)。代码仓库写明模型有 848M 参数，在包含 270K 个独特概念的 SA-Co 基准上达到人类水平的 75%–80%；2026 年 3 月 27 日又发布了 SAM 3.1 Object Multiplex[[A3]](#a3)。权重要申请访问，使用 SAM License[[A3]](#a3)[[F1]](#f1)。

LED 换了思路：把多模态大模型（MLLM）里语言模型中间层的隐藏状态，经交叉注意力适配器接入开放词表检测器（open-vocabulary detector，能按任意文字类别找物体）[[A7]](#a7)。以 Swin-T 为视觉编码器时，Qwen2-0.5B 加 LED 让 GroundingDINO 在 OmniLabel 上提高 3.82%，额外计算量（以 GFLOPs 计）只多 8.7%，换更大的视觉主干后提高到 6.22%[[A7]](#a7)。作者还发现只适配语言模型的前几层就能拿到大部分收益[[A7]](#a7)。

### 2.3 对内容识别意味着什么

- **文字密集的广告。** 短视频广告的线索分散在画面、语音和字幕里[[C9]](#c9)，公开审核数据也把屏幕文字单独用 OCR 提取出来[[F1]](#f1)。SigLIP-HD 说明不加分辨率也能把 OCR 类任务做得更好[[A5]](#a5)，SigLIP 2 的多语言训练则照顾到多语种文案[[A1]](#a1)。
- **同样的算力，看更多帧。** 只编码变化区域的做法[[A4]](#a4)让整条视频都看一遍变得更便宜，不必只靠封面或少数几帧。
- **用名词短语预标注。** SAM 3 的概念提示能把名词短语直接变成所有实例的掩码，再交给人确认[[A2]](#a2)；训练数据里有困难负例，正好对应审核中“相似但不违规”的情况[[A2]](#a2)。
- **按新类别扩展。** LED 用小语言模型就给检测器补上了语义[[A7]](#a7)，适合需要快速增加新物品类别的场景。

## 3. 多模态理解

### 3.1 开放模型家族

| 家族 | 公开的尺寸 | 与内容识别相关的特点 |
|---|---|---|
| Qwen3-VL[[B2]](#b2) | 稠密 2B/4B/8B/32B，混合专家（MoE）30B-A3B/235B-A22B | 原生支持最长 256K token 的图文视频交错上下文；视频改用文字时间戳对齐，时间定位更准 |
| Qwen3.5[[B3]](#b3) | 首个开放权重模型 Qwen3.5-397B-A17B，总参数 3970 亿，每次前向激活 170 亿；小尺寸见下文的表 | 语言和方言支持从 119 种扩到 201 种 |
| GLM-4.1V-Thinking / GLM-4.5V / GLM-4.6V[[B1]](#b1) | 开源 GLM-4.1V-9B-Thinking 和 GLM-4.5V，另有 GLM-4.6V 系列 | 课程采样强化学习（Reinforcement Learning with Curriculum Sampling，RLCS）；GLM-4.5V 在 42 个公开基准上评测；GLM-4.6V 原生调用工具，上下文 128K |
| Gemma 4[[B4]](#b4) | 2.3B 到 31B，有稠密和 MoE | 所有尺寸都改进了视觉和音频编码器；12B 是不带编码器、直接读原始音频和图像块的统一架构；有思考模式 |
| Qwen3-Omni[[B5]](#b5) | Qwen3-Omni-30B-A3B 系列，Apache 2.0 | 文本、图像、音频、视频统一在单个模型里；在 36 个音频和音视频基准中，32 个达到开源最佳；冷启动时端到端首包延迟理论值 234 ms |
| InternVL3.5[[B12]](#b12) | 最大 InternVL3.5-241B-A28B | 级联强化学习；按需调整视觉分辨率的路由器（ViR）；推理速度比 InternVL3 快 4.05 倍 |

能同时听和看的模型[[B5]](#b5)对广告审核有实际意义：口播里的说法和字幕、画面一样是证据，跨模态不一致本身也是审核要抓的问题[[C9]](#c9)。

### 3.2 视频理解

**评测从“答对一题”转向“答得一致”。** Video-MME-v2 指出现有视频基准趋于饱和，排行榜分数和真实能力之间出现差距；它不再逐题计分，而是按组计分，相关的几道题要答得一致、多步推理要连贯才给分，靠猜碰对的不得分[[B6]](#b6)。数据由 12 名标注员和 50 名独立审核者完成，投入 3,300 人时、最多 5 轮质检[[B6]](#b6)。结果显示当前最好的 Gemini-3-Pro 和人类专家之间仍有明显差距，带思考的推理有字幕时更好，纯视觉时有时反而更差[[B6]](#b6)。

**从离线看完到边看边答。** RIVER 把评测改成实时对话式：回顾记忆、实时感知和主动预判等任务；离线模型单问单答表现好，实时处理却吃力[[B7]](#b7)。Streamo 用 Streamo-Instruct-465K 指令数据训练，让模型同时做实时解说、动作理解、事件描述、时间定位和时间敏感问答[[B8]](#b8)。

**视频为什么还贵。** 有综述指出，视频大模型的计算和内存开销随帧数和上下文长度增长，限制了实时、移动端和资源受限场景的部署[[B9]](#b9)。它按方法作用的环节归类：抽帧、模态编码、连接层的 token 压缩、大模型的预填充和解码，并指出音视频联合效率和标准化评测仍是空白[[B9]](#b9)。

**定位到具体时刻。** TaRO 用预先生成的密集描述构造带时间戳的推理路径，并用“时间敏感奖励”检验推理是否锚定在事件边界上，在视频时间定位（VTG）基准上取得最好成绩[[B10]](#b10)。对广告审核，这类能力对应“违规说法出现在第几秒”。

### 3.3 推理与强化学习

GLM 团队先用大规模预训练得到强视觉底座（作者认为这决定最终上限），再用课程采样强化学习把能力推广到视频理解、内容识别、定位和长文档等任务[[B1]](#b1)。更小的 GLM-4.1V-9B-Thinking 在 29 个基准上超过大得多的 Qwen2.5-VL-72B[[B1]](#b1)。

另一面是“想得越多，看得越少”。VAPO 的研究发现，多模态推理提升逻辑推断的同时，会逐渐削弱感知落地，让模型在基础视觉问题上识别失败；作者称之为视觉遗忘：推理越长，模型越不看视觉输入[[B11]](#b11)。他们提出的 VAPO（视觉锚定的策略优化）在推理过程中显式把模型拉回有视觉依据的路径，得到 VAPO-Thinker-7B[[B11]](#b11)。

### 3.4 8 GB 显存笔记本可跑的 2B–4B 开放检查点

许可证和门控于 2026-11-17 在 Hugging Face 上核对；显存一栏是按权重加 16 帧输入的估算，bf16 指不量化、直接用半精度权重运行[[F1]](#f1)。

| 检查点 | 许可证 | 门控 | 原生视频输入 | 8 GB 显存（估算） |
|---|---|---|---|---|
| Qwen/Qwen3-VL-2B-Instruct | Apache-2.0 | 否 | 是 | bf16 可直接跑 |
| Qwen/Qwen3-VL-4B-Instruct | Apache-2.0 | 否 | 是 | 要 4-bit 量化 |
| Qwen/Qwen3.5-2B | Apache-2.0 | 否 | 是 | bf16 可直接跑 |
| Qwen/Qwen3.5-4B | Apache-2.0 | 否 | 是 | 要 4-bit 量化 |
| OpenGVLab/InternVL3_5-2B-HF | Apache-2.0 | 否 | 是 | bf16 可直接跑 |
| OpenGVLab/InternVL3_5-4B-HF | Apache-2.0 | 否 | 是 | 要 4-bit 量化 |
| HuggingFaceTB/SmolVLM2-2.2B-Instruct | Apache-2.0 | 否 | 是 | bf16 可直接跑 |
| google/gemma-4-E2B-it（有效 2.3B，含嵌入 5.1B） | Apache-2.0 | 否 | 按帧处理，最长 60 秒，每秒 1 帧 | 要 4-bit 量化 |
| allenai/Molmo2-4B | Apache-2.0 | 否 | 是 | 要 4-bit 量化 |
| Qwen/Qwen2.5-VL-3B-Instruct | qwen-research | 否 | 是 | — |
| google/gemma-3-4b-it | Gemma | 是，人工审批 | 只收帧 | — |

## 4. 内容安全识别

### 4.1 从固定类别到“政策即输入”

早几代开放守卫模型多按固定的风险类别训练。ShieldGemma 2 是基于 Gemma 3 的 4B 图像审核模型，覆盖色情、暴力血腥和危险内容等关键类别[[C17]](#c17)；LlavaGuard 已经允许自定义安全分类，模型从 0.5B 到 7B，训练数据由专家给每张图标注安全等级、类别和理由[[C16]](#c16)；GuardReasoner-VL 用在线强化学习让守卫先推理再判断，训练语料有 123K 个样本和 631K 个推理步，平均 F1（兼顾查准和查全的分数）比次优方法高 19.27%[[C18]](#c18)。视频方面，SafeWatch 把每段政策并行编码，再按政策挑出最相关的视觉 token，以便遵循定制的安全政策[[C19]](#c19)。

最近的工作里，“政策即输入”成为主线：

- **自带政策。** gpt-oss-safeguard 按使用者自己写的安全政策给文字内容分类，推理强度可以按低、中、高调节；gpt-oss-safeguard-20b 总参数 21B、激活 3.6B，可放进 16GB 显存的 GPU，采用 Apache 2.0 许可证[[C1]](#c1)。
- **覆盖全部模态。** GuardReasoner-Omni 覆盖文本、图像、视频和音频四种模态，训练语料有 181k 个样本；先用监督微调（supervised fine-tuning，SFT）教会它先推理再判断，再用只奖励“简洁且正确”的强化学习压掉冗余生成，发布 3B 和 7B 版本[[C2]](#c2)。
- **严格程度可调。** FlexGuard 的出发点是：各平台对“有害”的界定松紧不同，还会随时间变化[[C3]](#c3)。它的 FlexBench 在多种严格程度下评测，发现在一种严格程度下表现好的模型换到另一种会明显退化；FlexGuard 改为输出校准过的连续风险分，部署时按目标严格程度选阈值[[C3]](#c3)。
- **规则逐条核对。** SingGuard 把当前生效的政策当作运行时输入，逐条规则核对，同时输出安全标签和被触发的规则，支持从直接判断到逐条推理的快、慢多档推理[[C4]](#c4)。配套的 SingGuard-Bench 有 80 多种细分风险，包括“每种模态单看无害、合起来才有害”的跨模态组合风险；SingGuard 在 6 个基准家族（35 个数据集）上的平均 F1 都是最好，规则在运行时变化后，政策遵循准确率从 0.6465 提到 0.7415[[C4]](#c4)。
- **不推理也行吗。** ResponseGuard 从请求、回复和图像的一个池化表示里一次前向读出判断[[C5]](#c5)。在常用的多模态守卫基准上，2B 的 ResponseGuard 在回复有害性检测上胜过较新的 3B 推理守卫，而且不做推理、耗时约低 150 倍；在请求有害性上推理守卫仍整体领先，两条赛道剩下的差距都集中在只有图像的单元格[[C5]](#c5)。作者认为差距可能来自推理和非推理设计共用的冻结视觉编码器；单次前向也让它能随回复的流式输出逐句筛查[[C5]](#c5)。

开放权重方面，GuardReasoner-VL-3B 采用 Apache-2.0、不用申请，ShieldGemma 2 需要人工审批，GuardReasoner-Omni-3B 的模型页没有许可证标签，它的底座 Qwen2.5-Omni-3B 是 qwen-research 许可[[F1]](#f1)[[C17]](#c17)。

### 4.2 短视频与广告审核

- **KuaiMod：判例法式审核。** 这项工作面向快手场景，标注了首个带真实用户和审核员反馈的短视频平台审核基准，并提出 KuaiMod 框架：用 VLM 和思维链（chain of thought，CoT）推理，根据稀疏的用户反馈建模视频的有害程度，让审核政策能快速更新，并已在快手上线部署[[C8]](#c8)。
- **AdDriftBench：广告审核里的漂移。** 它专门针对短视频广告的合规审核，共 3,480 条短视频广告，其中 2,280 条用于数据漂移场景（内容分布快速变化），另外 1,200 条用于标签漂移场景（审核标准随时间细化，类别边界随之移动）[[C20]](#c20)。在 16 个开源 VLM 上的实验显示，模型在细粒度语义和适应变化的指令上表现一般；论文写明数据集将会公开[[C20]](#c20)。
- **BLM-Guard：政策对齐的广告审核。** 短视频平台上的多模态广告，画面、语音和字幕需要比社区安全过滤更细、以政策为准的审核[[C9]](#c9)。BLM-Guard 用规则驱动的数据合成生成场景描述、推理链和标签以降低标注成本，再用兼顾因果连贯和政策符合度的奖励做强化学习，并同时建模单一模态内的问题（如夸大的画面）和跨模态的不一致（如字幕和语音对不上）[[C9]](#c9)。
- **Summarize-Judge-Refine（SJR）：先看懂，再判断。** 多模态的内容模型先写出结构化的文字摘要，纯文本的政策模型再对照政策定义分类，这样不必每次政策变化都重训整条流水线[[C10]](#c10)。迭代的协同训练用 GRPO（一种强化学习算法）让摘要更贴近政策要点，并在文字空间生成对抗性的摘要变体做数据增强，这在原始多媒体上做不到[[C10]](#c10)。在误导性广告检测上，SJR 的非误导类（即合规广告）F1 比零样本思维链基线相对提高 23.6%；完全没有真实违规样本、正例全部合成的变体，违规类 F1 与全量数据模型的相对差距在 0.2% 以内[[C10]](#c10)。作者据此认为“new policies can launch without any real violation data”[[C10]](#c10)。
- **从失败分类到改进手段。** 面向业界规模视频和直播审核的诊断研究指出，平台要适配自己的数据分布、政策目标和产品安全约束，通用公开模型和外部 API（接口服务）难以直接满足；它把模型失败归入可观察的失败特征，每类对应一组改进手段，并在支持 100 多个地区的音视频语言模型（AVLM）上落地[[C11]](#c11)。
- **有害视频要讲出理由。** HarmVideoBench 有 1,379 个视频和 4,137 道选择题，分可观察证据、片段内含义和片段外推理三层，评测了 19 个模型；作者提出的 BCR 方法只在需要时检索上下文，把宏平均从 61.7% 提到 84.4%[[C6]](#c6)。
- **隐性有害。** DeepHarm-7K 关注通过叙事、情绪渲染或跨模态线索传递误导性价值观的短视频，有 7,110 个按细粒度分类标注的样本；配套的 DeepHarm-VL 用两轮推理，先找表层证据，再推断潜在含义[[C7]](#c7)。

### 4.3 过度拒绝

安全机制可能矫枉过正，模型出于过度谨慎拒绝正常请求，这就是过度拒绝（over-refusal）；理想行为是“安全完成”：完成请求中无害的部分，同时明确提示潜在风险[[C12]](#c12)。DUAL-Bench 在 12 个危害类别、语义不变的视觉扰动下评测了 18 个 VLM，表现最好的 GPT-5-Nano 安全完成率为 12.9%，GPT-5 和 Qwen 系列平均为 7.9% 和 3.9%[[C12]](#c12)。放到广告审核里，过度拒绝对应的是合规广告被误拒，下文把它列为要单独衡量的指标。

### 4.4 生成内容与深度伪造的解释

深度伪造检测模型越来越多地给出自然语言解释，但解释常常没有落在真实的篡改痕迹上[[C13]](#c13)。XPlainVerse 有百万张真实和被篡改的图像，真实图来自五个来源，伪造图由十二个现成的编辑和生成模型产生；多阶段过滤流程 Edit-Check 确认每次编辑确实达到预期[[C13]](#c13)。新指标 EntityScore 和 EvidenceScore 检查解释是否指出了被篡改的对象和视觉证据，2,000 对解释的人工标注验证了数据质量[[C13]](#c13)。对用生成工具制作的广告素材，这类“指出哪里被改过”的解释能帮审核员更快确认。

### 4.5 政策即提示词的治理

大模型可以直接把政策文本当输入来判断，不再需要先把政策转成标注指南、再标大量数据[[C14]](#c14)。Policy-as-Prompt 一文归纳了四个层面的五个挑战：政策怎样翻译成提示词，结果对提示词结构和格式敏感，政策制定被技术牵着走的风险，政策团队和机器学习团队角色的变化，以及模型治理与问责[[C14]](#c14)。后续工作进一步论证，只把规则交给大模型并不够，提示词本身需要治理，结论是“writing prompts alone is not appropriate for ensuring meaningful community governance”[[C15]](#c15)。

### 4.6 公开基准一览

访问方式和许可证于 2026-11-17 核对[[F1]](#f1)。

| 基准 | 内容 | 访问方式 | 许可证 |
|---|---|---|---|
| KuaiMod[[C8]](#c8) | 快手短视频，封面加最多 8 帧，附标题、OCR 和语音识别（ASR）文字 | 不用登录 | Hugging Face 标 CC BY-NC 4.0，GitHub 仓库标 CC-BY-SA-4.0，彼此不一致[[F1]](#f1) |
| SafeWatch-Bench[[C19]](#c19) | 超过 2M 个视频，六个安全类别 | 人工审批 | Hugging Face 标 Apache-2.0[[F1]](#f1) |
| HarmVideoBench[[C6]](#c6) | 1,379 个视频、4,137 道选择题 | 摘要没有给数据链接 | — |
| DeepHarm-7K[[C7]](#c7) | 7,110 个短视频样本，侧重隐性有害 | 摘要没有给数据链接 | — |
| AdDriftBench[[C20]](#c20) | 3,480 条短视频广告，分数据漂移和标签漂移 | 论文写明将会公开 | — |
| BLM-Guard 数据[[C9]](#c9) | 真实短视频广告 | 摘要没有提数据发布 | — |

截至 2026-11-17，没有找到可公开下载、又专门覆盖短视频广告政策的基准：专门做短视频广告的 AdDriftBench 和 BLM-Guard 还没有可下载的数据[[C20]](#c20)[[C9]](#c9)，可下载的 KuaiMod 覆盖的是短视频平台的一般审核[[C8]](#c8)[[F1]](#f1)。

## 5. 自动化数据生产

### 5.1 预标注：模型标出，人来决定

有工作针对由生成流水线制作的商品视频广告，把评审拆成每个镜头六个问题：商品、画面质量、发音、脚本、声音和字幕；由视觉语言、语音和语言模型组成的判官把每条发现归到对应问题下，评审逐条接受或驳回，并补上判官漏掉的[[D4]](#d4)。在 120 条广告上，带判官提示的评审把 60% 的镜头标为商品有误[[D4]](#d4)。在其中 60 条广告上，评审标出 58% 的镜头商品有误，远多于后来在隐藏判官提示的情况下看同样广告的两位评审标出的 26%；一位评审在有提示和无提示时都看过，分别标出 57% 和 19%[[D4]](#d4)。在 120 个盲标镜头上，结合商品目录照片看片段的商品判官 AUC（区分有问题和没问题镜头的能力）达到 0.84，而视频质量分和照片相似度分接近随机[[D4]](#d4)。判官按召回率调：几乎抓到所有缺陷，也会把很多干净镜头标出来，所以由评审决定；没有判官提示时，一位评审的标注在一次评审过程中逐渐变少[[D4]](#d4)。作者把这种分工概括为“the judges widen what reviewers mark and reviewers add what the judges miss”[[D4]](#d4)。

平台侧的评估也采用同样的分工。Pinterest 以领域专家（subject matter expert，SME）整理的高可信金标集（golden set）作为基准答案，用倾向分（propensity score）抽样扩大覆盖，用来比较不同大模型代理的成本和效果、做数据驱动的提示词优化、管理政策演变[[E5]](#e5)。它的违规率（prevalence）指标是某一天流向违规内容的浏览量占比：系统每天按机器学习辅助的权重从浏览流里抽样，把标注预算集中到高曝光、高风险内容上，样本由受政策提示词约束、经金标集验证的多模态大模型标注，给出带置信区间的无偏估计，并可按版位、地区等维度拆开看[[E6]](#e6)。

### 5.2 大模型标签与人工标签

有研究在 278K 条德语政治类 TikTok 评论上训练仇恨言论分类器，在相同条件下比较人工和大模型标注[[D3]](#d3)。大规模的大模型标注以约十分之一的成本胜过人工标注监督的分类器，闭源的 GPT-5.2 和开放权重的 Qwen3.5-122B-A10B 都是如此，换成软标签评估结论不变[[D3]](#d3)。前提是标注界面：只有模仿人工标注任务的两问拆解才行[[D3]](#d3)。在预先过滤过的样本池里，主动学习（active learning）并不比随机抽样稳定更好[[D3]](#d3)。错误结构因模型而异：只有 GPT-5.2 的误报和漏报比例与人工相当，其他模型会过度标记边境管控和经济竞争类话语[[D3]](#d3)。作者的结论是“Humans remain essential as evaluators”，训练标签的问题则变成选哪个模型、用什么界面、样本池长什么样[[D3]](#d3)。

### 5.3 合成数据：能做什么，边界在哪

**能做什么。** 用合成数据训练的小模型可以超过生成数据的大模型：在 11 种语言、4 个分类任务上，少量合成数据就让小模型胜过生成它的大模型，低资源语言上尤其明显[[D12]](#d12)。前面提到的 SJR 零真实违规样本版本[[C10]](#c10)和 BLM-Guard 的规则驱动合成[[C9]](#c9)，都是用合成数据给审核模型起步。

**边界在哪。** 任务越主观，无论是任务层面还是单个样本层面，用合成数据训练的模型表现越差[[D7]](#d7)。合成数据格式单一会导致模式过拟合、输出分布明显偏移，削弱指令遵循能力[[D11]](#d11)。在 350M 和 1.4B 参数模型上的对照实验显示，合成数据的多样性与预训练和监督微调效果正相关，而且预训练阶段的多样性对后续微调的影响比对预训练本身更大[[D10]](#d10)。综述按“生成、筛选、评估”的流程梳理了这类研究[[D8]](#d8)，最佳实践文章强调事实性、保真度和无偏性[[D9]](#d9)。

**模型坍塌（model collapse）。** 用每一代合成数据替换原始真实数据会走向坍塌，而把各代合成数据和原始真实数据累积在一起可以避免；在线性模型的分析框架下，数据累积时测试误差有与迭代次数无关的有限上界[[D6]](#d6)。

### 5.4 污染与数据划分

基准数据污染（benchmark data contamination）指模型训练时无意中接触了评测数据，使评测结果不可靠[[D15]](#d15)。训练集和测试集重叠会抬高成绩，把记忆误当作泛化[[D17]](#d17)。随机划分和标准划分一样会给出过于乐观的估计，即使是刻意制造偏差的最坏划分，也常常低估模型在同领域新样本上的错误；作者建议准备多个独立测试集，做不到时用多个有偏划分[[D16]](#d16)。GSM1k 按 GSM8k 的风格和难度重新出题，部分模型在新题上准确率最多下降 8%；模型生成 GSM8k 样本的概率与它在两套题上的成绩差正相关（Spearman r^2 = 0.36），但许多前沿模型几乎没有过拟合迹象[[D18]](#d18)。

### 5.5 先筛选，再扩规模

UniFilter 把高效的多模态大模型训练成数据质量分类器，同时筛选图文描述数据和图文交错文档[[D1]](#d1)。带标签的多模态数据难收集，作者用半合成方法：拿现成的原始图像，生成四个质量等级的配文，得到“样本、分数”对[[D1]](#d1)。用它从 DataComp 和 OBELICS 里筛出的数据预训练，模型的零样本推理和上下文学习能力更强，筛出的 OBELICS-HQ 也已公开[[D1]](#d1)。

20/20 Vision Language Models 固定架构、训练配方和算力，只改训练数据：筛选流水线让 VLM 在 20 个公开基准上平均提高 11.7 个百分点，这些基准覆盖定位、问答、OCR 与文档、计数、图表、品牌识别等能力[[D2]](#d2)。在 2B 规模，筛选后的模型超过 InternVL3.5-2B 9.9 个百分点，训练算力少约 17 倍；与 Qwen3-VL-2B 的差距缩小到 1.8 个百分点以内，训练算力少约 87 倍[[D2]](#d2)。不同训练种子之间的单项能力标准差下降约 67%；在 4B 规模，筛选后的模型达到接近前沿的准确率，回答时的浮点运算量（FLOPs）比 Qwen3-VL-4B 低 3.3 倍[[D2]](#d2)。

### 5.6 用微调做小任务模型

微调后的小模型（BERT 类）在文本分类上持续、显著地优于零样本提示的大模型，对比对象包括 ChatGPT（GPT-3.5/GPT-4）和 Claude Opus，任务覆盖情感、赞成与反对、情绪和政党立场[[D13]](#d13)。SLM-Mod 用 15 个热门 Reddit 社区的 150K 条评论微调 15B 以下的小模型，平均准确率比零样本大模型高 11.5%、召回率高 25.7%；少样本的上下文学习只给大模型带来很小的提升[[D14]](#d14)。关于微调方式，LoRA Without Regret 发现，在中小规模的指令微调和推理数据集上，低秩适配（LoRA）的效果与全量微调相同；LoRA 的最佳学习率稳定在全量微调的 10 倍；某些情况下 LoRA 对大批量更不耐受[[D5]](#d5)。

## 6. 行业实践

本节数字均为各公司自己发布的数字。

| 公司 | 报告内容 | 数字 | 页面日期 | 来源 |
|---|---|---|---|---|
| Meta | 更先进的 AI 执法系统覆盖的语言 | 覆盖全球 98% 网民使用的语言 | 2026-03 | [[E1]](#e1) |
| Meta | 检测违规的成人性诱导内容 | AI 抓到的违规内容是人工的两倍多，错误率降低超过 60% | 2026-03 | [[E1]](#e1) |
| Meta | 冒充知名人物 | 相关用户举报减少 80% | 2026-03 | [[E1]](#e1) |
| Meta | 大模型与人工执法对比 | 平均少犯 13% 的错误，多发现 10% 的真实违规 | 2026-10 | [[E2]](#e2) |
| Meta | 成人裸露和性活动内容的违规率 | 从 2025 年第四季度的 0.10%–0.11% 降到 2026 年第二季度的 0.07%–0.08% | 2026-10 | [[E2]](#e2) |
| Meta | 诈骗广告 | 在 Facebook 和 Instagram 上移除 6,500 万条，其中 94% 在有人举报前移除 | 2026-10 | [[E2]](#e2) |
| Google | 违规广告的投放前拦截 | 超过 99% 在投放前被拦下 | 2026-04 | [[E3]](#e3) |
| Google | 全年执法量 | 拦截或移除超过 83 亿条广告，封禁 2,490 万个账户 | 2026-04 | [[E3]](#e3) |
| Google | 误判 | 错误的广告主封禁减少 80% | 2026-04 | [[E3]](#e3) |
| TikTok | 欧盟《数字服务法》（DSA）透明度报告 | 自动化系统在无人工复核的情况下处理了 94.1% 的违规内容；移除约 1.04 亿条违规内容 | 2026-08 | [[E4]](#e4) |
| Pinterest | 审核决策质量评估框架 | 方法论文，没有汇总数字 | 2026-02 | [[E5]](#e5) |
| Pinterest | 违规率测量系统 | 方法论文，没有汇总数字 | 2026-02 | [[E6]](#e6) |
| 快手 | KuaiMod 上线（大规模线上 A/B 测试） | 用户举报率下降 20% | 2025-04 | [[C8]](#c8) |

共同的做法有：

- **逐步切换。** Meta 写明，只有当新技术“consistently performed better than our existing systems in all of our tests”，才会转向 AI 优先的执法[[E1]](#e1)；它在 2026 年 10 月的报告里把成人裸露类内容违规率的下降归因于用大模型补上执法缺口[[E2]](#e2)。
- **误判和覆盖一起报告。** Meta 报告错误率的下降[[E1]](#e1)[[E2]](#e2)，Google 报告错误封禁的减少，并称大多数自适应搜索广告（Responsive Search Ads）已能即时审核[[E3]](#e3)。对广告主而言，这意味着更快上线、更少被误伤。

## 7. 对实践的启示

| 场景 | 建议做法 | 依据 |
|---|---|---|
| 短视频广告审核：证据在哪里 | 按标题与文案、屏幕文字（OCR）、画面、音频（口播与 ASR）分别取证，每个判断注明证据出处；专门检查跨模态不一致，例如字幕与口播对不上 | [[C8]](#c8) [[C9]](#c9) [[C10]](#c10) [[B5]](#b5) [[F1]](#f1) |
| 成本受限时的抽帧 | 先稀疏均匀抽帧，再在出现商品、屏幕文字或口播说法的片段加密；优先选在标准分辨率下 OCR 更强、能跳过静态背景的编码器；长视频和直播考虑流式模型 | [[B9]](#b9) [[A4]](#a4) [[A5]](#a5) [[B7]](#b7) [[B8]](#b8) [[B10]](#b10) |
| 快守卫加慢推理器的级联 | 单次前向的小守卫给每条内容打连续风险分；各版位、各地区按自己的严格程度设阈值；落在中间带的再交给按政策逐条推理、能指出触发规则的模型 | [[C5]](#c5) [[C3]](#c3) [[C4]](#c4) [[C1]](#c1) [[C19]](#c19) |
| 把误拒当作一等指标 | 同时报告漏放率和误拒率；测试集里放入“看着像违规、其实合规”的困难负例；用合规类 F1 和错误拒绝、错误封禁的比例衡量改进 | [[C12]](#c12) [[C10]](#c10) [[E3]](#e3) [[E1]](#e1) [[E2]](#e2) |
| 预标注 | 用概念分割器按名词短语出掩码，用多模态判官按问题清单列出发现；判官按召回率调，人逐条接受或驳回 | [[A2]](#a2) [[A3]](#a3) [[D4]](#d4) [[C16]](#c16) |
| 小任务分类器的合成数据 | 按模板、商品、写法划分训练集和测试集，不随机切行；跨集合去掉近似重复；控制格式和多样性；保留并累积真实数据，主观类别多用真实标注 | [[D7]](#d7) [[D10]](#d10) [[D11]](#d11) [[D12]](#d12) [[D16]](#d16) [[D17]](#d17) [[D6]](#d6) [[C10]](#c10) |
| 先筛选，再扩规模 | 扩充数据之前，先训练或借用质量分类器筛掉低质样本 | [[D1]](#d1) [[D2]](#d2) |
| 金标集与漂移检查 | 由专家维护金标集；定期用风险加权抽样估计违规率；政策或内容分布变化后，用新的独立测试集复核 | [[E5]](#e5) [[E6]](#e6) [[D18]](#d18) [[D16]](#d16) [[C20]](#c20) [[C3]](#c3) |

### 尚未定论的问题

1. **推理应该放在哪一层？** ResponseGuard 在回复检测上不推理也够用，但在请求检测上推理守卫仍领先[[C5]](#c5)；长推理会削弱视觉落地[[B11]](#b11)，带思考的推理依赖文字线索[[B6]](#b6)。对证据主要在画面里的短视频广告，推理的收益和代价还缺少公开的系统比较。
2. **政策写成提示词以后，一致性和问责由谁保证？** 结果对提示词的结构和格式敏感，政策团队和机器学习团队的分工也在变[[C14]](#c14)；只写提示词不足以保证有意义的治理[[C15]](#c15)；现有审核模型在不同严格程度之间表现不一致[[C3]](#c3)，政策在运行时变化后的遵循准确率仍有提升空间[[C4]](#c4)。
3. **合成数据和模型标签能替代多少真实标注？** 零真实违规样本的模型几乎追平全量数据模型[[C10]](#c10)，大模型标签训练出的分类器胜过人工监督的分类器[[D3]](#d3)；但任务越主观，合成数据越不管用[[D7]](#d7)，这些结论也还没有在公开的短视频广告数据上检验过[[C20]](#c20)。

## 8. 本项目实验与文献的关系

[`eval/EVALUATION.zh-CN.md`](../eval/EVALUATION.zh-CN.md) 里的实验，在小范围内检验了上面的几条：为本项目渲染的 96 条短视频广告，214 条文字广告（168 条是为有据审核编写的，46 个真实案例），以及 KuaiMod 的 300 条真实视频。每个模型只跑一次，在一块笔记本显卡上或通过一个接口，所以这些数字只说明方向，不代表普遍的比例。

| 上文的观点 | 实验的发现 |
|---|---|
| 证据分别在文案、屏幕文字和画面里，要分别取证（第 7 节）[[C8]](#c8)[[C9]](#c9)[[C10]](#c10) | 只给文案时，claude-haiku-5-5 在 64 条违规广告里抓到 25 条；给 16 帧时 64 条全部抓到。把 16 帧的 OCR 文字交给 Qwen3.5-4B，抓到 57 条，但 16 个只在画面里的违规只找到 5 个，而且都是从商品推断出来的，不是看到的。 |
| 视频模型的成本随帧数增长[[B9]](#b9) | Qwen3.5-4B 每条广告用时的中位数：文案加 OCR 文字 2.4 秒，8 帧 9.2 秒，16 帧视频 19.2 秒。只出现半秒的贴纸，claude-haiku-5-5 每秒看一帧时在 20 条里找到 13 条，每秒两帧时找到 19 条。 |
| 先总结成文字，再交给文字模型判断[[C10]](#c10) | 这里最便宜的路线是同样的结构：OCR 把画面变成文字，文字分类器来判断。用合成广告微调的 Qwen3-1.7B 只看文案加 OCR 文字，判对 80% 的视频，和 claude-haiku-5-5 看同样的文字一样，比 claude-haiku-5-5 看 16 帧（90%）低 10 个百分点。SJR 的总结由多模态模型来写[[C10]](#c10)，要覆盖只在画面里的内容，总结就得这样写。 |
| 微调过的小模型对比零样本的大模型[[D13]](#d13)[[D14]](#d14)；用合成数据训练的小模型可以超过生成数据的模型[[D12]](#d12) | 只用 claude-haiku-5-5 写的 1,529 条合成广告训练，加 LoRA 的 Qwen3-1.7B 在有据审核上得 79%，claude-haiku-5-5 得 82%；在 46 个真实案例上得 83%，claude-haiku-5-5 得 72%（放宽回答长度后 76%）。两个差距都没有超出 95% 区间。 |
| 随机或按模板划分会高估效果[[D16]](#d16)[[D17]](#d17) | 合成验证集按商品和写法单独留出，LoRA 模型在有据审核的测试题上仍比在验证集上低 12 个百分点左右。 |
| 合成数据格式单一，可能导致过拟合[[D11]](#d11) | mmBERT-base 在合成验证集上和 1.7B LoRA 模型持平（91%），但在真实案例的 35 条违规里只抓到 15 条，LoRA 模型抓到 33 条。格式过拟合是一种可能的原因，没有专门检验。 |
| 把误拒单独当作指标，测试集里放入陷阱题[[C12]](#c12)[[C10]](#c10) | 抓得越多的输入，错拒的合规广告也越多：claude-haiku-5-5 看 16 帧时，22 道陷阱题错拒 9 道，Qwen3.5-4B 看 8 帧时错拒 17 道。在 KuaiMod 上，文字之外再给画面，127 条违规里抓到的从 73 条升到 111 条，173 条正常视频里被标成违规的从 62 条升到 70 条（claude-haiku-5-5）；Qwen3.5-4B 分别从 58 条到 97 条、从 17 条到 45 条。 |
| 平台要适应自己的数据和规则[[C11]](#c11)；KuaiMod 的标签按该平台的规则和用户反馈来定[[C8]](#c8) | 在 KuaiMod 上零样本、只给 15 种违规类型的名称时，最好的一组（Qwen3.5-4B 看画面加文字）判对 75% 的视频，把 26% 的正常视频标成违规；claude-haiku-5-5 判对 71%，标错 40%。在提示词里写明了类别定义的渲染广告上，claude-haiku-5-5 判对 90%。 |

所有运行都关闭了思考，所以上面第一个尚未定论的问题（推理在哪里有用）没有检验。

## 9. 参考文献

- <a id="a1"></a>**[A1]** SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features. Tschannen et al., 2025-02. arXiv. <https://arxiv.org/abs/2502.14786>
- <a id="a2"></a>**[A2]** SAM 3: Segment Anything with Concepts. Carion et al., 2025-11. arXiv. <https://arxiv.org/abs/2511.16719>
- <a id="a3"></a>**[A3]** SAM 3 repository README (SAM 3.1 Object Multiplex release note). Meta (facebookresearch), 2026-03. GitHub 仓库. <https://github.com/facebookresearch/sam3>
- <a id="a4"></a>**[A4]** OneVision-Encoder: Codec-Aligned Sparsity as a Foundational Principle for Multimodal Intelligence. Tang et al., 2026-02. arXiv. <https://arxiv.org/abs/2602.08683>
- <a id="a5"></a>**[A5]** SigLIP-HD by Fine-to-Coarse Supervision. Yang et al., 2026-07. ICLR 2026；arXiv. <https://arxiv.org/abs/2607.09488>
- <a id="a6"></a>**[A6]** Unbundling Multimodal Vision Encoding. Brown et al., 2026-11. arXiv. <https://arxiv.org/abs/2611.17334>
- <a id="a7"></a>**[A7]** LED: LLM Enhanced Open-Vocabulary Object Detection without Human Curated Data Generation. Zhou et al., 2025-03. arXiv. <https://arxiv.org/abs/2503.13794>
- <a id="a8"></a>**[A8]** DINOv3. Siméoni et al., 2025-08. arXiv. <https://arxiv.org/abs/2508.10104>
- <a id="a9"></a>**[A9]** Perception Encoder: The best visual embeddings are not at the output of the network. Bolya et al., 2025-04. arXiv. <https://arxiv.org/abs/2504.13181>
- <a id="b1"></a>**[B1]** GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning. GLM-V Team, 2025-07. arXiv. <https://arxiv.org/abs/2507.01006>
- <a id="b2"></a>**[B2]** Qwen3-VL Technical Report. Bai et al., 2025-11. arXiv. <https://arxiv.org/abs/2511.21631>
- <a id="b3"></a>**[B3]** Qwen3.5: Towards Native Multimodal Agents. Alibaba Cloud (Qwen), 2026-02. 博客. <https://www.alibabacloud.com/blog/qwen3-5-towards-native-multimodal-agents_602894>
- <a id="b4"></a>**[B4]** Gemma 4 Technical Report. Gemma Team, 2026-07. arXiv. <https://arxiv.org/abs/2607.02770>
- <a id="b5"></a>**[B5]** Qwen3-Omni Technical Report. Xu et al., 2025-09. arXiv. <https://arxiv.org/abs/2509.17765>
- <a id="b6"></a>**[B6]** Video-MME-v2: Towards the Next Stage in Benchmarks for Comprehensive Video Understanding. Fu et al., 2026-04. arXiv. <https://arxiv.org/abs/2604.05015>
- <a id="b7"></a>**[B7]** RIVER: A Real-Time Interaction Benchmark for Video LLMs. Shi et al., 2026-03. arXiv. <https://arxiv.org/abs/2603.03985>
- <a id="b8"></a>**[B8]** Streaming Video Instruction Tuning. Xia et al., 2025-12. CVPR 2026；arXiv. <https://arxiv.org/abs/2512.21334>
- <a id="b9"></a>**[B9]** Why Is Video Still So Expensive? A Survey of Inference-Efficiency Mechanisms in Video and Audiovisual LLMs. Steunou et al., 2026-09. arXiv. <https://arxiv.org/abs/2609.10355>
- <a id="b10"></a>**[B10]** Temporal-Aware Reasoning Optimization for Video Temporal Grounding. Zheng et al., 2026-06. ICML 2026；arXiv. <https://arxiv.org/abs/2606.09248>
- <a id="b11"></a>**[B11]** More Thought, Less Accuracy? On the Dual Nature of Reasoning in Vision-Language Models. Tian et al., 2025-09. ICLR 2026；arXiv. <https://arxiv.org/abs/2509.25848>
- <a id="b12"></a>**[B12]** InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency. Wang et al., 2025-08. arXiv. <https://arxiv.org/abs/2508.18265>
- <a id="c1"></a>**[C1]** gpt-oss-safeguard-20b model card. OpenAI, 2026-01（最后更新）. Hugging Face 模型卡. <https://huggingface.co/openai/gpt-oss-safeguard-20b>
- <a id="c2"></a>**[C2]** GuardReasoner-Omni: A Reasoning-based Multi-modal Guardrail for Text, Image, Video, and Audio. Zhu et al., 2026-02. arXiv. <https://arxiv.org/abs/2602.03328>
- <a id="c3"></a>**[C3]** FlexGuard: Continuous Risk Scoring for Strictness-Adaptive LLM Content Moderation. Ding et al., 2026-02. ACL 2026；arXiv. <https://arxiv.org/abs/2602.23636>
- <a id="c4"></a>**[C4]** SingGuard: A Policy-Adaptive Multimodal LLM Guardrail with Dynamic Reasoning. SingGuard Team, 2026-06. arXiv. <https://arxiv.org/abs/2606.22873>
- <a id="c5"></a>**[C5]** When Are Reasoning-Based Guardrails Not Efficient? ResponseGuard: A Fast Vision-Language Guard for Real-Time Moderation. Na, 2026-07. arXiv. <https://arxiv.org/abs/2607.21401>
- <a id="c6"></a>**[C6]** HarmVideoBench: Benchmarking Harmful Video Understanding in Large Multimodal Models. Wu et al., 2026-06. arXiv. <https://arxiv.org/abs/2606.27187>
- <a id="c7"></a>**[C7]** Beyond the Explicit: Benchmarking and Detecting Implicit Harm in Short Videos. Li et al., 2026-11. arXiv. <https://arxiv.org/abs/2611.15306>
- <a id="c8"></a>**[C8]** VLM as Policy: Common-Law Content Moderation Framework for Short Video Platform. Lu et al., 2025-04. arXiv. <https://arxiv.org/abs/2504.14904>
- <a id="c9"></a>**[C9]** BLM-Guard: Explainable Multimodal Ad Moderation with Chain-of-Thought and Policy-Aligned Rewards. Yang et al., 2026-02. AAAI 2026；arXiv. <https://arxiv.org/abs/2602.18193>
- <a id="c10"></a>**[C10]** Summarize, Judge, Refine: Decoupled Content Understanding and Policy Learning for Multimodal Content Moderation. Ahmed et al., 2026-08. arXiv. <https://arxiv.org/abs/2609.22094>
- <a id="c11"></a>**[C11]** From Failure Taxonomy to Intervention: A Diagnostic Methodology for Industry-Scale AVLM in Video and Live-Streaming Platform Moderation. Ye et al., 2026-06. NeurIPS 2026；arXiv. <https://arxiv.org/abs/2606.30059>
- <a id="c12"></a>**[C12]** DUAL-Bench: Measuring Over-Refusal and Robustness in Vision-Language Models. Ren et al., 2025-10. arXiv. <https://arxiv.org/abs/2510.10846>
- <a id="c13"></a>**[C13]** XPlainVerse: A Million-Scale Benchmark for Explainable Deepfake Detection. Narang et al., 2026-07. arXiv. <https://arxiv.org/abs/2607.03562>
- <a id="c14"></a>**[C14]** Policy-as-Prompt: Rethinking Content Moderation in the Age of Large Language Models. Palla et al., 2025-02. arXiv. <https://arxiv.org/abs/2502.18695>
- <a id="c15"></a>**[C15]** It is not enough to give your moderation rules to ChatGPT: Policy-as-Prompt Moderation and Its Potential Impacts on Community Governance. Neumann et al., 2026-07. MuC 2026 研讨会；arXiv. <https://arxiv.org/abs/2607.12149>
- <a id="c16"></a>**[C16]** LlavaGuard: An Open VLM-based Framework for Safeguarding Vision Datasets and Models. Helff et al., 2024-06. ICML 2025；arXiv. <https://arxiv.org/abs/2406.05113>
- <a id="c17"></a>**[C17]** ShieldGemma 2: Robust and Tractable Image Content Moderation. Zeng et al., 2025-04. arXiv. <https://arxiv.org/abs/2504.01081>
- <a id="c18"></a>**[C18]** GuardReasoner-VL: Safeguarding VLMs via Reinforced Reasoning. Liu et al., 2025-05. arXiv. <https://arxiv.org/abs/2505.11049>
- <a id="c19"></a>**[C19]** SafeWatch: An Efficient Safety-Policy Following Video Guardrail Model with Transparent Explanations. Chen et al., 2024-12. arXiv. <https://arxiv.org/abs/2412.06878>
- <a id="c20"></a>**[C20]** AdDriftBench: A Benchmark for Detecting Data Drift and Label Drift in Short Video Advertising. Song et al., 2025-11. Findings of EMNLP 2025. <https://aclanthology.org/2025.findings-emnlp.545/>
- <a id="d1"></a>**[D1]** Train a Unified Multimodal Data Quality Classifier with Synthetic Data. Wang et al., 2025-10. Findings of EMNLP 2025；arXiv. <https://arxiv.org/abs/2510.15162>
- <a id="d2"></a>**[D2]** 20/20 Vision Language Models: A Prescription for Better VLMs through Data Curation Alone. DatologyAI, 2026-05. arXiv. <https://arxiv.org/abs/2605.11405>
- <a id="d3"></a>**[D3]** Do We Still Need Humans in the Loop? Human vs. LLM Annotation in Active Learning for TikTok Hate Speech Detection. Hakimi et al., 2026-04. arXiv. <https://arxiv.org/abs/2604.13899>
- <a id="d4"></a>**[D4]** Multimodal Judges Flag, Humans Decide: A Review-and-Edit Framework for AI-Generated Product Video Ads. Agarwaal et al., 2026-11. WACV 2027 WVAQ 研讨会；arXiv. <https://arxiv.org/abs/2611.13178>
- <a id="d5"></a>**[D5]** LoRA Without Regret. Thinking Machines Lab, 2025-09. 博客. <https://thinkingmachines.ai/blog/lora/>
- <a id="d6"></a>**[D6]** Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data. Gerstgrasser et al., 2024-04. arXiv. <https://arxiv.org/abs/2404.01413>
- <a id="d7"></a>**[D7]** Synthetic Data Generation with Large Language Models for Text Classification: Potential and Limitations. Li et al., 2023-10. EMNLP 2023；arXiv. <https://arxiv.org/abs/2310.07849>
- <a id="d8"></a>**[D8]** On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey. Long et al., 2024-06. arXiv. <https://arxiv.org/abs/2406.15126>
- <a id="d9"></a>**[D9]** Best Practices and Lessons Learned on Synthetic Data. Liu et al., 2024-04. COLM 2024；arXiv. <https://arxiv.org/abs/2404.07503>
- <a id="d10"></a>**[D10]** On the Diversity of Synthetic Data and its Impact on Training Large Language Models. Chen et al., 2024-10. arXiv. <https://arxiv.org/abs/2410.15226>
- <a id="d11"></a>**[D11]** Unveiling the Flaws: Exploring Imperfections in Synthetic Data and Mitigation Strategies for Large Language Models. Chen et al., 2024-06. arXiv. <https://arxiv.org/abs/2406.12397>
- <a id="d12"></a>**[D12]** Better as Generators Than Classifiers: Leveraging LLMs and Synthetic Data for Low-Resource Multilingual Classification. Pecher et al., 2026-01. Findings of EACL 2026；arXiv. <https://arxiv.org/abs/2601.16278>
- <a id="d13"></a>**[D13]** Fine-Tuned 'Small' LLMs (Still) Significantly Outperform Zero-Shot Generative AI Models in Text Classification. Bucher & Martini, 2024-06. arXiv. <https://arxiv.org/abs/2406.08660>
- <a id="d14"></a>**[D14]** SLM-Mod: Small Language Models Surpass LLMs at Content Moderation. Zhan et al., 2024-10. NAACL 2025；arXiv. <https://arxiv.org/abs/2410.13155>
- <a id="d15"></a>**[D15]** Benchmark Data Contamination of Large Language Models: A Survey. Xu et al., 2024-06. arXiv. <https://arxiv.org/abs/2406.04244>
- <a id="d16"></a>**[D16]** We Need to Talk About Random Splits. Søgaard et al., 2020-05. EACL 2021；arXiv. <https://arxiv.org/abs/2005.00636>
- <a id="d17"></a>**[D17]** Memorization vs. Generalization: Quantifying Data Leakage in NLP Performance Evaluation. Elangovan et al., 2021-02. EACL 2021；arXiv. <https://arxiv.org/abs/2102.01818>
- <a id="d18"></a>**[D18]** A Careful Examination of Large Language Model Performance on Grade School Arithmetic. Zhang et al., 2024-05. NeurIPS 2024 Datasets and Benchmarks；arXiv. <https://arxiv.org/abs/2405.00332>
- <a id="e1"></a>**[E1]** Transforming content enforcement with AI. Meta Transparency Center, 2026-03. 网页. <https://transparency.meta.com/enforcement/detecting-violations/transforming-content-enforcement-with-AI/>
- <a id="e2"></a>**[E2]** Integrity Reports, H2 2026. Meta Transparency Center, 2026-10. 网页. <https://transparency.meta.com/reports/integrity-reports-h2-2026/>
- <a id="e3"></a>**[E3]** 2025 Ads Safety Report. Google, 2026-04. 博客. <https://blog.google/products/ads-commerce/2025-ads-safety-report/>
- <a id="e4"></a>**[E4]** Digital Services Act: our seventh transparency report on content moderation in Europe. TikTok Newsroom, 2026-08. 新闻页. <https://newsroom.tiktok.com/digital-services-act-our-seventh-transparency-report-on-content-moderation-in-europe?lang=en-150>
- <a id="e5"></a>**[E5]** Decision Quality Evaluation Framework at Pinterest. Tian et al. (Pinterest), 2026-02. arXiv. <https://arxiv.org/abs/2602.15809>
- <a id="e6"></a>**[E6]** Measuring the Prevalence of Policy Violating Content with ML Assisted Sampling and LLM Labeling. Dobi et al. (Pinterest), 2026-02. arXiv. <https://arxiv.org/abs/2602.18518>
- <a id="f1"></a>**[F1]** 从 Hugging Face Hub 接口读取的模型和数据集许可证、门控和模型卡信息，以及 GitHub 仓库许可证，2026-11-17 核对. Hugging Face; GitHub, 2026-11. Hugging Face Hub 接口. <https://huggingface.co/docs/hub/api>
