# Vision, Multimodal Understanding and Content-Safety Recognition: November 2025 to November 2026

[中文](REPORT.zh-CN.md)

Scope: public work on computer vision, multimodal understanding and content-safety recognition from November 2025 to November 2026, and what it means for recognising short-video and ad content and for automated data production.

## 0. Summary

- **Vision backbones are getting better at fine detail and cheaper on redundant pixels at the same time.** SigLIP-HD trains mid-resolution features to imitate high-resolution ones, keeping the inference budget unchanged, with the largest gains on optical character recognition (OCR) tasks [[A5]](#a5); OneVision-Encoder spends compute only on the 3.1%–25% of regions that carry the most information [[A4]](#a4); SAM 3 takes a noun phrase or an example image as a prompt and segments and tracks every matching object in images and video [[A2]](#a2). Text-heavy ads and pre-labelling both benefit directly.
- **Small open multimodal models now take video natively, but longer reasoning is not automatically more accurate.** Qwen3-VL ships dense 2B and 4B versions [[B2]](#b2), Gemma 4 starts at 2.3B [[B4]](#b4), and most of the 2B–4B checkpoints listed here are Apache-2.0 and download without an access request [[F1]](#f1). Long reasoning chains make models gradually stop looking at the image [[B11]](#b11), and thinking-mode reasoning leans heavily on subtitles and other text cues [[B6]](#b6).
- **Safety recognition is moving from fixed-taxonomy guard models to policy-as-input guards.** SingGuard reads natural-language rules at run time, checks content against them one by one and names the rule that fired [[C4]](#c4); FlexGuard outputs a continuous risk score so each platform can set its own threshold [[C3]](#c3); ResponseGuard's 2B single-pass guard beats a recent 3B reasoning guard on response-harm detection at about 150 times lower time cost [[C5]](#c5). Dedicated short-video ad methods exist [[C9]](#c9) [[C10]](#c10), but as of 2026-11-17 no openly downloadable benchmark targeting short-video ad policy specifically could be found [[C20]](#c20) [[F1]](#f1).
- **Data production is converging on a models-flag, people-decide pattern.** In a product-video ad review study, reviewers who saw model findings marked the product wrong on 58% of scenes across 60 ads, against 26% for two reviewers who saw the same ads with findings hidden [[D4]](#d4); on 278K TikTok comments, classifiers trained on large language model (LLM) labels beat human-supervised ones at roughly one-tenth the cost, while people remain the evaluators [[D3]](#d3). Changing only the training data, with architecture and compute fixed, lifted a vision-language model (VLM) by 11.7 points on average across 20 public benchmarks [[D2]](#d2).
- **Company reports show automation already handles most enforcement, and wrongful decisions are tracked as a metric to push down.** Google says its systems caught over 99% of policy-violating ads before they served and cut incorrect advertiser suspensions by 80% [[E3]](#e3); TikTok says automated systems actioned 94.1% of violating content without human review [[E4]](#e4); Meta says LLMs make 13% fewer enforcement mistakes than humans on average [[E2]](#e2).

## 1. Scope and method

Beyond that window, a few earlier foundational works are cited for context. The sources are papers (mostly on arXiv), model cards and code repositories, and transparency reports and blog posts published by companies. Every arXiv ID, first-version date and abstract was retrieved through the official arXiv API (export.arxiv.org) on 2026-11-17, and the text states only what the abstracts say. Web pages were opened one by one and quoted; licences and access gating for models and datasets were read from the Hugging Face API [[F1]](#f1). Company figures are reported as stated, with nothing derived from them.

## 2. Vision foundations

### 2.1 Lineage

SigLIP 2, released in February 2025, is the usual reference point. It adds captioning-based pretraining, self-supervised losses and online data curation to SigLIP's image-text objective, improves localisation and dense prediction, and comes in four sizes: ViT-B (86M), L (303M), So400m (400M) and g (1B) [[A1]](#a1). Around the same time, Perception Encoder showed that contrastive image-text training alone yields general-purpose embeddings, but that the best ones sit in intermediate layers and need alignment to be drawn out [[A9]](#a9); DINOv3 took the self-supervised route and introduced Gram anchoring to stop dense features degrading over long training [[A8]](#a8).

Recent work has moved along these lines:

- **Finer detail at the same resolution.** SigLIP-HD builds fine-to-coarse supervision on SigLIP 2: the coarse features of a mid-resolution image are trained to mimic the fine features of its high-resolution version, at exactly the same inference budget, with the strongest gains on OCR tasks [[A5]](#a5).
- **Compute only where video changes.** OneVision-Encoder borrows from video codecs and processes only the 3.1%–25% of regions rich in signal entropy; plugged into an LLM, it beats Qwen3-ViT and SigLIP2 across 16 image, video and document benchmarks with fewer visual tokens and less pretraining data, and averages 4.1% better than Qwen3-ViT on video tasks [[A4]](#a4).
- **Dedicated vision capacity is what counts.** Unbundling compares encoder designs from scratch at matched data and compute. It finds that a pretrained visual prior is not necessary, since an encoder trained jointly with the backbone performs on par; what matters is dedicated vision parameters, which can sit in the backbone's early layers with comparable results and a shorter time to first token [[A6]](#a6).

### 2.2 Segmentation and detection

SAM 3 extends Segment Anything to promptable concept segmentation (PCS): the prompt can be a noun phrase, an image exemplar or both, and the model returns masks and identities for every matching instance, with images and video sharing one backbone [[A2]](#a2). Its data engine produced 4M unique concept labels, including hard negatives (examples that look like the concept but are not), and it doubles the accuracy of existing systems on image and video PCS [[A2]](#a2). The repository lists 848M parameters and 75-80% of human performance on the SA-Co benchmark, which contains 270K unique concepts; SAM 3.1 Object Multiplex followed on 27 March 2026 [[A3]](#a3). The checkpoints require an access request and use the SAM License [[A3]](#a3) [[F1]](#f1).

LED takes a different route: it feeds hidden states from the language-model layers of a multimodal LLM (MLLM) into an open-vocabulary detector, which finds objects for arbitrary text categories, through a zero-initialised cross-attention adapter [[A7]](#a7). With Swin-T as the vision encoder, Qwen2-0.5B plus LED lifts GroundingDINO by 3.82% on OmniLabel at 8.7% extra compute (GFLOPs), and a larger vision backbone raises the gain to 6.22% [[A7]](#a7). Adapting only the early LLM layers captures most of the benefit [[A7]](#a7).

### 2.3 What this means for content recognition

- **Text-heavy ads.** In short-video ads the evidence is spread across visuals, speech and subtitles [[C9]](#c9), and public moderation data stores on-screen text as a separate OCR field [[F1]](#f1). SigLIP-HD shows OCR-style tasks can improve without raising resolution [[A5]](#a5), and SigLIP 2's multilingual training helps with copy in many languages [[A1]](#a1).
- **More frames for the same compute.** Encoding only the regions that change [[A4]](#a4) makes it cheaper to look at a whole video rather than a cover image and a handful of frames.
- **Pre-labelling from a noun phrase.** SAM 3 turns a short noun phrase into masks for every instance, ready for a person to confirm [[A2]](#a2); its hard negatives correspond to the look-alike but compliant cases that matter in review [[A2]](#a2).
- **Adding categories quickly.** LED gives a detector extra semantics from a small language model [[A7]](#a7), which suits settings where new object categories have to be added fast.

## 3. Multimodal understanding

### 3.1 Open model families

| Family | Released sizes | Relevant features |
|---|---|---|
| Qwen3-VL [[B2]](#b2) | Dense 2B/4B/8B/32B; mixture-of-experts (MoE) 30B-A3B/235B-A22B | Native interleaved text-image-video context of up to 256K tokens; text-based timestamp alignment for more precise temporal grounding in video |
| Qwen3.5 [[B3]](#b3) | First open-weight model Qwen3.5-397B-A17B: 397 billion total parameters, 17 billion active per forward pass; small sizes in the table below | Language and dialect support expanded from 119 to 201 |
| GLM-4.1V-Thinking / GLM-4.5V / GLM-4.6V [[B1]](#b1) | GLM-4.1V-9B-Thinking and GLM-4.5V open-sourced, plus the GLM-4.6V series | Reinforcement Learning with Curriculum Sampling (RLCS); GLM-4.5V evaluated on 42 public benchmarks; GLM-4.6V adds native tool use and a 128K context window |
| Gemma 4 [[B4]](#b4) | 2.3B to 31B, dense and MoE | Improved vision and audio encoders at every size; an encoder-free 12B model that ingests raw audio and image patches; a thinking mode |
| Qwen3-Omni [[B5]](#b5) | Qwen3-Omni-30B-A3B family, Apache 2.0 | Text, image, audio and video in a single model; open-source state of the art on 32 of 36 audio and audio-visual benchmarks; theoretical cold-start first-packet latency of 234 ms |
| InternVL3.5 [[B12]](#b12) | Up to InternVL3.5-241B-A28B | Cascade reinforcement learning; a Visual Resolution Router (ViR); 4.05× inference speedup over InternVL3 |

Models that hear as well as see [[B5]](#b5) matter for ad review: a voice-over claim is evidence just like a subtitle or an image, and a mismatch between modalities is itself something review has to catch [[C9]](#c9).

### 3.2 Video understanding

**Benchmarks now reward consistency, not single right answers.** Video-MME-v2 notes that existing video benchmarks are saturating and leaderboard scores have drifted from real capability; instead of per-question accuracy it scores groups of related questions, giving credit only when answers are consistent and multi-step reasoning is coherent, so lucky guesses earn nothing [[B6]](#b6). It was built by 12 annotators and 50 independent reviewers over 3,300 human-hours with up to 5 rounds of quality assurance [[B6]](#b6). The best current model, Gemini-3-Pro, still trails human experts substantially, and "thinking-based reasoning is highly dependent on textual cues": it helps with subtitles and sometimes hurts in purely visual settings [[B6]](#b6).

**From watching offline to answering live.** RIVER turns evaluation into real-time dialogue with retrospective memory, live perception and proactive anticipation tasks; offline models do well on single questions but struggle with real-time processing [[B7]](#b7). Streamo is trained on Streamo-Instruct-465K to handle real-time narration, action understanding, event captioning, temporal grounding and time-sensitive questions in one model [[B8]](#b8).

**Why video is still expensive.** A survey points out that the compute and memory cost of video LLMs grows with frame count and context length, which limits real-time, mobile and resource-constrained deployment [[B9]](#b9). It organises efficiency methods by pipeline stage (frame sampling, modality encoding, connector-level token reduction, LLM prefilling and decoding) and identifies audio-visual efficiency and standardised evaluation as open gaps [[B9]](#b9).

**Pinpointing the moment.** TaRO builds timestamped reasoning paths from pre-generated dense captions and uses a temporal-sensitivity reward to check that reasoning is anchored to event boundaries, reaching state-of-the-art results on video temporal grounding (VTG) benchmarks [[B10]](#b10). In ad review this is the ability to say at which second a claim appears.

### 3.3 Reasoning and reinforcement learning

The GLM team first builds a strong vision foundation through large-scale pre-training (which, they argue, sets the upper bound on final performance), then uses curriculum-sampled reinforcement learning (RL) to extend it to video understanding, content recognition, grounding and long documents [[B1]](#b1). The smaller GLM-4.1V-9B-Thinking beats the much larger Qwen2.5-VL-72B on 29 benchmarks [[B1]](#b1).

The flip side comes from the VAPO study, which finds that multimodal reasoning strengthens logical inference but can gradually erode perceptual grounding, causing failures on basic visual questions; the authors call this visual forgetting, where longer reasoning makes the model increasingly ignore the visual input [[B11]](#b11). Their Vision-Anchored Policy Optimization steers reasoning back to visually grounded trajectories, producing VAPO-Thinker-7B [[B11]](#b11).

### 3.4 Open 2B–4B checkpoints that fit an 8 GB laptop GPU

Licences and gating were checked on Hugging Face on 2026-11-17; the memory column is an estimate from weight size plus a 16-frame input, and bf16 means unquantised half-precision weights [[F1]](#f1).

| Checkpoint | Licence | Gated | Native video input | 8 GB GPU (estimate) |
|---|---|---|---|---|
| Qwen/Qwen3-VL-2B-Instruct | Apache-2.0 | No | Yes | Runs in bf16 |
| Qwen/Qwen3-VL-4B-Instruct | Apache-2.0 | No | Yes | Needs 4-bit quantisation |
| Qwen/Qwen3.5-2B | Apache-2.0 | No | Yes | Runs in bf16 |
| Qwen/Qwen3.5-4B | Apache-2.0 | No | Yes | Needs 4-bit quantisation |
| OpenGVLab/InternVL3_5-2B-HF | Apache-2.0 | No | Yes | Runs in bf16 |
| OpenGVLab/InternVL3_5-4B-HF | Apache-2.0 | No | Yes | Needs 4-bit quantisation |
| HuggingFaceTB/SmolVLM2-2.2B-Instruct | Apache-2.0 | No | Yes | Runs in bf16 |
| google/gemma-4-E2B-it (2.3B effective, 5.1B with embeddings) | Apache-2.0 | No | As frames, up to 60 seconds at one frame per second | Needs 4-bit quantisation |
| allenai/Molmo2-4B | Apache-2.0 | No | Yes | Needs 4-bit quantisation |
| Qwen/Qwen2.5-VL-3B-Instruct | qwen-research | No | Yes | — |
| google/gemma-3-4b-it | Gemma | Yes, manual approval | Frames only | — |

## 4. Content-safety recognition

### 4.1 From fixed taxonomies to policy as input

Earlier open guard models were mostly trained on fixed risk categories. ShieldGemma 2 is a 4B image moderation model built on Gemma 3, covering key harms such as sexually explicit content, violence and gore, and dangerous content [[C17]](#c17). LlavaGuard already allowed a customisable safety taxonomy, with models from 0.5B to 7B trained on expert labels giving each image a safety rating, category and rationale [[C16]](#c16). GuardReasoner-VL used online RL to make the guard reason before deciding, trained on 123K samples and 631K reasoning steps, and beat the runner-up by 19.27% F1 (a score balancing precision and recall) on average [[C18]](#c18). For video, SafeWatch encodes each policy chunk in parallel and keeps the visual tokens most relevant to each policy, so it can follow customised safety policies [[C19]](#c19).

In the most recent work, policy as input has become the main thread:

- **Bring your own policy.** gpt-oss-safeguard classifies text content against a safety policy the user writes, with adjustable reasoning effort (low, medium, high); gpt-oss-safeguard-20b has 21B parameters with 3.6B active, fits on GPUs with 16GB of VRAM, and is released under Apache 2.0 [[C1]](#c1).
- **All four modalities.** GuardReasoner-Omni moderates text, image, video and audio, trained on 181k samples: supervised fine-tuning (SFT) first teaches it to reason before deciding, then RL with a concise correctness reward suppresses redundant generation; it comes in 3B and 7B sizes [[C2]](#c2).
- **Adjustable strictness.** FlexGuard starts from the observation that platforms define harm more or less strictly, and that this shifts over time [[C3]](#c3). Its FlexBench evaluates moderators under several strictness regimes and finds that models doing well under one can degrade substantially under another; FlexGuard instead outputs a calibrated continuous risk score and picks a threshold for the target strictness at deployment [[C3]](#c3).
- **Rule-by-rule checks.** SingGuard treats the active policy as a run-time input, checks the content against it rule by rule, and predicts both the safety label and the triggered rule, with inference modes from fast direct judgements to slow policy-grounded deliberation [[C4]](#c4). Its benchmark spans 80+ fine-grained risk types, including cross-modal cases where each modality is harmless alone but the combination is not; SingGuard has the best average F1 in all six benchmark families (35 datasets), and under run-time policy shifts its policy-following accuracy rises from 0.6465 to 0.7415 [[C4]](#c4).
- **Does a guard need to reason?** ResponseGuard reads its verdict from a single pooled representation of the request, the response and the image in one forward pass [[C5]](#c5). On a standard multimodal guardrail benchmark, the 2B ResponseGuard beats a recent 3B reasoning-based guard on response harmfulness without any reasoning and at about 150 times lower time cost; on request harmfulness the reasoning guard keeps an overall lead, and the remaining gap on both tracks sits in the image-only cells [[C5]](#c5). The authors suggest that gap may come from the frozen vision encoders both designs share; single-pass detection also lets the guard screen a streaming answer sentence by sentence [[C5]](#c5).

On open weights, GuardReasoner-VL-3B is Apache-2.0 and ungated, ShieldGemma 2 requires manual approval, and the GuardReasoner-Omni-3B model page carries no licence tag while its base model, Qwen2.5-Omni-3B, uses the qwen-research licence [[F1]](#f1) [[C17]](#c17).

### 4.2 Short-video and ad moderation

- **KuaiMod: common-law moderation.** Working in Kuaishou scenarios, this project annotated the first short-video moderation benchmark with real user and reviewer feedback, and proposed the KuaiMod framework, which uses a VLM with chain-of-thought (CoT) reasoning to model video toxicity from sparse user feedback so that moderation policy can be updated quickly, and has been deployed on Kuaishou [[C8]](#c8).
- **AdDriftBench: drift in ad review.** Built specifically for compliance review of short-video ads, it contains 3,480 short video ads: 2,280 labelled under data-drift scenarios (fast-changing content distributions) and an additional 1,200 under label-drift scenarios (review guidelines that become more detailed over time and move category boundaries) [[C20]](#c20). Across 16 open-source VLMs, models perform moderately, especially on fine-grained semantics and shifting instructions; the paper says the dataset will be made publicly available [[C20]](#c20).
- **BLM-Guard: policy-aligned ad moderation.** Multimodal ads on short-video platforms need finer-grained, policy-driven review of their visuals, speech and subtitles than community safety filters provide [[C9]](#c9). BLM-Guard synthesises scene descriptions, reasoning chains and labels with rules to cut annotation cost, refines the model with RL on a reward balancing causal coherence and policy adherence, and models both single-modality issues (such as exaggerated imagery) and cross-modal mismatches (such as subtitles drifting from speech) [[C9]](#c9).
- **Summarize-Judge-Refine (SJR): understand first, then judge.** A multimodal content model writes a structured text summary and a text-only policy model classifies it against policy definitions, which avoids retraining the whole pipeline for every policy change [[C10]](#c10). An iterative co-training loop uses GRPO (a reinforcement learning algorithm) to make summaries more policy-relevant, and text-space augmentation generates adversarial summary variants, something not possible on raw media [[C10]](#c10). On misleading-ad detection, SJR improves non-misleading F1, the score for compliant ads, by 23.6% relative to a zero-shot chain-of-thought baseline; a variant trained on zero real violating examples, with all positive data synthetic, comes within 0.2% relative of the full-data model on violating F1 [[C10]](#c10). The authors conclude that "new policies can launch without any real violation data" [[C10]](#c10).
- **From failure taxonomy to intervention.** A diagnostic study of industry-scale video and live-stream moderation notes that platforms must adapt to their own data distributions, policy objectives and product safety constraints, which generic public models and external APIs struggle to meet; it maps model failures into observable failure signatures, links each class to candidate interventions, and applies this to an audio-visual-language model (AVLM) for a platform supporting over 100 regions [[C11]](#c11).
- **Explaining why a video is harmful.** HarmVideoBench pairs 1,379 videos with 4,137 multiple-choice questions across three levels (observable evidence, clip-internal meaning, beyond-clip reasoning) and evaluates 19 models; the authors' BCR method retrieves context only when needed and raises the macro average from 61.7 percent to 84.4 percent [[C6]](#c6).
- **Implicit harm.** DeepHarm-7K targets short videos that promote misleading values through narrative, emotional framing or cross-modal cues, with 7,110 samples labelled under a fine-grained taxonomy; its DeepHarm-VL framework uses two rounds of reasoning, finding surface evidence first and then inferring latent meaning [[C7]](#c7).

### 4.3 Over-refusal

Safety mechanisms can overshoot: a model refuses a benign request out of excess caution, which is over-refusal, while the ideal is safe completion, doing the harmless part of the request and flagging any risk explicitly [[C12]](#c12). DUAL-Bench evaluated 18 VLMs across 12 hazard categories under meaning-preserving visual perturbations; the best model, GPT-5-Nano, reached 12.9% safe completion, with the GPT-5 and Qwen families averaging 7.9% and 3.9% [[C12]](#c12). In ad review, over-refusal is a compliant ad being rejected, which is why the recommendations below treat it as its own metric.

### 4.4 Explaining generated and manipulated media

Deepfake detectors increasingly explain their decisions in natural language, but the explanations are often weakly grounded in the actual manipulation artefacts [[C13]](#c13). XPlainVerse contains one million real and manipulated images, pairing authentic images from five sources with forgeries from twelve off-the-shelf editing and synthesis models, and its multi-stage Edit-Check filter verifies that each edit did what it was meant to [[C13]](#c13). New metrics, EntityScore and EvidenceScore, check whether an explanation names the manipulated entities and visual evidence, and human annotations on 2,000 explanation pairs validate the data [[C13]](#c13). For ad creatives made with generative tools, explanations that point at what was changed help reviewers confirm faster.

### 4.5 Governing policy-as-prompt

LLMs can take policy text directly as input, removing the step of turning policies into annotation guidelines and labelling large datasets [[C14]](#c14). The Policy-as-Prompt paper lists five challenges across four domains: translating policy into prompts, sensitivity to prompt structure and formatting, the risk of technological determinism in policy formation, the changing roles of policy and machine-learning teams, and model governance and accountability [[C14]](#c14). A follow-up argues that handing the rules to an LLM is not enough and prompts need governance of their own, concluding that "writing prompts alone is not appropriate for ensuring meaningful community governance" [[C15]](#c15).

### 4.6 Public benchmarks at a glance

Access and licences were checked on 2026-11-17 [[F1]](#f1).

| Benchmark | Content | Access | Licence |
|---|---|---|---|
| KuaiMod [[C8]](#c8) | Kuaishou short videos: cover plus up to 8 frames, with title, OCR and speech-recognition (ASR) text | No login | CC BY-NC 4.0 on Hugging Face, CC-BY-SA-4.0 on GitHub; the two differ [[F1]](#f1) |
| SafeWatch-Bench [[C19]](#c19) | Over 2M videos, six safety categories | Manual approval | Apache-2.0 on Hugging Face [[F1]](#f1) |
| HarmVideoBench [[C6]](#c6) | 1,379 videos, 4,137 multiple-choice questions | No data link in the abstract | — |
| DeepHarm-7K [[C7]](#c7) | 7,110 short-video samples focused on implicit harm | No data link in the abstract | — |
| AdDriftBench [[C20]](#c20) | 3,480 short video ads, data-drift and label-drift splits | Paper says it will be made public | — |
| BLM-Guard data [[C9]](#c9) | Real short-video ads | Abstract does not mention a data release | — |

As of 2026-11-17 no openly downloadable benchmark covering short-video ad policy specifically could be found: the two ad-specific efforts, AdDriftBench and BLM-Guard, have no downloadable data yet [[C20]](#c20) [[C9]](#c9), and the downloadable KuaiMod covers general short-video moderation [[C8]](#c8) [[F1]](#f1).

## 5. Automated data production

### 5.1 Pre-labelling: models flag, people decide

One study of product video ads made by a generative pipeline splits review into six questions per scene (product, scene quality, pronunciation, script, sound and captions); judges built from vision-language, speech and language models file each finding under the matching question, and reviewers accept or reject each finding and add what the judges missed [[D4]](#d4). Across 120 ads reviewed with judge findings on screen, reviewers marked the product wrong on 60% of scenes [[D4]](#d4). On 60 of those ads they marked the product wrong on 58% of scenes, far more than the 26% marked by two reviewers who later saw the same ads with the findings hidden; one reviewer who saw the ads both ways marked 57% with findings and 19% without [[D4]](#d4). On 120 blind-labelled scenes, a product judge that watches the clip alongside the catalogue photos reached an AUC (a measure of how well it separates flawed from clean scenes) of 0.84, while video-quality and photo-similarity scores were close to chance [[D4]](#d4). The judges are tuned for recall: they catch nearly all defects but also flag many clean scenes, so reviewers decide; without findings, one reviewer's marking faded over a session [[D4]](#d4). The authors sum up the split as "the judges widen what reviewers mark and reviewers add what the judges miss" [[D4]](#d4).

Platform-side evaluation uses the same split. Pinterest uses a high-trust golden set curated by subject matter experts (SMEs) as ground truth and propensity-score sampling to widen coverage, and uses it to compare the cost and performance of LLM agents, optimise prompts with data, and manage policy changes [[E5]](#e5). Its prevalence metric is the share of a day's impressions that went to violating content: daily probability samples are drawn with machine-learning-assisted weights that concentrate label budget on high-exposure, high-risk content, labelled by a multimodal LLM governed by policy prompts and gold-set validation, and turned into unbiased estimates with confidence intervals that can be broken down by surface, geography and other segments [[E6]](#e6).

### 5.2 LLM labels and human labels

A study trained hate-speech classifiers on 278K German political TikTok comments and compared human and LLM annotation under matched conditions [[D3]](#d3). LLM annotation at scale beat human-supervised classifiers at roughly one-tenth the cost, for both closed GPT-5.2 and open-weight Qwen3.5-122B-A10B, and the advantage held under soft-label evaluation [[D3]](#d3). It depends on the interface: only a two-question decomposition mirroring the human task unlocked it [[D3]](#d3). In a prefiltered pool, active learning gave no reliable advantage over random sampling [[D3]](#d3). Error patterns differ by model: only GPT-5.2 matched the human balance of false positives and false negatives, while other variants over-flagged border-control and economic-competition discourse [[D3]](#d3). The authors conclude that "Humans remain essential as evaluators", and that for training labels the question becomes which LLM, which interface and what shape of pool [[D3]](#d3).

### 5.3 Synthetic data: what it can do and where it stops

**What it can do.** Small models trained on synthetic data can beat the model that generated it: across 11 languages and 4 classification tasks, even small amounts of synthetic data let smaller models outperform the large generator, especially in low-resource languages [[D12]](#d12). SJR's variant with no real violating examples [[C10]](#c10) and BLM-Guard's rule-driven synthesis [[C9]](#c9) both use synthetic data to bootstrap moderation models.

**Where it stops.** The more subjective the task, at both task and instance level, the worse models trained on synthetic data perform [[D7]](#d7). The uniform format of synthetic data can cause pattern overfitting and shift the output distribution, weakening instruction following [[D11]](#d11). In controlled experiments with 350M and 1.4B parameter models, synthetic-data diversity correlates positively with both pre-training and fine-tuning performance, and diversity during pre-training affects supervised fine-tuning more than pre-training itself [[D10]](#d10). A survey organises the field around a generate, curate and evaluate workflow [[D8]](#d8), and a best-practice review stresses factuality, fidelity and unbiasedness [[D9]](#d9).

**Model collapse.** Replacing the original real data with each generation's synthetic data tends towards model collapse, whereas accumulating successive synthetic generations alongside the original real data avoids it; in an analytically tractable linear setting, accumulation keeps test error under a finite bound that does not depend on the number of iterations [[D6]](#d6).

### 5.4 Contamination and splits

Benchmark data contamination happens when a model's training data inadvertently includes evaluation material, making results unreliable [[D15]](#d15). Overlap between training and test sets inflates scores and mistakes memorisation for generalisation [[D17]](#d17). Random splits, like standard splits, give overly optimistic estimates, and even deliberately biased worst-case splits often under-estimate the error on new in-domain samples; the authors recommend multiple independent test sets or, failing that, multiple biased splits [[D16]](#d16). GSM1k re-creates GSM8k's style and difficulty with new problems, and some models drop up to 8% in accuracy on it; a model's probability of generating GSM8k examples correlates with its gap between the two sets (Spearman's r^2 = 0.36), though many frontier models show minimal signs of overfitting [[D18]](#d18).

### 5.5 Curation before scale

UniFilter trains an efficient MLLM as a quality classifier for both image-text caption data and interleaved documents [[D1]](#d1). Because labelled multimodal data is hard to collect, it uses a semi-synthetic approach, generating text at four quality levels for existing raw images to create sample-score pairs [[D1]](#d1). Models pre-trained on data it filtered from DataComp and OBELICS show stronger zero-shot reasoning and in-context learning, and the filtered OBELICS-HQ subset is released [[D1]](#d1).

20/20 Vision Language Models holds architecture, training recipe and compute fixed and varies only the data: its curation pipeline lifts a VLM by 11.7 percentage points on average across 20 public benchmarks spanning grounding, visual question answering, OCR and documents, counting, charts, brand identification and more [[D2]](#d2). At 2B, the curated model surpasses InternVL3.5-2B by 9.9 points with about 17 times less training compute, and comes within 1.8 points of Qwen3-VL-2B with about 87 times less compute [[D2]](#d2). Per-capability standard deviation across training seeds drops by about 67%, and at 4B the curated model matches near-frontier accuracy with 3.3 times fewer response FLOPs than Qwen3-VL-4B [[D2]](#d2).

### 5.6 Small task models through fine-tuning

Smaller fine-tuned models (BERT-style) consistently and significantly outperform larger zero-shot prompted models in text classification, compared against ChatGPT (GPT-3.5/GPT-4) and Claude Opus on sentiment, approval and disapproval, emotions and party positions [[D13]](#d13). SLM-Mod fine-tunes models under 15B parameters on 150K comments from 15 popular Reddit communities and finds 11.5% higher accuracy and 25.7% higher recall on average than zero-shot LLMs, with few-shot prompting adding only a marginal gain for the LLMs [[D14]](#d14). On the method itself, LoRA Without Regret finds that for supervised fine-tuning on small-to-medium instruction and reasoning datasets, low-rank adaptation (LoRA) performs the same as full fine-tuning, that the best LoRA learning rate is consistently 10x the full fine-tuning one, and that LoRA can be less tolerant of large batch sizes [[D5]](#d5).

## 6. Industry practice

All figures in this section are self-reported by the companies.

| Company | What they report | Figure | Page date | Source |
|---|---|---|---|---|
| Meta | Languages covered by more advanced AI enforcement systems | Languages spoken by 98% of people online | 2026-03 | [[E1]](#e1) |
| Meta | Detecting violating adult sexual solicitation | AI caught over two times more violating content than people did, while cutting the mistake rate by over 60% | 2026-03 | [[E1]](#e1) |
| Meta | High-profile impersonation | User reports down 80% | 2026-03 | [[E1]](#e1) |
| Meta | LLM versus human enforcement | 13% fewer mistakes on average while finding 10% more actual violations | 2026-10 | [[E2]](#e2) |
| Meta | Prevalence of adult nudity and sexual activity | From 0.10–0.11% in Q4 2025 to 0.07–0.08% in Q2 2026 | 2026-10 | [[E2]](#e2) |
| Meta | Scam ads | 65 million removed from Facebook and Instagram, 94% before anyone reported them | 2026-10 | [[E2]](#e2) |
| Google | Catching violating ads before they serve | Over 99% | 2026-04 | [[E3]](#e3) |
| Google | Annual enforcement volume | Over 8.3 billion ads blocked or removed, 24.9 million accounts suspended | 2026-04 | [[E3]](#e3) |
| Google | Wrongful decisions | Incorrect advertiser suspensions down 80% | 2026-04 | [[E3]](#e3) |
| TikTok | EU Digital Services Act (DSA) transparency report | Automated systems actioned 94.1% of violating content without human review; around 104 million pieces of violating content removed | 2026-08 | [[E4]](#e4) |
| Pinterest | Decision-quality evaluation framework | Methods paper, no headline figure | 2026-02 | [[E5]](#e5) |
| Pinterest | Prevalence measurement system | Methods paper, no headline figure | 2026-02 | [[E6]](#e6) |
| Kuaishou | KuaiMod deployment (large-scale online A/B test) | User reporting rate down 20% | 2025-04 | [[C8]](#c8) |

Two practices are shared:

- **Switch over gradually.** Meta says it moves to AI-first enforcement only once the technology has "consistently performed better than our existing systems in all of our tests" [[E1]](#e1); its October 2026 report credits LLMs deployed to close enforcement gaps for the drop in adult-nudity prevalence [[E2]](#e2).
- **Report wrongful decisions alongside coverage.** Meta reports falling mistake rates [[E1]](#e1) [[E2]](#e2), and Google reports fewer incorrect suspensions and says the majority of Responsive Search Ads created in Google Ads were reviewed instantly [[E3]](#e3). For advertisers, that means faster launches and fewer wrongful rejections.

## 7. What this means

| Scenario | Recommended practice | Evidence |
|---|---|---|
| Short-video ad review: where the evidence lives | Collect evidence separately from the caption and copy, on-screen text (OCR), visuals and audio (voice-over and ASR), and record the source of each decision; check explicitly for cross-modal mismatches such as subtitles that drift from speech | [[C8]](#c8) [[C9]](#c9) [[C10]](#c10) [[B5]](#b5) [[F1]](#f1) |
| Sampling frames under a cost budget | Start with sparse uniform sampling, then sample densely where a product, on-screen text or a spoken claim appears; prefer encoders that read text well at standard resolution and skip static background; consider streaming models for long videos and live streams | [[B9]](#b9) [[A4]](#a4) [[A5]](#a5) [[B7]](#b7) [[B8]](#b8) [[B10]](#b10) |
| Fast guard plus slower policy reasoner | Let a small single-pass guard give every item a continuous risk score; set thresholds per surface and region according to their own strictness; send the middle band to a model that reasons through the policy rule by rule and names the triggered rule | [[C5]](#c5) [[C3]](#c3) [[C4]](#c4) [[C1]](#c1) [[C19]](#c19) |
| Wrongful rejections as a first-class metric | Report miss rate and wrongful-rejection rate side by side; include hard negatives that look violating but are compliant in the test set; track improvement with compliant-class F1 and the rate of wrongful rejections and suspensions | [[C12]](#c12) [[C10]](#c10) [[E3]](#e3) [[E1]](#e1) [[E2]](#e2) |
| Pre-labelling | Use a concept segmenter to produce masks from noun phrases and multimodal judges to list findings against a question checklist; tune judges for recall and let people accept or reject each finding | [[A2]](#a2) [[A3]](#a3) [[D4]](#d4) [[C16]](#c16) |
| Synthetic data for small task classifiers | Split train and test by template, product and writing style rather than by random rows; remove near-duplicates across sets; control format and diversity; keep and accumulate real data, and rely more on real labels for subjective categories | [[D7]](#d7) [[D10]](#d10) [[D11]](#d11) [[D12]](#d12) [[D16]](#d16) [[D17]](#d17) [[D6]](#d6) [[C10]](#c10) |
| Curation before scale | Before adding more data, train or borrow a quality classifier and filter out low-quality samples | [[D1]](#d1) [[D2]](#d2) |
| Gold sets and drift checks | Keep an expert-maintained golden set; estimate prevalence regularly with risk-weighted sampling; after a policy change or a shift in content, re-check on a fresh independent test set | [[E5]](#e5) [[E6]](#e6) [[D18]](#d18) [[D16]](#d16) [[C20]](#c20) [[C3]](#c3) |

### Open questions

1. **Where should reasoning sit?** ResponseGuard works without reasoning for response screening, but the reasoning guard still leads on request screening [[C5]](#c5); long reasoning erodes visual grounding [[B11]](#b11), and thinking-mode reasoning depends on text cues [[B6]](#b6). For short-video ads, where much of the evidence is visual, the benefit and cost of reasoning have not been compared systematically in public.
2. **Once policy becomes a prompt, who guarantees consistency and accountability?** Outcomes are sensitive to prompt structure and formatting, and the roles of policy and machine-learning teams are shifting [[C14]](#c14); writing prompts alone is not enough for meaningful governance [[C15]](#c15); current moderators are inconsistent across strictness regimes [[C3]](#c3), and policy-following accuracy under run-time policy shifts still has room to grow [[C4]](#c4).
3. **How much real labelling can synthetic data and model labels replace?** A model with no real violating examples nearly matches the full-data model [[C10]](#c10), and classifiers trained on LLM labels beat human-supervised ones [[D3]](#d3); but synthetic data helps less as tasks get more subjective [[D7]](#d7), and none of these findings has yet been tested on public short-video ad data [[C20]](#c20).

## 8. How this project's experiments relate

The experiments in [`eval/EVALUATION.md`](../eval/EVALUATION.md) put a few of the points above to a small test: 96 short-video ads rendered for this project, 214 text ads (168 written for Grounded Review and 46 real cases) and 300 real videos from KuaiMod. Each model ran once, on one laptop GPU or through one API, so the numbers show direction rather than general rates.

| Point above | What the experiments found |
|---|---|
| Evidence sits in the caption, the on-screen text and the footage, and should be collected from each (§7) [[C8]](#c8) [[C9]](#c9) [[C10]](#c10) | Given only the caption, claude-haiku-5-5 caught 25 of 64 violating ads; with 16 frames it caught all 64. OCR of the 16 frames passed as text to Qwen3.5-4B caught 57 of 64, but found only 5 of the 16 violations that exist only in the footage, each inferred from the product rather than seen. |
| The cost of video models grows with frame count [[B9]](#b9) | Qwen3.5-4B took a median of 2.4 seconds per ad on the caption plus OCR text, 9.2 seconds on 8 frames and 19.2 seconds on the 16-frame video. A sticker shown for half a second was found in 13 of 20 ads by claude-haiku-5-5 at one frame per second and in 19 of 20 at two. |
| Summarise into text, then judge with a text model [[C10]](#c10) | The cheap route here has the same shape: OCR turns the frames into text and a text classifier judges it. Qwen3-1.7B fine-tuned on synthetic ads was right on 80% of the videos from the caption plus OCR, the same as claude-haiku-5-5 on the same text and 10 points below claude-haiku-5-5 looking at 16 frames (90%). In SJR the summary is written by a multimodal model [[C10]](#c10), which a summary would need in order to cover what only the footage shows. |
| Fine-tuned small models against zero-shot large ones [[D13]](#d13) [[D14]](#d14); small models trained on synthetic data can beat the generator [[D12]](#d12) | Trained only on 1,529 synthetic ads written by claude-haiku-5-5, Qwen3-1.7B with LoRA scored 79% on Grounded Review against 82% for claude-haiku-5-5, and 83% on the 46 real cases against 72% (76% with a longer answer limit). Neither gap is outside its 95% interval. |
| Random or template splits give optimistic estimates [[D16]](#d16) [[D17]](#d17) | The synthetic validation set was held out by product and writing style, and the LoRA models still scored about 12 points lower on Grounded Review's test ads than on it. |
| The uniform format of synthetic data can cause overfitting [[D11]](#d11) | mmBERT-base matched the 1.7B LoRA model on synthetic validation (91%) but caught 15 of 35 violations in the real cases, against 33. Format overfitting is one possible reason; it was not tested. |
| Wrongful rejections as their own metric, with look-alikes in the test set [[C12]](#c12) [[C10]](#c10) | Inputs that caught more also rejected more compliant ads: with 16 frames claude-haiku-5-5 wrongly rejected 9 of 22 look-alikes, Qwen3.5-4B with 8 frames 17. On KuaiMod, adding frames to the text raised violations caught from 73 to 111 of 127 and fine videos flagged from 62 to 70 of 173 for claude-haiku-5-5, and from 58 to 97 and 17 to 45 for Qwen3.5-4B. |
| Platforms must adapt to their own data and rules [[C11]](#c11); KuaiMod's labels follow that platform's rules and feedback [[C8]](#c8) | Zero-shot on KuaiMod, given only the names of the 15 violation types, the best run (Qwen3.5-4B with frames and text) was right on 75% of the videos and flagged 26% of the fine ones; claude-haiku-5-5 reached 71% and flagged 40%. On the rendered ads, whose categories were defined in the prompt, claude-haiku-5-5 was right on 90%. |

All runs had thinking turned off, so the first open question above, where reasoning helps, was not tested.

## 9. References

- <a id="a1"></a>**[A1]** SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features. Tschannen et al., 2025-02. arXiv. <https://arxiv.org/abs/2502.14786>
- <a id="a2"></a>**[A2]** SAM 3: Segment Anything with Concepts. Carion et al., 2025-11. arXiv. <https://arxiv.org/abs/2511.16719>
- <a id="a3"></a>**[A3]** SAM 3 repository README (SAM 3.1 Object Multiplex release note). Meta (facebookresearch), 2026-03. GitHub. <https://github.com/facebookresearch/sam3>
- <a id="a4"></a>**[A4]** OneVision-Encoder: Codec-Aligned Sparsity as a Foundational Principle for Multimodal Intelligence. Tang et al., 2026-02. arXiv. <https://arxiv.org/abs/2602.08683>
- <a id="a5"></a>**[A5]** SigLIP-HD by Fine-to-Coarse Supervision. Yang et al., 2026-07. ICLR 2026; arXiv. <https://arxiv.org/abs/2607.09488>
- <a id="a6"></a>**[A6]** Unbundling Multimodal Vision Encoding. Brown et al., 2026-11. arXiv. <https://arxiv.org/abs/2611.17334>
- <a id="a7"></a>**[A7]** LED: LLM Enhanced Open-Vocabulary Object Detection without Human Curated Data Generation. Zhou et al., 2025-03. arXiv. <https://arxiv.org/abs/2503.13794>
- <a id="a8"></a>**[A8]** DINOv3. Siméoni et al., 2025-08. arXiv. <https://arxiv.org/abs/2508.10104>
- <a id="a9"></a>**[A9]** Perception Encoder: The best visual embeddings are not at the output of the network. Bolya et al., 2025-04. arXiv. <https://arxiv.org/abs/2504.13181>
- <a id="b1"></a>**[B1]** GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning. GLM-V Team, 2025-07. arXiv. <https://arxiv.org/abs/2507.01006>
- <a id="b2"></a>**[B2]** Qwen3-VL Technical Report. Bai et al., 2025-11. arXiv. <https://arxiv.org/abs/2511.21631>
- <a id="b3"></a>**[B3]** Qwen3.5: Towards Native Multimodal Agents. Alibaba Cloud (Qwen), 2026-02. Blog. <https://www.alibabacloud.com/blog/qwen3-5-towards-native-multimodal-agents_602894>
- <a id="b4"></a>**[B4]** Gemma 4 Technical Report. Gemma Team, 2026-07. arXiv. <https://arxiv.org/abs/2607.02770>
- <a id="b5"></a>**[B5]** Qwen3-Omni Technical Report. Xu et al., 2025-09. arXiv. <https://arxiv.org/abs/2509.17765>
- <a id="b6"></a>**[B6]** Video-MME-v2: Towards the Next Stage in Benchmarks for Comprehensive Video Understanding. Fu et al., 2026-04. arXiv. <https://arxiv.org/abs/2604.05015>
- <a id="b7"></a>**[B7]** RIVER: A Real-Time Interaction Benchmark for Video LLMs. Shi et al., 2026-03. arXiv. <https://arxiv.org/abs/2603.03985>
- <a id="b8"></a>**[B8]** Streaming Video Instruction Tuning. Xia et al., 2025-12. CVPR 2026; arXiv. <https://arxiv.org/abs/2512.21334>
- <a id="b9"></a>**[B9]** Why Is Video Still So Expensive? A Survey of Inference-Efficiency Mechanisms in Video and Audiovisual LLMs. Steunou et al., 2026-09. arXiv. <https://arxiv.org/abs/2609.10355>
- <a id="b10"></a>**[B10]** Temporal-Aware Reasoning Optimization for Video Temporal Grounding. Zheng et al., 2026-06. ICML 2026; arXiv. <https://arxiv.org/abs/2606.09248>
- <a id="b11"></a>**[B11]** More Thought, Less Accuracy? On the Dual Nature of Reasoning in Vision-Language Models. Tian et al., 2025-09. ICLR 2026; arXiv. <https://arxiv.org/abs/2509.25848>
- <a id="b12"></a>**[B12]** InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency. Wang et al., 2025-08. arXiv. <https://arxiv.org/abs/2508.18265>
- <a id="c1"></a>**[C1]** gpt-oss-safeguard-20b model card. OpenAI, 2026-01 (last updated). Hugging Face. <https://huggingface.co/openai/gpt-oss-safeguard-20b>
- <a id="c2"></a>**[C2]** GuardReasoner-Omni: A Reasoning-based Multi-modal Guardrail for Text, Image, Video, and Audio. Zhu et al., 2026-02. arXiv. <https://arxiv.org/abs/2602.03328>
- <a id="c3"></a>**[C3]** FlexGuard: Continuous Risk Scoring for Strictness-Adaptive LLM Content Moderation. Ding et al., 2026-02. ACL 2026; arXiv. <https://arxiv.org/abs/2602.23636>
- <a id="c4"></a>**[C4]** SingGuard: A Policy-Adaptive Multimodal LLM Guardrail with Dynamic Reasoning. SingGuard Team, 2026-06. arXiv. <https://arxiv.org/abs/2606.22873>
- <a id="c5"></a>**[C5]** When Are Reasoning-Based Guardrails Not Efficient? ResponseGuard: A Fast Vision-Language Guard for Real-Time Moderation. Na, 2026-07. arXiv. <https://arxiv.org/abs/2607.21401>
- <a id="c6"></a>**[C6]** HarmVideoBench: Benchmarking Harmful Video Understanding in Large Multimodal Models. Wu et al., 2026-06. arXiv. <https://arxiv.org/abs/2606.27187>
- <a id="c7"></a>**[C7]** Beyond the Explicit: Benchmarking and Detecting Implicit Harm in Short Videos. Li et al., 2026-11. arXiv. <https://arxiv.org/abs/2611.15306>
- <a id="c8"></a>**[C8]** VLM as Policy: Common-Law Content Moderation Framework for Short Video Platform. Lu et al., 2025-04. arXiv. <https://arxiv.org/abs/2504.14904>
- <a id="c9"></a>**[C9]** BLM-Guard: Explainable Multimodal Ad Moderation with Chain-of-Thought and Policy-Aligned Rewards. Yang et al., 2026-02. AAAI 2026; arXiv. <https://arxiv.org/abs/2602.18193>
- <a id="c10"></a>**[C10]** Summarize, Judge, Refine: Decoupled Content Understanding and Policy Learning for Multimodal Content Moderation. Ahmed et al., 2026-08. arXiv. <https://arxiv.org/abs/2609.22094>
- <a id="c11"></a>**[C11]** From Failure Taxonomy to Intervention: A Diagnostic Methodology for Industry-Scale AVLM in Video and Live-Streaming Platform Moderation. Ye et al., 2026-06. NeurIPS 2026; arXiv. <https://arxiv.org/abs/2606.30059>
- <a id="c12"></a>**[C12]** DUAL-Bench: Measuring Over-Refusal and Robustness in Vision-Language Models. Ren et al., 2025-10. arXiv. <https://arxiv.org/abs/2510.10846>
- <a id="c13"></a>**[C13]** XPlainVerse: A Million-Scale Benchmark for Explainable Deepfake Detection. Narang et al., 2026-07. arXiv. <https://arxiv.org/abs/2607.03562>
- <a id="c14"></a>**[C14]** Policy-as-Prompt: Rethinking Content Moderation in the Age of Large Language Models. Palla et al., 2025-02. arXiv. <https://arxiv.org/abs/2502.18695>
- <a id="c15"></a>**[C15]** It is not enough to give your moderation rules to ChatGPT: Policy-as-Prompt Moderation and Its Potential Impacts on Community Governance. Neumann et al., 2026-07. MuC 2026 workshop; arXiv. <https://arxiv.org/abs/2607.12149>
- <a id="c16"></a>**[C16]** LlavaGuard: An Open VLM-based Framework for Safeguarding Vision Datasets and Models. Helff et al., 2024-06. ICML 2025; arXiv. <https://arxiv.org/abs/2406.05113>
- <a id="c17"></a>**[C17]** ShieldGemma 2: Robust and Tractable Image Content Moderation. Zeng et al., 2025-04. arXiv. <https://arxiv.org/abs/2504.01081>
- <a id="c18"></a>**[C18]** GuardReasoner-VL: Safeguarding VLMs via Reinforced Reasoning. Liu et al., 2025-05. arXiv. <https://arxiv.org/abs/2505.11049>
- <a id="c19"></a>**[C19]** SafeWatch: An Efficient Safety-Policy Following Video Guardrail Model with Transparent Explanations. Chen et al., 2024-12. arXiv. <https://arxiv.org/abs/2412.06878>
- <a id="c20"></a>**[C20]** AdDriftBench: A Benchmark for Detecting Data Drift and Label Drift in Short Video Advertising. Song et al., 2025-11. Findings of EMNLP 2025. <https://aclanthology.org/2025.findings-emnlp.545/>
- <a id="d1"></a>**[D1]** Train a Unified Multimodal Data Quality Classifier with Synthetic Data. Wang et al., 2025-10. Findings of EMNLP 2025; arXiv. <https://arxiv.org/abs/2510.15162>
- <a id="d2"></a>**[D2]** 20/20 Vision Language Models: A Prescription for Better VLMs through Data Curation Alone. DatologyAI, 2026-05. arXiv. <https://arxiv.org/abs/2605.11405>
- <a id="d3"></a>**[D3]** Do We Still Need Humans in the Loop? Human vs. LLM Annotation in Active Learning for TikTok Hate Speech Detection. Hakimi et al., 2026-04. arXiv. <https://arxiv.org/abs/2604.13899>
- <a id="d4"></a>**[D4]** Multimodal Judges Flag, Humans Decide: A Review-and-Edit Framework for AI-Generated Product Video Ads. Agarwaal et al., 2026-11. WACV 2027 WVAQ workshop; arXiv. <https://arxiv.org/abs/2611.13178>
- <a id="d5"></a>**[D5]** LoRA Without Regret. Thinking Machines Lab, 2025-09. Blog. <https://thinkingmachines.ai/blog/lora/>
- <a id="d6"></a>**[D6]** Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data. Gerstgrasser et al., 2024-04. arXiv. <https://arxiv.org/abs/2404.01413>
- <a id="d7"></a>**[D7]** Synthetic Data Generation with Large Language Models for Text Classification: Potential and Limitations. Li et al., 2023-10. EMNLP 2023; arXiv. <https://arxiv.org/abs/2310.07849>
- <a id="d8"></a>**[D8]** On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey. Long et al., 2024-06. arXiv. <https://arxiv.org/abs/2406.15126>
- <a id="d9"></a>**[D9]** Best Practices and Lessons Learned on Synthetic Data. Liu et al., 2024-04. COLM 2024; arXiv. <https://arxiv.org/abs/2404.07503>
- <a id="d10"></a>**[D10]** On the Diversity of Synthetic Data and its Impact on Training Large Language Models. Chen et al., 2024-10. arXiv. <https://arxiv.org/abs/2410.15226>
- <a id="d11"></a>**[D11]** Unveiling the Flaws: Exploring Imperfections in Synthetic Data and Mitigation Strategies for Large Language Models. Chen et al., 2024-06. arXiv. <https://arxiv.org/abs/2406.12397>
- <a id="d12"></a>**[D12]** Better as Generators Than Classifiers: Leveraging LLMs and Synthetic Data for Low-Resource Multilingual Classification. Pecher et al., 2026-01. Findings of EACL 2026; arXiv. <https://arxiv.org/abs/2601.16278>
- <a id="d13"></a>**[D13]** Fine-Tuned 'Small' LLMs (Still) Significantly Outperform Zero-Shot Generative AI Models in Text Classification. Bucher & Martini, 2024-06. arXiv. <https://arxiv.org/abs/2406.08660>
- <a id="d14"></a>**[D14]** SLM-Mod: Small Language Models Surpass LLMs at Content Moderation. Zhan et al., 2024-10. NAACL 2025; arXiv. <https://arxiv.org/abs/2410.13155>
- <a id="d15"></a>**[D15]** Benchmark Data Contamination of Large Language Models: A Survey. Xu et al., 2024-06. arXiv. <https://arxiv.org/abs/2406.04244>
- <a id="d16"></a>**[D16]** We Need to Talk About Random Splits. Søgaard et al., 2020-05. EACL 2021; arXiv. <https://arxiv.org/abs/2005.00636>
- <a id="d17"></a>**[D17]** Memorization vs. Generalization: Quantifying Data Leakage in NLP Performance Evaluation. Elangovan et al., 2021-02. EACL 2021; arXiv. <https://arxiv.org/abs/2102.01818>
- <a id="d18"></a>**[D18]** A Careful Examination of Large Language Model Performance on Grade School Arithmetic. Zhang et al., 2024-05. NeurIPS 2024 Datasets and Benchmarks; arXiv. <https://arxiv.org/abs/2405.00332>
- <a id="e1"></a>**[E1]** Transforming content enforcement with AI. Meta Transparency Center, 2026-03. Web page. <https://transparency.meta.com/enforcement/detecting-violations/transforming-content-enforcement-with-AI/>
- <a id="e2"></a>**[E2]** Integrity Reports, H2 2026. Meta Transparency Center, 2026-10. Web page. <https://transparency.meta.com/reports/integrity-reports-h2-2026/>
- <a id="e3"></a>**[E3]** 2025 Ads Safety Report. Google, 2026-04. Blog. <https://blog.google/products/ads-commerce/2025-ads-safety-report/>
- <a id="e4"></a>**[E4]** Digital Services Act: our seventh transparency report on content moderation in Europe. TikTok Newsroom, 2026-08. Newsroom. <https://newsroom.tiktok.com/digital-services-act-our-seventh-transparency-report-on-content-moderation-in-europe?lang=en-150>
- <a id="e5"></a>**[E5]** Decision Quality Evaluation Framework at Pinterest. Tian et al. (Pinterest), 2026-02. arXiv. <https://arxiv.org/abs/2602.15809>
- <a id="e6"></a>**[E6]** Measuring the Prevalence of Policy Violating Content with ML Assisted Sampling and LLM Labeling. Dobi et al. (Pinterest), 2026-02. arXiv. <https://arxiv.org/abs/2602.18518>
- <a id="f1"></a>**[F1]** Licence, gating and model-card fields read from the Hugging Face Hub API (models and datasets), plus GitHub repository licences, checked on 2026-11-17. Hugging Face; GitHub, 2026-11. Hugging Face Hub API. <https://huggingface.co/docs/hub/api>
