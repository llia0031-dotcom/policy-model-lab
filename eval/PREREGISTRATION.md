# Pre-registration

Each part below was written and frozen before its test run. Times are UTC+8. After a run, changes are only
added under "Changes after the run" at the end of the part; nothing above that line is edited.

## Part 1. Video test (frozen 2026-11-17 08:21, before any test video was sent to a model)

**Items.** 96 short-video ads written for this project (`testset/items_video_zh.yaml`, `items_video_en.yaml`), 48 Chinese and 48 English, each 8 s with four 2-second shots, a subtitle per shot, a post caption and in some items a sticker shown for 0.5 s. 64 violate at least one category (72 labels), 32 are compliant; 22 of the compliant ones are look-alikes (risky words used in an allowed sense, or footage that resembles a violation: tea, water, juice or coffee instead of alcohol, lanterns instead of a flag, two colour options or an unboxing instead of a before-and-after). Where the violation sits: caption 16, subtitle 20 (3 on the cover shot), 0.5 s sticker 20, footage only 16. Answers were fixed when the items were written. The videos are rendered by `testset/render_video.py`; their hashes are in `testset/video_manifest.json`.

**Development items.** Four separate videos (`testset/items_video_dev.yaml`) were used to check that the pipeline runs and that answers parse. They are never scored. After the first development run the prompt was revised once (version 0 listed categories in a way the 2B model copied as a checklist); the revised prompt in `lab/common.py` is the one frozen here.

**Inputs compared** (all with the post caption): text only; cover frame; 8 frames (1 per second); native video (16 frames at 2 fps, Qwen models); 16 frames as images (Claude); OCR text of the 8 frames; OCR text of the 16 frames.

**Models.** Qwen/Qwen3-VL-2B-Instruct (bf16), Qwen/Qwen3.5-4B (NF4 4-bit, thinking off), greedy decoding; claude-haiku-5-5 (thinking off, Message Batches API) on text, cover, 8 frames, 16 frames and OCR of 8 frames; SigLIP 2 so400m zero-shot on 8 and 16 frames (footage only, fixed descriptions in `lab/siglip_frames.py`). One run each.

**Scoring** (`lab/score.py`): decision accuracy (violating vs compliant), violating items caught, compliant items wrongly rejected (look-alikes separately), labels found overall and by location, exact category set, whether the evidence text overlaps the labelled words, unparsed answers (counted as nothing listed), median seconds per item, peak GPU memory.

**Predictions.**
1. Text only finds at most 30% of the 72 labels for each local model (16 labels sit in the caption).
2. The cover frame finds at most 25% of the 16 footage-only labels for each model (2 of them are on the cover shot).
3. For Qwen3.5-4B, 0.5 s stickers found with native video are at least 15 points higher than with 8 frames (8 frames show the sticker in only part of the items; 2 fps always does).
4. With native video, Qwen3.5-4B's decision accuracy is at least 10 points higher than Qwen3-VL-2B's.
5. Qwen3-VL-2B wrongly rejects more compliant items than Qwen3.5-4B, with 8 frames and with native video.
6. For Qwen3.5-4B, OCR of 16 frames + caption finds the caption, subtitle and sticker labels within 10 points of native video, and finds at most 25% of the footage-only labels.
7. claude-haiku-5-5 with 16 frames has the highest decision accuracy of all runs, at least 85%.
8. SigLIP 2 decision accuracy is at most 60% with either frame plan.
9. For both local models, the median time per item with native video is at least three times the median with text only.

**File hashes (SHA-256, first 16 hex characters) at freeze:**

| file | hash |
|---|---|
| testset/items_video_zh.yaml | 7e72a69809e2774d |
| testset/items_video_en.yaml | adef2c017b84d1c4 |
| testset/video_manifest.json | c2835a564b58ffd0 |
| testset/render_video.py | e1dcf079dcd6461f |
| lab/common.py | f2ef40b1d170fac7 |
| lab/frames.py | ded1e5057a13e5a3 |
| lab/local_vlm.py | d8ba2ce43cec00eb |
| lab/siglip_frames.py | 47a03712adf41f8a |
| lab/claude.py | 66a5130faea13274 |
| lab/claude_video.py | 23a669ddfbaa0bd3 |
| lab/score.py | 76f37e80ec0e13d0 |

### Changes after the run (Part 1)

1. `lab/score.py`: answers from the Message Batches API carry no per-item time, and the median-time line failed on them. The line now skips missing times. No scoring rule changed. (Edited 2026-11-17 08:31, after the claude-haiku-5-5 video run, before any score was read.)
2. `lab/score.py`: a partly finished run with no compliant item yet made the table fail on a missing count; every count now starts at zero. No scoring rule changed. (Same time.)
3. `lab/claude.py`: large request sets are split into several batches (see Part 3, change 1). The video batch had already finished in one upload; its answers are unaffected.
4. The claude-haiku-5-5 run on OCR of 8 frames was left out of the first video submission (2026-11-17 08:28), which was sent before the OCR step had run, and was then forgotten. It was sent on 2026-11-18 at about 05:15 with the frozen prompt and settings, after the other video scores had been read.
5. Qwen3-VL-2B's per-item times are not comparable with the other local runs. During its run other programs were using the same GPU and memory use was close to the card's 8 GB (the model itself peaked at 6.6 GB). One 16-frame video took 6.1 seconds in a timing check before the freeze; in the run the median was 71.6 seconds. Its answers are not affected by speed (greedy decoding).

## Part 2. Small-model fine-tuning (frozen 2026-11-17 08:30, before any training run)

**Training data.** 1,529 synthetic ads (766 Chinese, 763 English) in `$POLICY_LAB_DATA/synth/train.jsonl`, built by `finetune/synth.py` from 234 generation requests (claude-haiku-5-5, Message Batches API, USD 0.0557). Each ad carries its own labels from generation; violating words are checked verbatim. 655 are compliant (214 written as look-alikes), 874 violate one or two categories. Ads close to any test item are removed (3 removed). A synthetic validation set of 191 ads comes from held-out (product, style) buckets. The filter rules were changed once before this freeze, after reading only the generation logs (no model had been trained or tested): see DECISIONS D-11.

**Test sets** (never used for training; built by `finetune/make_testsets.py`): grounded-review's public items (168: its dev, test, hard and memory splits, as text; picture-only labels removed), this project's 96 video items as text (caption, subtitles, sticker), grounded-review's 46 real cases (totals only), the synthetic validation set, and the video items as caption + OCR of 16 frames.

**Methods** (same prompt, `finetune/prompt.py`): Qwen3-0.6B and Qwen3-1.7B zero-shot (thinking off) and with LoRA (rank 16, alpha 32, all linear layers, learning rate 2e-4, 3 epochs, effective batch 16, seed 0, last epoch kept); mmBERT-base fine-tuned as a multi-label classifier (learning rate 3e-5, 4 epochs, threshold 0.5); claude-haiku-5-5 zero-shot (thinking off). Greedy decoding. One run each.

`finetune/make_testsets.py` was edited once more before any training (its default folder for grounded-review now points at a folder named `grounded-review`); the test file it builds is unchanged (same hash).

**Predictions** (on grounded-review's 168 items unless stated):
1. Qwen3-1.7B + LoRA has decision accuracy at least 10 points above Qwen3-1.7B zero-shot.
2. Qwen3-1.7B + LoRA has macro-F1 at least 0.15 above Qwen3-1.7B zero-shot.
3. claude-haiku-5-5 has the highest decision accuracy, and Qwen3-1.7B + LoRA is within 10 points of it.
4. Qwen3-0.6B + LoRA has lower decision accuracy than Qwen3-1.7B + LoRA.
5. mmBERT wrongly rejects more look-alikes than Qwen3-1.7B + LoRA.
6. On the synthetic validation set the LoRA models reach at least 90% decision accuracy, and their grounded-review accuracy is at least 10 points lower than that (the synthetic ads are easier than the grounded-review items).
7. On the 46 real cases, every fine-tuned model has lower decision accuracy than claude-haiku-5-5.
8. Qwen3-1.7B + LoRA needs less than 0.5 s per item (median, batched on the laptop GPU).

**File hashes (SHA-256, first 16 hex characters) at freeze:**

| file | hash |
|---|---|
| data_public/text_tests.jsonl | f6fcc211974714dd |
| $POLICY_LAB_DATA/synth/train.jsonl | 049525e580b39a1d |
| $POLICY_LAB_DATA/synth/val.jsonl | 8a5dc559bcd52101 |
| finetune/make_testsets.py | fbb5b80b2ba3b980 |
| finetune/synth.py | 048ff134d6e5dc5e |
| finetune/prompt.py | 46f5bcac9cfcffd2 |
| finetune/train_lora.py | 714e7275553e8da6 |
| finetune/train_encoder.py | a73e39b93c00a237 |
| finetune/predict.py | 03da04944fd65aed |
| finetune/score_text.py | b772b366249f9d50 |

### Changes after the run (Part 2)

1. The first Qwen3-1.7B training run was stopped by the operating system during its third epoch (2026-11-17
   about 18:40), before the adapter was saved. It was started again from the beginning with the same script,
   data and seed (21:58). Its losses differ slightly from the stopped run (epoch 1 training loss 0.1554 against
   0.1583), because GPU arithmetic is not bit-for-bit repeatable. Only the second run is reported.
2. `finetune/predict.py` and `finetune/score_text.py`: an extra run `haiku-400` was added after the scores of the
   preregistered `haiku` run had been read. With the frozen answer limit of 120 tokens, 37 of claude-haiku-5-5's
   597 answers reasoned before giving the list and were cut off, and 32 could not be parsed (15 of the 46 real
   cases). `haiku-400` sends the same prompts with a limit of 400 tokens. The preregistered `haiku` run stays the
   one the predictions are checked against; `haiku-400` is reported beside it. (2026-11-17 22:55.)
3. `finetune/score_text.py`: when a test set's labels are not in the copy being scored (the 46 real cases are
   never published), the table keeps that set's stored totals instead of dropping them, so the offline rerun in
   the public repository rebuilds the same table, and it writes its JSON in a fixed order. No scoring rule changed. A new script, `finetune/compare.py`,
   reports the paired differences between methods with bootstrap intervals; it was written after the scores had
   been read. (2026-11-18 06:00.)
4. Wording only: the end of prediction 6 above now names the grounded-review items instead of describing how
   they were written. The prediction and how it is checked are unchanged. (2026-11-18 06:15.)

## Part 3. Real-data check on KuaiMod (frozen 2026-11-17 08:35, before any KuaiMod video was sent to a model)

**Data.** The KuaiMod benchmark (1,000 real short videos, Hugging Face card licence CC BY-NC 4.0; the GitHub repository states CC-BY-SA-4.0). A fixed random sample of 300 (seed 20261117, same fine/violating ratio as the full set): 173 fine, 127 violating, 20 of them commerce-type (private trading, exaggerated product effects, hype selling, false promises of benefits). Only totals are published.

**Question.** Does the video violate the platform's community rules (yes/no)? Inputs: text (title, OCR, speech transcript as supplied by the dataset); frames (cover and frames, with the title); frames + text. Models: Qwen3.5-4B (NF4, thinking off) on text and frames + text; claude-haiku-5-5 (thinking off, batch) on all three. One run each.

**Predictions.**
1. For both models, frames + text is at least as accurate as text alone.
2. For claude-haiku-5-5, frames alone is less accurate than frames + text.
3. With frames + text, claude-haiku-5-5 is at least as accurate as Qwen3.5-4B.
4. Every run's accuracy lies between 60% and 85%.
5. With frames + text, claude-haiku-5-5 catches at least half of the 20 commerce-type violations.

**File hashes at freeze:** `lab/kuaimod.py` 657e6bbf3b0dc195; `sample300.json` e544f4e352657252; `test/benchmark.jsonl` 5e03db2b09fbd7f2.

### Changes after the run (Part 3)

1. `lab/claude.py`: the first claude-haiku-5-5 KuaiMod submission (900 requests with images, one upload of about 240 MB) failed with a connection error before the batch was created; nothing was processed or billed. Large request sets are now sent as several batches of at most about 40 MB each. Prompts, images and settings are unchanged. (2026-11-17 08:40.) The same edit is listed under Part 1 because the file is frozen there too.
2. `lab/kuaimod.py`: Qwen3.5-4B ran out of GPU memory on the frames + text input (2026-11-18 05:44): some videos
   have 9 frames of up to 1080×1440, which need more than the card's 8 GB. For the local model, images larger
   than 640 pixels on the long side are now scaled down to 640 (most KuaiMod frames are 360×640 and are left as
   they are), and the one frames + text answer already written at full size was removed and asked again. The
   text run is unaffected. claude-haiku-5-5 was sent the frames at their original size, so with frames + text the
   two models did not see identical images (prediction 3). (2026-11-18 05:50.)
