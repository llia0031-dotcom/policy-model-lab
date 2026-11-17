# Text classifiers: paired differences

Decision accuracy of the second method minus the first, in percentage points, on the items both answered. 95% interval from 4,000 paired bootstrap resamples (seed 0).

| test set | second method | first method | items | difference | 95% interval |
|---|---|---|---|---|---|
| grounded-review | qwen3-1.7b-lora | qwen3-1.7b-zs | 168 | +11.3 | +3.6 to +19.0 |
| grounded-review | qwen3-0.6b-lora | qwen3-0.6b-zs | 168 | +17.9 | +8.3 to +26.8 |
| grounded-review | qwen3-1.7b-lora | haiku | 168 | -3.0 | -9.5 to +3.6 |
| grounded-review | qwen3-1.7b-lora | haiku-400 | 168 | -3.0 | -9.5 to +3.6 |
| grounded-review | qwen3-1.7b-lora | mmbert | 168 | +0.6 | -7.1 to +7.7 |
| grounded-review | qwen3-1.7b-lora | qwen3-0.6b-lora | 168 | +1.8 | -4.2 to +7.7 |
| video-text | qwen3-1.7b-lora | qwen3-1.7b-zs | 96 | +12.5 | +4.2 to +20.8 |
| video-text | qwen3-0.6b-lora | qwen3-0.6b-zs | 96 | +29.2 | +17.7 to +39.6 |
| video-text | qwen3-1.7b-lora | haiku | 96 | +6.2 | +0.0 to +12.5 |
| video-text | qwen3-1.7b-lora | haiku-400 | 96 | +7.3 | +1.0 to +14.6 |
| video-text | qwen3-1.7b-lora | mmbert | 96 | +7.3 | +0.0 to +15.6 |
| video-text | qwen3-1.7b-lora | qwen3-0.6b-lora | 96 | +3.1 | -1.0 to +8.3 |
| real | qwen3-1.7b-lora | qwen3-1.7b-zs | 46 | +37.0 | +19.6 to +54.3 |
| real | qwen3-0.6b-lora | qwen3-0.6b-zs | 46 | +23.9 | +6.5 to +41.3 |
| real | qwen3-1.7b-lora | haiku | 46 | +10.9 | +0.0 to +21.7 |
| real | qwen3-1.7b-lora | haiku-400 | 46 | +6.5 | -4.3 to +17.4 |
| real | qwen3-1.7b-lora | mmbert | 46 | +28.3 | +8.7 to +47.8 |
| real | qwen3-1.7b-lora | qwen3-0.6b-lora | 46 | +4.3 | -4.3 to +13.0 |
| synth-val | qwen3-1.7b-lora | qwen3-1.7b-zs | 191 | +20.4 | +13.6 to +27.2 |
| synth-val | qwen3-0.6b-lora | qwen3-0.6b-zs | 191 | +27.7 | +19.9 to +35.6 |
| synth-val | qwen3-1.7b-lora | haiku | 191 | +6.3 | +1.0 to +12.0 |
| synth-val | qwen3-1.7b-lora | haiku-400 | 191 | +2.6 | -2.6 to +7.9 |
| synth-val | qwen3-1.7b-lora | mmbert | 191 | +0.5 | -3.1 to +4.7 |
| synth-val | qwen3-1.7b-lora | qwen3-0.6b-lora | 191 | +2.1 | -1.6 to +5.8 |
| video-ocr | qwen3-1.7b-lora | qwen3-1.7b-zs | 96 | +11.5 | +3.1 to +20.8 |
| video-ocr | qwen3-0.6b-lora | qwen3-0.6b-zs | 96 | +18.8 | +7.3 to +30.2 |
| video-ocr | qwen3-1.7b-lora | haiku | 96 | +0.0 | -6.2 to +6.2 |
| video-ocr | qwen3-1.7b-lora | haiku-400 | 96 | -5.2 | -11.5 to +1.0 |
| video-ocr | qwen3-1.7b-lora | mmbert | 96 | +5.2 | -3.1 to +13.5 |
| video-ocr | qwen3-1.7b-lora | qwen3-0.6b-lora | 96 | +6.2 | +0.0 to +13.5 |
