# Footage-only violations: does the evidence point at the footage?

Exploratory, not pre-registered. Built by `python -m lab.visual_evidence` (rule in the script's header).

| model | input | sees the footage | footage-only labels caught | evidence says the footage shows it | category only | alcohol look-alikes wrongly flagged (of 7) | of those, evidence says drinking is shown |
|---|---|---|---|---|---|---|---|
| haiku-400 | ocr16-text-classifier | no | 6/16 | – | 6 | 0 | – |
| haiku | ocr16-text-classifier | no | 1/16 | – | 1 | 1 | – |
| haiku | cover | yes | 9/16 | 2 | 7 | 2 | 0 |
| haiku | frames16 | yes | 16/16 | 16 | 0 | 2 | 0 |
| haiku | frames8 | yes | 15/16 | 15 | 0 | 3 | 1 |
| haiku | ocr8 | no | 7/16 | – | 7 | 2 | – |
| haiku | text | no | 3/16 | – | 3 | 0 | – |
| mmbert | ocr16-text-classifier | no | 3/16 | – | 3 | 2 | – |
| qwen3-0.6b-lora | ocr16-text-classifier | no | 3/16 | – | 3 | 2 | – |
| qwen3-0.6b-zs | ocr16-text-classifier | no | 1/16 | – | 1 | 1 | – |
| qwen3-1.7b-lora | ocr16-text-classifier | no | 1/16 | – | 1 | 1 | – |
| qwen3-1.7b-zs | ocr16-text-classifier | no | 3/16 | – | 3 | 1 | – |
| qwen35-4b | cover | yes | 9/16 | 2 | 7 | 1 | 0 |
| qwen35-4b | frames8 | yes | 12/16 | 12 | 0 | 4 | 2 |
| qwen35-4b | ocr16 | no | 5/16 | – | 5 | 2 | – |
| qwen35-4b | ocr8 | no | 7/16 | – | 7 | 2 | – |
| qwen35-4b | video | yes | 12/16 | 12 | 0 | 3 | 1 |
| qwen3vl-2b | cover | yes | 12/16 | 3 | 9 | 2 | 0 |
| qwen3vl-2b | frames8 | yes | 15/16 | 6 | 9 | 6 | 1 |
| qwen3vl-2b | ocr16 | no | 8/16 | – | 8 | 3 | – |
| qwen3vl-2b | ocr8 | no | 5/16 | – | 5 | 2 | – |
| qwen3vl-2b | text | no | 1/16 | – | 1 | 0 | – |
| qwen3vl-2b | video | yes | 11/16 | 9 | 2 | 2 | 0 |
| siglip2 | siglip-frames16 | no | 7/16 | – | 7 | 0 | – |
| siglip2 | siglip-frames8 | no | 7/16 | – | 7 | 0 | – |

## Categories listed per ad

| model | input | categories listed | per ad | per compliant ad | listed with no evidence |
|---|---|---|---|---|---|
| haiku | cover | 50 | 0.52 | 0.25 | 2 |
| haiku | frames16 | 94 | 0.98 | 0.38 | 0 |
| haiku | frames8 | 89 | 0.93 | 0.41 | 3 |
| haiku | ocr8 | 75 | 0.78 | 0.28 | 2 |
| haiku | text | 33 | 0.34 | 0.12 | 6 |
| qwen35-4b | cover | 52 | 0.54 | 0.38 | 0 |
| qwen35-4b | frames8 | 84 | 0.88 | 0.72 | 1 |
| qwen35-4b | ocr16 | 68 | 0.71 | 0.28 | 1 |
| qwen35-4b | ocr8 | 63 | 0.66 | 0.31 | 0 |
| qwen35-4b | text | 17 | 0.18 | 0.03 | 0 |
| qwen35-4b | video | 74 | 0.77 | 0.41 | 10 |
| qwen3vl-2b | cover | 78 | 0.81 | 0.50 | 0 |
| qwen3vl-2b | frames8 | 270 | 2.81 | 3.22 | 103 |
| qwen3vl-2b | ocr16 | 65 | 0.68 | 0.56 | 0 |
| qwen3vl-2b | ocr8 | 76 | 0.79 | 0.88 | 0 |
| qwen3vl-2b | text | 17 | 0.18 | 0.00 | 0 |
| qwen3vl-2b | video | 145 | 1.51 | 2.16 | 56 |
