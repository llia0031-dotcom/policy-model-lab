# All results

Built by `python -m lab.report` from the files in `results/`. Method and predictions: `eval/PREREGISTRATION.md`. Reading guide: `eval/EVALUATION.md`.

## Video test: results

96 videos (48 Chinese, 48 English): 64 violating with 72 labels, 32 compliant (look-alike traps are counted in their own column).

| model | input | decision accuracy | violating caught | compliant wrongly rejected | traps wrongly rejected | labels found | exact | evidence matches span | unparsed | median s | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| haiku | text | 53/96 (55%) [45–65] | 25/64 (39%) | 4/32 (12%) | 2/22 (9%) | 25/72 (35%) | 44/96 (46%) | 16/22 (73%) | 24 | None | – |
| haiku | cover | 63/96 (66%) [56–74] | 38/64 (59%) | 7/32 (22%) | 7/22 (32%) | 36/72 (50%) | 53/96 (55%) | 19/27 (70%) | 0 | None | – |
| haiku | frames8 | 78/96 (81%) [72–88] | 59/64 (92%) | 13/32 (41%) | 12/22 (55%) | 64/72 (89%) | 65/96 (68%) | 48/49 (98%) | 0 | None | – |
| haiku | frames16 | 86/96 (90%) [82–94] | 64/64 (100%) | 10/32 (31%) | 9/22 (41%) | 71/72 (99%) | 75/96 (78%) | 55/55 (100%) | 0 | None | – |
| haiku | ocr8 | 78/96 (81%) [72–88] | 54/64 (84%) | 8/32 (25%) | 8/22 (36%) | 57/72 (79%) | 66/96 (69%) | 48/50 (96%) | 0 | None | – |
| haiku | ocr16-text-classifier | 77/96 (80%) [71–87] | 54/64 (84%) | 9/32 (28%) | 9/22 (41%) | 56/72 (78%) | 59/96 (61%) | 0/55 (0%) | 8 | None | – |
| haiku-400 | ocr16-text-classifier | 82/96 (85%) [77–91] | 57/64 (89%) | 7/32 (22%) | 7/22 (32%) | 61/72 (85%) | 69/96 (72%) | 0/55 (0%) | 0 | None | – |
| mmbert | ocr16-text-classifier | 72/96 (75%) [65–83] | 44/64 (69%) | 4/32 (12%) | 4/22 (18%) | 45/72 (62%) | 65/96 (68%) | 0/42 (0%) | 0 | 0.02 | – |
| qwen3-0.6b-lora | ocr16-text-classifier | 71/96 (74%) [64–82] | 47/64 (73%) | 8/32 (25%) | 8/22 (36%) | 50/72 (69%) | 63/96 (66%) | 0/47 (0%) | 0 | 0.1 | – |
| qwen3-0.6b-zs | ocr16-text-classifier | 53/96 (55%) [45–65] | 24/64 (38%) | 3/32 (9%) | 3/22 (14%) | 19/72 (26%) | 36/96 (38%) | 0/18 (0%) | 0 | 0.09 | – |
| qwen3-1.7b-lora | ocr16-text-classifier | 77/96 (80%) [71–87] | 50/64 (78%) | 5/32 (16%) | 5/22 (23%) | 52/72 (72%) | 70/96 (73%) | 0/51 (0%) | 0 | 0.1 | – |
| qwen3-1.7b-zs | ocr16-text-classifier | 66/96 (69%) [59–77] | 40/64 (62%) | 6/32 (19%) | 6/22 (27%) | 40/72 (56%) | 51/96 (53%) | 0/37 (0%) | 0 | 0.08 | – |
| qwen35-4b | text | 47/96 (49%) [39–59] | 16/64 (25%) | 1/32 (3%) | 1/22 (5%) | 15/72 (21%) | 42/96 (44%) | 11/15 (73%) | 0 | 0.97 | 3.45 |
| qwen35-4b | cover | 60/96 (62%) [53–72] | 40/64 (62%) | 12/32 (38%) | 10/22 (45%) | 36/72 (50%) | 49/96 (51%) | 14/27 (52%) | 0 | 2.37 | 3.55 |
| qwen35-4b | frames8 | 69/96 (72%) [62–80] | 60/64 (94%) | 23/32 (72%) | 17/22 (77%) | 56/72 (78%) | 57/96 (59%) | 37/44 (84%) | 0 | 9.15 | 5.77 |
| qwen35-4b | video | 80/96 (83%) [75–89] | 61/64 (95%) | 13/32 (41%) | 13/22 (59%) | 58/72 (81%) | 69/96 (72%) | 36/46 (78%) | 0 | 19.22 | 5.8 |
| qwen35-4b | ocr8 | 74/96 (77%) [68–84] | 52/64 (81%) | 10/32 (31%) | 10/22 (45%) | 50/72 (69%) | 64/96 (67%) | 38/43 (88%) | 0 | 1.96 | 3.46 |
| qwen35-4b | ocr16 | 80/96 (83%) [75–89] | 57/64 (89%) | 9/32 (28%) | 9/22 (41%) | 56/72 (78%) | 70/96 (73%) | 48/51 (94%) | 0 | 2.39 | 3.46 |
| qwen3vl-2b | text | 49/96 (51%) [41–61] | 17/64 (27%) | 0/32 (0%) | 0/22 (0%) | 12/72 (17%) | 41/96 (43%) | 9/11 (82%) | 0 | 0.27 | 4.05 |
| qwen3vl-2b | cover | 66/96 (69%) [59–77] | 50/64 (78%) | 16/32 (50%) | 11/22 (50%) | 37/72 (51%) | 42/96 (44%) | 11/25 (44%) | 0 | 2.36 | 4.21 |
| qwen3vl-2b | frames8 | 65/96 (68%) [58–76] | 61/64 (95%) | 28/32 (88%) | 21/22 (95%) | 64/72 (89%) | 29/96 (30%) | 26/49 (53%) | 0 | 58.15 | 6.6 |
| qwen3vl-2b | video | 54/96 (56%) [46–66] | 40/64 (62%) | 18/32 (56%) | 13/22 (59%) | 42/72 (58%) | 37/96 (39%) | 24/31 (77%) | 0 | 71.56 | 6.63 |
| qwen3vl-2b | ocr8 | 49/96 (51%) [41–61] | 36/64 (56%) | 19/32 (59%) | 14/22 (64%) | 35/72 (49%) | 40/96 (42%) | 26/30 (87%) | 0 | 12.19 | 4.07 |
| qwen3vl-2b | ocr16 | 55/96 (57%) [47–67] | 38/64 (59%) | 15/32 (47%) | 11/22 (50%) | 35/72 (49%) | 46/96 (48%) | 23/27 (85%) | 0 | 9.52 | 4.07 |
| siglip2 | siglip-frames8 | 58/96 (60%) [50–70] | 34/64 (53%) | 8/32 (25%) | 8/22 (36%) | 27/72 (38%) | 38/96 (40%) | 0/20 (0%) | 0 | 0.31 | – |
| siglip2 | siglip-frames16 | 62/96 (65%) [55–73] | 37/64 (58%) | 7/32 (22%) | 7/22 (32%) | 30/72 (42%) | 39/96 (41%) | 0/23 (0%) | 0 | 0.62 | – |

## Labels found, by where the violation sits

| model | input | caption | subtitle on cover | subtitle later | 0.5 s sticker | footage only |
|---|---|---|---|---|---|---|
| haiku | text | 16/16 (100%) | 0/3 (0%) | 4/17 (24%) | 2/20 (10%) | 3/16 (19%) |
| haiku | cover | 16/16 (100%) | 3/3 (100%) | 5/17 (29%) | 3/20 (15%) | 9/16 (56%) |
| haiku | frames8 | 16/16 (100%) | 3/3 (100%) | 17/17 (100%) | 13/20 (65%) | 15/16 (94%) |
| haiku | frames16 | 16/16 (100%) | 3/3 (100%) | 17/17 (100%) | 19/20 (95%) | 16/16 (100%) |
| haiku | ocr8 | 16/16 (100%) | 3/3 (100%) | 17/17 (100%) | 14/20 (70%) | 7/16 (44%) |
| haiku | ocr16-text-classifier | 16/16 (100%) | 3/3 (100%) | 17/17 (100%) | 19/20 (95%) | 1/16 (6%) |
| haiku-400 | ocr16-text-classifier | 16/16 (100%) | 2/3 (67%) | 17/17 (100%) | 20/20 (100%) | 6/16 (38%) |
| mmbert | ocr16-text-classifier | 14/16 (88%) | 2/3 (67%) | 15/17 (88%) | 11/20 (55%) | 3/16 (19%) |
| qwen3-0.6b-lora | ocr16-text-classifier | 13/16 (81%) | 2/3 (67%) | 16/17 (94%) | 16/20 (80%) | 3/16 (19%) |
| qwen3-0.6b-zs | ocr16-text-classifier | 6/16 (38%) | 1/3 (33%) | 6/17 (35%) | 5/20 (25%) | 1/16 (6%) |
| qwen3-1.7b-lora | ocr16-text-classifier | 14/16 (88%) | 2/3 (67%) | 17/17 (100%) | 18/20 (90%) | 1/16 (6%) |
| qwen3-1.7b-zs | ocr16-text-classifier | 9/16 (56%) | 3/3 (100%) | 10/17 (59%) | 15/20 (75%) | 3/16 (19%) |
| qwen35-4b | text | 14/16 (88%) | 0/3 (0%) | 0/17 (0%) | 1/20 (5%) | 0/16 (0%) |
| qwen35-4b | cover | 15/16 (94%) | 3/3 (100%) | 5/17 (29%) | 4/20 (20%) | 9/16 (56%) |
| qwen35-4b | frames8 | 13/16 (81%) | 3/3 (100%) | 14/17 (82%) | 14/20 (70%) | 12/16 (75%) |
| qwen35-4b | video | 14/16 (88%) | 3/3 (100%) | 14/17 (82%) | 15/20 (75%) | 12/16 (75%) |
| qwen35-4b | ocr8 | 14/16 (88%) | 3/3 (100%) | 16/17 (94%) | 10/20 (50%) | 7/16 (44%) |
| qwen35-4b | ocr16 | 14/16 (88%) | 3/3 (100%) | 17/17 (100%) | 17/20 (85%) | 5/16 (31%) |
| qwen3vl-2b | text | 9/16 (56%) | 0/3 (0%) | 1/17 (6%) | 1/20 (5%) | 1/16 (6%) |
| qwen3vl-2b | cover | 14/16 (88%) | 2/3 (67%) | 4/17 (24%) | 5/20 (25%) | 12/16 (75%) |
| qwen3vl-2b | frames8 | 15/16 (94%) | 3/3 (100%) | 16/17 (94%) | 15/20 (75%) | 15/16 (94%) |
| qwen3vl-2b | video | 10/16 (62%) | 1/3 (33%) | 11/17 (65%) | 9/20 (45%) | 11/16 (69%) |
| qwen3vl-2b | ocr8 | 10/16 (62%) | 2/3 (67%) | 10/17 (59%) | 8/20 (40%) | 5/16 (31%) |
| qwen3vl-2b | ocr16 | 8/16 (50%) | 1/3 (33%) | 9/17 (53%) | 9/20 (45%) | 8/16 (50%) |
| siglip2 | siglip-frames8 | 1/16 (6%) | 2/3 (67%) | 10/17 (59%) | 7/20 (35%) | 7/16 (44%) |
| siglip2 | siglip-frames16 | 2/16 (12%) | 2/3 (67%) | 10/17 (59%) | 9/20 (45%) | 7/16 (44%) |

## Footage-only violations: does the evidence point at the footage?

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

## Text classifiers: results

| test set | method | decision accuracy | violating caught | compliant wrongly rejected | look-alikes wrongly rejected | labels found | exact | macro-F1 | unparsed | median s/item |
|---|---|---|---|---|---|---|---|---|---|---|
| grounded-review | qwen3-0.6b-zs | 100/168 (60%) [52–67] | 36/93 (39%) | 11/75 (15%) | 11/56 (20%) | 25/99 (25%) | 77/168 (46%) | 0.199 | 0 | 0.0558 |
| grounded-review | qwen3-0.6b-lora | 130/168 (77%) [70–83] | 79/93 (85%) | 24/75 (32%) | 21/56 (38%) | 76/99 (77%) | 117/168 (70%) | 0.733 | 0 | 0.0777 |
| grounded-review | qwen3-1.7b-zs | 114/168 (68%) [60–74] | 52/93 (56%) | 13/75 (17%) | 10/56 (18%) | 47/99 (47%) | 92/168 (55%) | 0.431 | 0 | 0.0721 |
| grounded-review | qwen3-1.7b-lora | 133/168 (79%) [72–85] | 84/93 (90%) | 26/75 (35%) | 19/56 (34%) | 83/99 (84%) | 123/168 (73%) | 0.774 | 0 | 0.0966 |
| grounded-review | mmbert | 132/168 (79%) [72–84] | 73/93 (78%) | 16/75 (21%) | 10/56 (18%) | 73/99 (74%) | 128/168 (76%) | 0.775 | 0 | 0.0075 |
| grounded-review | haiku | 138/168 (82%) [76–87] | 92/93 (99%) | 29/75 (39%) | 20/56 (36%) | 96/99 (97%) | 122/168 (73%) | 0.832 | 3 | None |
| grounded-review | haiku-400 | 138/168 (82%) [76–87] | 92/93 (99%) | 29/75 (39%) | 19/56 (34%) | 97/99 (98%) | 120/168 (71%) | 0.833 | 0 | None |
| video-text | qwen3-0.6b-zs | 58/96 (60%) [50–70] | 18/52 (35%) | 4/44 (9%) | 3/22 (14%) | 17/56 (30%) | 46/96 (48%) | 0.286 | 0 | 0.0663 |
| video-text | qwen3-0.6b-lora | 86/96 (90%) [82–94] | 48/52 (92%) | 6/44 (14%) | 6/22 (27%) | 48/56 (86%) | 79/96 (82%) | 0.8 | 0 | 0.0809 |
| video-text | qwen3-1.7b-zs | 77/96 (80%) [71–87] | 41/52 (79%) | 8/44 (18%) | 6/22 (27%) | 40/56 (71%) | 60/96 (62%) | 0.635 | 0 | 0.0591 |
| video-text | qwen3-1.7b-lora | 89/96 (93%) [86–96] | 50/52 (96%) | 5/44 (11%) | 5/22 (23%) | 52/56 (93%) | 85/96 (89%) | 0.892 | 0 | 0.0942 |
| video-text | mmbert | 82/96 (85%) [77–91] | 42/52 (81%) | 4/44 (9%) | 2/22 (9%) | 42/56 (75%) | 77/96 (80%) | 0.745 | 0 | 0.0033 |
| video-text | haiku | 83/96 (86%) [78–92] | 52/52 (100%) | 13/44 (30%) | 9/22 (41%) | 56/56 (100%) | 72/96 (75%) | 0.85 | 0 | None |
| video-text | haiku-400 | 82/96 (85%) [77–91] | 52/52 (100%) | 14/44 (32%) | 9/22 (41%) | 56/56 (100%) | 71/96 (74%) | 0.838 | 0 | None |
| real | qwen3-0.6b-zs | 25/46 (54%) [40–68] | 18/35 (51%) | 4/11 (36%) | 4/11 (36%) | 9/40 (22%) | 10/46 (22%) | 0.116 | 0 | 0.1359 |
| real | qwen3-0.6b-lora | 36/46 (78%) [64–88] | 33/35 (94%) | 8/11 (73%) | 8/11 (73%) | 29/40 (72%) | 24/46 (52%) | 0.598 | 0 | 0.1607 |
| real | qwen3-1.7b-zs | 21/46 (46%) [32–60] | 13/35 (37%) | 3/11 (27%) | 3/11 (27%) | 13/40 (32%) | 12/46 (26%) | 0.233 | 0 | 0.1774 |
| real | qwen3-1.7b-lora | 38/46 (83%) [69–91] | 33/35 (94%) | 6/11 (55%) | 6/11 (55%) | 30/40 (75%) | 26/46 (57%) | 0.591 | 0 | 0.2276 |
| real | mmbert | 25/46 (54%) [40–68] | 15/35 (43%) | 1/11 (9%) | 1/11 (9%) | 14/40 (35%) | 20/46 (43%) | 0.265 | 0 | 0.0352 |
| real | haiku | 33/46 (72%) [57–83] | 31/35 (89%) | 9/11 (82%) | 9/11 (82%) | 27/40 (68%) | 13/46 (28%) | 0.459 | 15 | None |
| real | haiku-400 | 35/46 (76%) [62–86] | 31/35 (89%) | 7/11 (64%) | 7/11 (64%) | 33/40 (82%) | 18/46 (39%) | 0.545 | 3 | None |
| synth-val | qwen3-0.6b-zs | 117/191 (61%) [54–68] | 20/89 (22%) | 5/102 (5%) | 5/71 (7%) | 11/98 (11%) | 104/191 (54%) | 0.122 | 0 | 0.0563 |
| synth-val | qwen3-0.6b-lora | 170/191 (89%) [84–93] | 78/89 (88%) | 10/102 (10%) | 6/71 (8%) | 80/98 (82%) | 162/191 (85%) | 0.831 | 0 | 0.0872 |
| synth-val | qwen3-1.7b-zs | 135/191 (71%) [64–77] | 34/89 (38%) | 1/102 (1%) | 1/71 (1%) | 26/98 (27%) | 122/191 (64%) | 0.231 | 0 | 0.06 |
| synth-val | qwen3-1.7b-lora | 174/191 (91%) [86–94] | 78/89 (88%) | 6/102 (6%) | 5/71 (7%) | 79/98 (81%) | 165/191 (86%) | 0.851 | 0 | 0.0872 |
| synth-val | mmbert | 173/191 (91%) [86–94] | 77/89 (87%) | 6/102 (6%) | 5/71 (7%) | 79/98 (81%) | 164/191 (86%) | 0.867 | 0 | 0.007 |
| synth-val | haiku | 162/191 (85%) [79–89] | 84/89 (94%) | 24/102 (24%) | 18/71 (25%) | 88/98 (90%) | 143/191 (75%) | 0.817 | 6 | None |
| synth-val | haiku-400 | 169/191 (88%) [83–92] | 84/89 (94%) | 17/102 (17%) | 13/71 (18%) | 90/98 (92%) | 157/191 (82%) | 0.874 | 0 | None |
| video-ocr | qwen3-0.6b-zs | 53/96 (55%) [45–65] | 24/64 (38%) | 3/32 (9%) | – | 19/72 (26%) | 36/96 (38%) | 0.251 | 0 | 0.0896 |
| video-ocr | qwen3-0.6b-lora | 71/96 (74%) [64–82] | 47/64 (73%) | 8/32 (25%) | – | 50/72 (69%) | 63/96 (66%) | 0.659 | 0 | 0.1004 |
| video-ocr | qwen3-1.7b-zs | 66/96 (69%) [59–77] | 40/64 (62%) | 6/32 (19%) | – | 40/72 (56%) | 51/96 (53%) | 0.572 | 0 | 0.0756 |
| video-ocr | qwen3-1.7b-lora | 77/96 (80%) [71–87] | 50/64 (78%) | 5/32 (16%) | – | 52/72 (72%) | 70/96 (73%) | 0.714 | 0 | 0.1016 |
| video-ocr | mmbert | 72/96 (75%) [65–83] | 44/64 (69%) | 4/32 (12%) | – | 45/72 (62%) | 65/96 (68%) | 0.643 | 0 | 0.018 |
| video-ocr | haiku | 77/96 (80%) [71–87] | 54/64 (84%) | 9/32 (28%) | – | 56/72 (78%) | 59/96 (61%) | 0.65 | 8 | None |
| video-ocr | haiku-400 | 82/96 (85%) [77–91] | 57/64 (89%) | 7/32 (22%) | – | 61/72 (85%) | 69/96 (72%) | 0.751 | 0 | None |

## Text classifiers: paired differences

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

## KuaiMod real-data check: totals only

Fixed random sample of 300 of the 1,000 KuaiMod test videos (seed 20261117): 173 fine, 127 violating. Question asked: does the video violate the platform's community rules (yes/no). No frame, title or quotation is shown here.

| model | input | accuracy | violating caught | fine wrongly flagged | commerce-type violations caught | F1 (violating) | unparsed |
|---|---|---|---|---|---|---|---|
| haiku | frames | 207/300 (69%) [64–74] | 104/127 (82%) | 70/173 (40%) | 17/20 | 0.691 | 2 |
| haiku | frames+text | 214/300 (71%) [66–76] | 111/127 (87%) | 70/173 (40%) | 19/20 | 0.721 | 2 |
| haiku | text | 184/300 (61%) [56–67] | 73/127 (57%) | 62/173 (36%) | 17/20 | 0.557 | 2 |
| qwen35-4b | frames+text | 225/300 (75%) [70–80] | 97/127 (76%) | 45/173 (26%) | 18/20 | 0.721 | 0 |
| qwen35-4b | text | 214/300 (71%) [66–76] | 58/127 (46%) | 17/173 (10%) | 15/20 | 0.574 | 0 |

## Training runs

| run | examples | trainable / total parameters | epochs (train loss, validation loss) | minutes | peak GB |
|---|---|---|---|---|---|
| mmbert-base | 1529 | all | 1: 0.2183, 0.0706; 2: 0.0333, 0.0764; 3: 0.0056, 0.0709; 4: 0.0005, 0.0642 | 22.4 | – |
| qwen3-0.6b-lora | 1529 | 10,092,544 / 606,142,464 | 1: 0.1201, 0.0819; 2: 0.0328, 0.0736; 3: 0.0141, 0.0658 | 22.2 | 5.55 |
| qwen3-1.7b-lora | 1529 | 17,432,576 / 1,738,007,552 | 1: 0.1554, 0.08; 2: 0.0313, 0.0654; 3: 0.0136, 0.0627 | 46.2 | 7.76 |

## Claude API cost (batch prices)

| batch | input tokens | output tokens | USD |
|---|---|---|---|
| synth-train | 124,091 | 198,058 | 0.0557 |
| video-test | 1,550,772 | 24,535 | 0.0837 |
| text-haiku | 279,807 | 13,963 | 0.0175 |
| kuaimod-haiku | 609,713 | 5,824 | 0.0319 |
| kuaimod-haiku | 310,778 | 353 | 0.0156 |
| kuaimod-haiku | 304,527 | 326 | 0.0153 |
| kuaimod-haiku | 282,070 | 314 | 0.0142 |
| kuaimod-haiku | 310,016 | 300 | 0.0156 |
| kuaimod-haiku | 318,966 | 361 | 0.0160 |
| kuaimod-haiku | 223,394 | 177 | 0.0112 |
| kuaimod-haiku | 227,363 | 399 | 0.0115 |
| kuaimod-haiku | 237,641 | 243 | 0.0119 |
| kuaimod-haiku | 295,368 | 252 | 0.0148 |
| kuaimod-haiku | 287,016 | 334 | 0.0144 |
| kuaimod-haiku | 336,136 | 453 | 0.0169 |
| kuaimod-haiku | 216,542 | 194 | 0.0109 |
| kuaimod-haiku | 228,650 | 226 | 0.0115 |
| kuaimod-haiku | 373,514 | 485 | 0.0188 |
| kuaimod-haiku | 300,115 | 529 | 0.0151 |
| kuaimod-haiku | 272,621 | 447 | 0.0137 |
| kuaimod-haiku | 358,364 | 387 | 0.0180 |
| kuaimod-haiku | 352,753 | 345 | 0.0177 |
| kuaimod-haiku | 276,614 | 319 | 0.0139 |
| kuaimod-haiku | 292,701 | 352 | 0.0147 |
| kuaimod-haiku | 327,285 | 284 | 0.0164 |
| kuaimod-haiku | 316,005 | 347 | 0.0159 |
| kuaimod-haiku | 236,972 | 197 | 0.0119 |
| kuaimod-haiku | 271,787 | 315 | 0.0137 |
| kuaimod-haiku | 247,758 | 241 | 0.0124 |
| kuaimod-haiku | 301,065 | 255 | 0.0151 |
| kuaimod-haiku | 300,211 | 336 | 0.0151 |
| kuaimod-haiku | 353,267 | 456 | 0.0178 |
| kuaimod-haiku | 224,472 | 195 | 0.0113 |
| kuaimod-haiku | 240,978 | 223 | 0.0121 |
| kuaimod-haiku | 386,157 | 501 | 0.0194 |
| kuaimod-haiku | 302,579 | 629 | 0.0153 |
| kuaimod-haiku | 288,690 | 467 | 0.0146 |
| kuaimod-haiku | 40,055 | 80 | 0.0020 |
| text-haiku-400 | 279,807 | 21,636 | 0.0194 |
| video-test | 57,256 | 5,151 | 0.0042 |
| **total** | | | **0.6971** |
