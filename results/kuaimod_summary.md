# KuaiMod real-data check: totals only

Fixed random sample of 300 of the 1,000 KuaiMod test videos (seed 20261117): 173 fine, 127 violating. Question asked: does the video violate the platform's community rules (yes/no). No frame, title or quotation is shown here.

| model | input | accuracy | violating caught | fine wrongly flagged | commerce-type violations caught | F1 (violating) | unparsed |
|---|---|---|---|---|---|---|---|
| haiku | frames | 207/300 (69%) [64–74] | 104/127 (82%) | 70/173 (40%) | 17/20 | 0.691 | 2 |
| haiku | frames+text | 214/300 (71%) [66–76] | 111/127 (87%) | 70/173 (40%) | 19/20 | 0.721 | 2 |
| haiku | text | 184/300 (61%) [56–67] | 73/127 (57%) | 62/173 (36%) | 17/20 | 0.557 | 2 |
| qwen35-4b | frames+text | 225/300 (75%) [70–80] | 97/127 (76%) | 45/173 (26%) | 18/20 | 0.721 | 0 |
| qwen35-4b | text | 214/300 (71%) [66–76] | 58/127 (46%) | 17/173 (10%) | 15/20 | 0.574 | 0 |
