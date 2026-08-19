# VisionNarrator - Final System Summary

- **Captioner:** google/vit-base-patch16-224-in21k + gpt2 | best phase phase3 | val loss 2.3656
- **Story model:** Qwen/Qwen2.5-1.5B-Instruct (loaded)
- **Test images evaluated:** 4768
- **BLEU-1 (beam 5):** 0.7082
- **BLEU-2 (beam 5):** 0.5142
- **BLEU-3 (beam 5):** 0.3718
- **BLEU-4 (beam 5):** 0.2675
- **METEOR (beam 5):** 0.2285
- **ROUGE-L (beam 5):** 0.5362
- **CIDEr (beam 5):** 0.6038
- **BLEU-4 vs previous best (500 imgs):** 0.2684 vs 0.2537 (Fine-tuned ViT + Decoder)
- **Semantic layer (mean entities / actions per image):** 2.58 / 1.0
- **Story grounded ratio / entity coverage:** 0.561 / 0.9
- **Pipeline latency (story mode, mean):** 9.787 s
- **Pipeline latency (caption only, mean):** 0.241 s

## Components

| component | parameters | disk MB | device | load s |
|---|---|---|---|---|
| captioner (ViT + projection + GPT-2) | 211419648 | 846 | cuda | 4.6 |
| semantic layer (spaCy core_web_sm) | - | - | cpu | 0.2 |
| story model (Qwen/Qwen2.5-1.5B-Instruct) | 1543714304 | - | cuda | 5.8 |

## Stage latency (story mode, mean seconds)

- preprocess: 0.005
- encode: 0.01
- caption: 0.235
- semantics: 0.12
- text:short_description: 0.458
- text:detailed_explanation: 1.587
- text:story: 7.37
- total: 9.787
