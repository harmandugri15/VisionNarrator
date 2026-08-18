# Evaluation Results - VisionNarrator

Model: `vit_gpt2_captioner.pt` (best phase `phase3`, val loss 2.3656) | ViT `google/vit-base-patch16-224-in21k` + GPT-2 `gpt2` | 211,419,648 parameters

Test set: 4,768 images, 5.0 references per image | decoding: greedy and beam 5 (max 30 tokens, no-repeat 3-gram)

## Metrics (full test set)

|  | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|---|---|---|
| greedy | 0.6912 | 0.4912 | 0.3462 | 0.2440 | 0.2225 | 0.5252 | 0.5492 |
| beam 5 | 0.7082 | 0.5142 | 0.3718 | 0.2675 | 0.2285 | 0.5362 | 0.6038 |

## Model comparison (first 500 test images, BLEU)

| BLEU, first 500 test images | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---|---|---|---|
| Scratch ViT | 0.4904 | 0.2766 | 0.1616 | 0.1005 |
| Frozen ViT + Decoder | 0.6623 | 0.4671 | 0.3252 | 0.2267 |
| Fine-tuned ViT + Decoder | 0.6790 | 0.4916 | 0.3528 | 0.2537 |
| ViT + GPT-2 (greedy) | 0.6901 | 0.4892 | 0.3449 | 0.2445 |
| ViT + GPT-2 (beam 5) | 0.7103 | 0.5156 | 0.3725 | 0.2684 |
| ViT + GPT-2 (greedy) - recomputed (500 imgs) | 0.6901 | 0.4892 | 0.3449 | 0.2445 |
| ViT + GPT-2 (beam 5) - recomputed (500 imgs) | 0.7103 | 0.5156 | 0.3725 | 0.2684 |

## Analysis

- Distinct caption ratio: 0.890 (most common caption: "A group of people are walking down the street ." x32)
- Mean caption length (words): beam 12.15 | greedy 12.29 | references 13.38
- Degenerate outputs: empty caption = 0, under 4 words = 0, no EOS within MAX_NEW_TOKENS = 1, repeated bigram = 531
- Share of images with CIDEr < 0.2: 0.299
- Spearman(confidence, CIDEr) = 0.376 | Spearman(length, CIDEr) = -0.248
