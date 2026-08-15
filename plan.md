# VisionNarrator — Complete Pipeline Plan

> **Goal**: Build an end-to-end image captioning system using a **Vision Transformer (ViT)**
> as the visual encoder and a **pretrained Language Model (GPT-2)** as the text decoder,
> trained on the **Flickr30k** dataset using **Transfer Learning**.

---

## Table of Contents

1. [Project Directory Structure](#1-project-directory-structure)
2. [Environment Setup](#2-environment-setup)
3. [Dataset — Flickr30k](#3-dataset--flickr30k)
4. [Data Exploration](#4-data-exploration)
5. [Data Preprocessing](#5-data-preprocessing)
6. [Model Architecture](#6-model-architecture)
7. [Transfer Learning Strategy](#7-transfer-learning-strategy)
8. [Training Pipeline](#8-training-pipeline)
9. [Evaluation](#9-evaluation)
10. [Inference Pipeline](#10-inference-pipeline)
11. [Notebook Breakdown](#11-notebook-breakdown)
12. [Tools and Libraries](#12-tools-and-libraries)

---

## 1. Project Directory Structure

```
VisionNarrator/
│
├── notebooks/                         # Individual Jupyter notebooks (one per stage)
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_architecture.ipynb
│   ├── 04_training.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_inference_demo.ipynb
│
├── src/                               # Not pushed to GitHub (local only)
│   ├── data/                          # Flickr30k images + captions (local)
│   │   ├── flickr30k_images/          # All ~31,000 .jpg images
│   │   └── results.csv                # image_name | comment_number | comment
│   ├── checkpoints/                   # Saved model weights during training
│   └── cache/                         # Tokenizer and feature caches
│
├── docs/                              # Documentation and model references
│   ├── architecture.md
│   ├── training_logs.md
│   └── evaluation_results.md
│
├── plan.md                            # This file — full pipeline plan
├── requirements.txt                   # All pip dependencies
└── README.md
```

---

## 2. Environment Setup

- Python version: **>= 3.11.9**
- Use a **virtual environment** (venv or conda)

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers accelerate
pip install datasets pillow
pip install pandas numpy matplotlib seaborn
pip install scikit-learn nltk
pip install pycocoevalcap
pip install tqdm
```

> Pin all versions in `requirements.txt` after setup.

---

## 3. Dataset — Flickr30k

### What is Flickr30k?
- ~**31,000 images** sourced from Flickr (everyday real-world scenes)
- Each image has **5 human-written captions** (crowd-sourced via Amazon Mechanical Turk)
- Total: ~**155,000 caption-image pairs**
- Data is **already downloaded locally** in `src/data/`

### Expected Local File Structure

```
src/data/
├── flickr30k_images/          # ~31,000 .jpg files
│   ├── 1000092795.jpg
│   ├── 1000268201.jpg
│   └── ...
└── results.csv                # Columns: image_name | comment_number | comment
```

### Dataset Splits (Standard Karpathy Split)

| Split      | Images  | Purpose                        |
|------------|---------|--------------------------------|
| Train      | 29,000  | Model training                 |
| Validation | 1,014   | Hyperparameter tuning          |
| Test       | 1,000   | Final evaluation (held-out)    |

> Use the **Karpathy split** — it is the standard in image captioning research
> and allows direct comparison with published baselines.

---

## 4. Data Exploration

**Notebook**: `notebooks/01_data_exploration.ipynb`

### Goals
- Understand the structure of the dataset
- Analyse caption length distribution
- Understand vocabulary size
- Visualise sample image-caption pairs
- Identify corrupt or missing files

### Steps

1. Load `results.csv` into a pandas DataFrame
2. Group captions by image — confirm every image has exactly 5 captions
3. Plot a **histogram of caption lengths** (in words and in tokens)
4. Build a **word frequency map** — identify top-100 and rare words
5. Compute total **vocabulary size** (unique tokens before/after lowercasing)
6. Display **10–20 sample image-caption pairs** using matplotlib
7. Check for and log any **corrupt or missing images**

### Key Questions to Answer

| Question | Why It Matters |
|---|---|
| Average caption length? | Sets the `max_length` for the tokenizer |
| Vocabulary size? | Confirms using a pretrained tokenizer is the right call |
| Are images all RGB? | ViT requires 3-channel input |
| Typical image dimensions? | Needed for understanding resize impact |
| Any missing/corrupt files? | Must be removed before training |

---

## 5. Data Preprocessing

**Notebook**: `notebooks/02_data_preprocessing.ipynb`

This stage converts raw images and raw text into model-ready tensors.

---

### 5a. Image Preprocessing (for ViT)

The Vision Transformer expects fixed-size, patch-able, normalized images.

**Steps:**
1. Resize all images to **224 × 224 pixels** (ViT's standard input)
2. Normalize using **ImageNet statistics**:
   - Mean: `[0.485, 0.456, 0.406]`
   - Std:  `[0.229, 0.224, 0.225]`
3. Convert to PyTorch tensors

Use HuggingFace's `ViTImageProcessor` to handle all of the above automatically:

```python
from transformers import ViTImageProcessor

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
# Handles: resize + normalize + to tensor
inputs = processor(images=pil_image, return_tensors="pt")
pixel_values = inputs['pixel_values']  # shape: (1, 3, 224, 224)
```

**Training-only augmentations** (applied before the processor):
- Random horizontal flip
- Random color jitter (brightness ±0.2, contrast ±0.2)

> Do **NOT** apply augmentations to validation or test sets.

---

### 5b. Text Preprocessing (for GPT-2 Decoder)

1. Load the **GPT-2 tokenizer** from HuggingFace
2. Add special tokens:
   - `<|startoftext|>` — prepended to every caption (Start Of Sequence)
   - `<|endoftext|>`   — appended to every caption (End Of Sequence / EOS)
   - `<|pad|>`         — padding token (GPT-2 has none by default)
3. Set `max_length = 64` tokens (covers ~99% of Flickr30k captions)
4. Pad or truncate all captions to `max_length`
5. Build **attention masks** (1 = real token, 0 = padding)

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({
    'bos_token': '<|startoftext|>',
    'eos_token': '<|endoftext|>',
    'pad_token': '<|pad|>'
})
```

**Label construction** (what the model learns to predict):
```
Input IDs : [SOS, A,   dog, runs, ..., EOS, PAD, PAD]
Labels    : [A,   dog, runs, ..., EOS, -100, -100, -100]
```
> `-100` tells PyTorch's `CrossEntropyLoss` to **ignore those positions** (padding).
> The model predicts the next token at each step — never sees the SOS as a target.

---

### 5c. PyTorch Dataset Class

```python
from torch.utils.data import Dataset
from PIL import Image

class Flickr30kDataset(Dataset):
    def __init__(self, df, image_dir, processor, tokenizer, max_len=64, split='train'):
        self.df         = df                # DataFrame with image_name and caption
        self.image_dir  = image_dir
        self.processor  = processor
        self.tokenizer  = tokenizer
        self.max_len    = max_len
        self.split      = split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        image   = Image.open(f"{self.image_dir}/{row['image_name']}").convert("RGB")
        caption = row['comment']

        # Image → tensor
        if self.split == 'train':
            image = apply_augmentations(image)  # your augment fn
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        # Caption → token ids + labels
        encoded = self.tokenizer(
            f"<|startoftext|>{caption}<|endoftext|>",
            max_length    = self.max_len,
            padding       = 'max_length',
            truncation    = True,
            return_tensors= "pt"
        )
        input_ids      = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return pixel_values, input_ids, attention_mask, labels
```

> For training: use **all 5 captions per image** (5x the data).
> For validation/test: use **1 caption per image** for loss; all 5 for metric computation.

---

### 5d. DataLoaders

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
```

---

## 6. Model Architecture

**Notebook**: `notebooks/03_model_architecture.ipynb`

The full model is an **Encoder → Bridge → Decoder** pipeline.

```
Input Image (224×224×3)
        │
        ▼
┌───────────────────────────────────────────────┐
│           ViT Encoder                         │
│  Splits image into 16×16 patches (196 total)  │
│  + 1 [CLS] token = 197 sequence positions     │
│  Each position: 768-dim embedding             │
│  Output shape: (batch, 197, 768)              │
└───────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│           Projection Layer                    │
│  Linear(768, 768)  [or Linear(768, 1024)]     │
│  Maps ViT's output space → GPT-2's input space│
│  Output shape: (batch, 197, gpt2_hidden_dim)  │
└───────────────────────────────────────────────┘
        │
        ▼ (used as visual prefix — prepended to text embeddings)
┌───────────────────────────────────────────────┐
│           GPT-2 Decoder                       │
│  Input = [visual_prefix | text_embeddings]    │
│  Generates tokens autoregressively            │
│  Output: logits over vocabulary               │
└───────────────────────────────────────────────┘
        │
        ▼
Caption: "A woman riding a horse near the ocean."
```

---

### 6a. Vision Encoder — ViT

**Why ViT?**
- Unlike CNNs, ViT treats an image as a **sequence of patches** — naturally compatible with
  transformer-based language models downstream.
- The `[CLS]` token learns a **global image representation**.
- The 196 patch tokens carry **local spatial information**.
- Pretrained on ImageNet-21k (21,000 classes) → rich, general visual features.

```python
from transformers import ViTModel

vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

# During forward pass:
outputs       = vit(pixel_values=pixel_values)
image_features = outputs.last_hidden_state   # shape: (batch, 197, 768)
```

---

### 6b. Language Decoder — GPT-2

**Why GPT-2?**
- GPT-2 is a **causal language model** — it generates text left-to-right, one token at a time.
- Pretrained on a large English corpus (WebText) — already understands grammar, syntax, 
  common phrases, and real-world knowledge.
- We give it image features as a **visual prefix** and ask it to continue with a caption.

```python
from transformers import GPT2LMHeadModel

gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
# GPT-2 small: 12 layers, 12 heads, 768 hidden dim, 124M params
```

**Alternatives** (if compute allows):
- `gpt2-medium` (345M) — better quality
- `distilgpt2` (82M) — faster, lighter

---

### 6c. Projection Layer (Cross-Modal Bridge)

The projection layer is the **only randomly initialized component**.
Everything else (ViT and GPT-2) starts from pretrained weights.

```python
import torch.nn as nn

# If ViT hidden_dim == GPT2 hidden_dim (both 768 for gpt2-small):
self.projection = nn.Linear(768, 768)

# If using gpt2-medium (1024-dim):
self.projection = nn.Linear(768, 1024)
```

For better projection capacity, optionally use an **MLP**:
```python
self.projection = nn.Sequential(
    nn.Linear(768, 768),
    nn.GELU(),
    nn.Linear(768, 768),
    nn.Dropout(0.1)
)
```

---

### 6d. Full VisionNarrator Model (forward pass)

```python
import torch
import torch.nn as nn
from transformers import ViTModel, GPT2LMHeadModel

class VisionNarrator(nn.Module):
    def __init__(self, vit_name='google/vit-base-patch16-224-in21k',
                       gpt2_name='gpt2', hidden_dim=768):
        super().__init__()
        self.vit        = ViTModel.from_pretrained(vit_name)
        self.projection = nn.Linear(self.vit.config.hidden_size, hidden_dim)
        self.gpt2       = GPT2LMHeadModel.from_pretrained(gpt2_name)

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        # --- Encode image ---
        image_features = self.vit(pixel_values=pixel_values).last_hidden_state
        # shape: (B, 197, 768)

        image_prefix   = self.projection(image_features)
        # shape: (B, 197, gpt2_hidden_dim)

        # --- Get GPT-2 word embeddings for the caption tokens ---
        text_embeddings = self.gpt2.transformer.wte(input_ids)
        # shape: (B, T, gpt2_hidden_dim)

        # --- Concatenate: [visual_prefix | text_embeddings] ---
        inputs_embeds = torch.cat([image_prefix, text_embeddings], dim=1)
        # shape: (B, 197 + T, gpt2_hidden_dim)

        # --- Extend attention mask to cover the prefix ---
        prefix_mask   = torch.ones(pixel_values.size(0), 197, device=pixel_values.device)
        full_mask     = torch.cat([prefix_mask, attention_mask.float()], dim=1)

        # --- Extend labels: ignore the visual prefix positions ---
        if labels is not None:
            prefix_labels = torch.full((pixel_values.size(0), 197), -100,
                                        device=pixel_values.device, dtype=torch.long)
            labels        = torch.cat([prefix_labels, labels], dim=1)

        # --- Forward through GPT-2 ---
        outputs = self.gpt2(
            inputs_embeds   = inputs_embeds,
            attention_mask  = full_mask,
            labels          = labels,
        )
        return outputs   # outputs.loss, outputs.logits
```

---

## 7. Transfer Learning Strategy

**Notebook**: `notebooks/04_training.ipynb` (Phase configuration at the top)

### Why Transfer Learning?
- Training from scratch requires **millions of images** and weeks of compute.
- ViT pretrained on ImageNet-21k already extracts rich visual features.
- GPT-2 pretrained on WebText already generates fluent English.
- We only need to **bridge** these two and **fine-tune** the connection — 
  achievable with ~29,000 images in a reasonable time.

---

### Phased Freezing Strategy

#### Phase 1 — Teach the Bridge (Epochs 1–3)

| Component       | Status   |
|-----------------|----------|
| ViT Encoder     | ❄️ Frozen |
| Projection Layer| ✅ Training |
| GPT-2 Decoder   | ❄️ Frozen |

**Goal**: The projection layer learns to map raw ViT image features into a space
that GPT-2's frozen self-attention can interpret. This is the warm-up phase.

**Learning rate**: `1e-3`

```python
# Freeze ViT
for param in model.vit.parameters():
    param.requires_grad = False

# Freeze GPT-2
for param in model.gpt2.parameters():
    param.requires_grad = False

# Projection only
for param in model.projection.parameters():
    param.requires_grad = True
```

---

#### Phase 2 — Fine-tune the Decoder (Epochs 4–8)

| Component              | Status       |
|------------------------|--------------|
| ViT Encoder            | ❄️ Frozen    |
| Projection Layer       | ✅ Training  |
| GPT-2 — top 4 layers   | ✅ Training  |
| GPT-2 — lower 8 layers | ❄️ Frozen   |

**Goal**: Adapt GPT-2's top transformer blocks to understand the visual prefix
and generate caption-style language. ViT stays frozen because its features are
already strong and we don't want to disturb them yet.

**Learning rate**: `5e-5`

```python
# Unfreeze GPT-2 last 4 transformer blocks (out of 12)
for i in range(8, 12):
    for param in model.gpt2.transformer.h[i].parameters():
        param.requires_grad = True
```

---

#### Phase 3 — Full Fine-tuning (Epochs 9–15, optional)

| Component       | Status      |
|-----------------|-------------|
| ViT Encoder     | ✅ Training |
| Projection Layer| ✅ Training |
| GPT-2 Decoder   | ✅ Training |

**Goal**: End-to-end fine-tuning. The ViT learns to extract features that are
specifically useful for captioning, not just classification. Use a very small
learning rate to avoid **catastrophic forgetting** of pretrained knowledge.

**Learning rate**: `1e-5`

> Monitor validation loss closely in Phase 3. Stop early if val loss increases.

---

### Optimizer and Scheduler

```python
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr           = phase_lr,     # changes per phase
    weight_decay = 1e-2
)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps   = len(train_loader) * 1,   # 1 epoch warmup
    num_training_steps = len(train_loader) * total_epochs
)
```

---

## 8. Training Pipeline

**Notebook**: `notebooks/04_training.ipynb`

### Loss Function

- **Cross-Entropy Loss** over vocabulary logits vs. ground-truth tokens
- Padding positions masked with `-100` → automatically ignored by PyTorch
- GPT-2 returns the loss directly when `labels` are passed

```
Loss = -1/N * Σ log P(token_t | image, token_1, ..., token_{t-1})
```

---

### Training Loop

```python
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for pixel_values, input_ids, attn_mask, labels in tqdm(train_loader):
        pixel_values = pixel_values.to(device)
        input_ids    = input_ids.to(device)
        attn_mask    = attn_mask.to(device)
        labels       = labels.to(device)

        outputs = model(pixel_values, input_ids, attn_mask, labels)
        loss    = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")
    validate(model, val_loader, tokenizer, device)
```

---

### Gradient Clipping
- Clip gradient norm to `max_norm=1.0` — prevents exploding gradients,
  critical when training transformers.

### Checkpointing

```python
# Save best model (by validation loss)
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), 'src/checkpoints/best_model.pth')
```

### Training Config Summary

| Hyperparameter     | Value              |
|--------------------|--------------------|
| Batch size         | 32                 |
| Max caption length | 64 tokens          |
| Total epochs       | 15 (3 phases)      |
| Optimizer          | AdamW              |
| Weight decay       | 0.01               |
| Gradient clipping  | max_norm = 1.0     |
| Mixed precision    | fp16 (optional)    |
| Warmup             | 1 epoch            |

---

## 9. Evaluation

**Notebook**: `notebooks/05_evaluation.ipynb`

### Standard Image Captioning Metrics

| Metric      | What It Measures                                                  |
|-------------|-------------------------------------------------------------------|
| **BLEU-1**  | Unigram precision (individual word overlap)                       |
| **BLEU-4**  | 4-gram precision — penalises short captions heavily               |
| **METEOR**  | Alignment using stemming + synonym matching                       |
| **ROUGE-L** | Longest common subsequence between generated and reference        |
| **CIDEr**   | Consensus-based — rewards captions similar to all 5 references    |

> CIDEr is the primary metric in most captioning papers. Target: **CIDEr > 0.8** on Flickr30k.

---

### Evaluation Process

1. Load best checkpoint: `model.load_state_dict(torch.load('src/checkpoints/best_model.pth'))`
2. Set model to `eval()` mode
3. For each image in the **test set** (1,000 images):
   - Generate a caption using **beam search** (beam size = 5)
   - Compare against all 5 reference captions
4. Compute BLEU-4, CIDEr, METEOR, ROUGE-L using `pycocoevalcap`

### Qualitative Evaluation

Display 10–20 samples showing:
- The input image
- Generated caption
- All 5 reference captions
- Per-sample BLEU-4 score

Include **failure cases** — what the model gets wrong and why.

---

## 10. Inference Pipeline

**Notebook**: `notebooks/06_inference_demo.ipynb`

### How Caption Generation Works

At inference, no ground-truth captions are provided. The model generates token-by-token:

```
Input Image → ViT → image_features → projection → image_prefix

Step 1: image_prefix              → P(next) → "A"
Step 2: image_prefix + "A"        → P(next) → "dog"
Step 3: image_prefix + "A dog"    → P(next) → "runs"
...
Step N: image_prefix + "A dog runs..." → <EOS>

Output: "A dog runs through a field of grass."
```

---

### Decoding Strategies

| Strategy           | Description                                           | Quality   |
|--------------------|-------------------------------------------------------|-----------|
| **Greedy**         | Always pick the highest-probability next token        | Fast, OK  |
| **Beam Search**    | Keep top-K sequences at each step (beam=5 recommended)| Best      |
| **Top-p Sampling** | Sample from top cumulative probability mass (p=0.9)   | Diverse   |

---

### Inference Function

```python
def generate_caption(image_path, model, processor, tokenizer, device,
                     beam_size=5, max_new_tokens=64):
    model.eval()
    image        = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        image_features = model.vit(pixel_values=pixel_values).last_hidden_state
        image_prefix   = model.projection(image_features)  # (1, 197, hidden_dim)

    generated_ids = model.gpt2.generate(
        inputs_embeds  = image_prefix,
        max_new_tokens = max_new_tokens,
        num_beams      = beam_size,
        early_stopping = True,
        eos_token_id   = tokenizer.eos_token_id,
        pad_token_id   = tokenizer.pad_token_id,
        no_repeat_ngram_size = 3,
    )
    caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return caption
```

---

## 11. Notebook Breakdown

| Notebook | Stage | Contents |
|---|---|---|
| `01_data_exploration.ipynb` | Explore | Load CSV, caption stats, vocab size, sample visualisations, corrupt file check |
| `02_data_preprocessing.ipynb` | Preprocess | ViTImageProcessor, GPT-2 tokenizer, Dataset class, DataLoaders, sanity checks |
| `03_model_architecture.ipynb` | Build | ViT encoder, GPT-2 decoder, projection layer, full VisionNarrator class, parameter count |
| `04_training.ipynb` | Train | 3-phase training loop, loss curves, LR schedule, checkpointing |
| `05_evaluation.ipynb` | Evaluate | BLEU/CIDEr/METEOR/ROUGE, qualitative examples, failure analysis |
| `06_inference_demo.ipynb` | Infer | Load checkpoint, generate captions on new images, beam search demo |

---

## 12. Tools and Libraries

| Tool / Library        | Purpose                                                  |
|-----------------------|----------------------------------------------------------|
| `torch` + `torchvision` | Core deep learning framework + image transforms        |
| `transformers`        | ViTModel, GPT2LMHeadModel, ViTImageProcessor, tokenizers |
| `accelerate`          | Mixed-precision training, device management              |
| `Pillow`              | Image loading and augmentation                           |
| `pandas` / `numpy`    | Data wrangling                                           |
| `matplotlib` / `seaborn` | Visualisation                                         |
| `nltk`                | BLEU tokenisation helper                                 |
| `pycocoevalcap`       | BLEU, CIDEr, METEOR, ROUGE-L evaluation metrics          |
| `scikit-learn`        | Train / val / test split                                 |
| `tqdm`                | Progress bars                                            |

---

## Pipeline at a Glance

```
Flickr30k (Images + 5 Captions each)
              │
              ▼
   ┌─ 1. Data Exploration ──────────────────────────┐
   │  Caption length, vocab stats, sample viz       │
   │  Remove corrupt images                         │
   └────────────────────────────────────────────────┘
              │
              ▼
   ┌─ 2. Preprocessing ─────────────────────────────┐
   │  Images: resize 224×224, normalise             │
   │  Text:   tokenise, add SOS/EOS, pad/mask       │
   │  Dataset class + DataLoaders                   │
   └────────────────────────────────────────────────┘
              │
              ▼
   ┌─ 3. Model Architecture ────────────────────────┐
   │  ViT Encoder  →  (batch, 197, 768)             │
   │        ↓  Projection Layer (Linear/MLP)        │
   │  GPT-2 Decoder → caption logits                │
   │        ↓  CrossEntropy Loss                    │
   └────────────────────────────────────────────────┘
              │
              ▼
   ┌─ 4. Transfer Learning (3 Phases) ──────────────┐
   │  Phase 1: Train projection only (LR=1e-3)      │
   │  Phase 2: Train projection + GPT-2 top layers  │
   │           (LR=5e-5)                            │
   │  Phase 3: Full fine-tune, all layers (LR=1e-5) │
   └────────────────────────────────────────────────┘
              │
              ▼
   ┌─ 5. Evaluation ────────────────────────────────┐
   │  BLEU-4, CIDEr, METEOR, ROUGE-L               │
   │  Qualitative sample analysis                   │
   │  Failure case study                            │
   └────────────────────────────────────────────────┘
              │
              ▼
   ┌─ 6. Inference ─────────────────────────────────┐
   │  Load best checkpoint                          │
   │  Beam Search decoding (beam=5)                 │
   │  Generate captions for new, unseen images      │
   └────────────────────────────────────────────────┘
```

---

> **Where to start**: Open `notebooks/01_data_exploration.ipynb`.
> Load the Flickr30k CSV from `src/data/results.csv`, check the structure,
> and visualise a few image-caption pairs. That will confirm the data is intact
> and ready for the preprocessing stage.
