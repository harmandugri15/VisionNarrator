# VisionNarrator
VisionNarrator is an AI image captioning system that combines a Vision Transformer (ViT) with a pretrained language model . Using transfer learning and the Flickr30k dataset , it extracts visual features and generates natural-language descriptions of what is happening in an image .

# Note
The dataset used is Flickr30k ( ~31,783 images , 5 captions each ) . It has been downloaded locally into /src/data/ and should NOT be pushed to GitHub ( covered by .gitignore ) .

Use a virtual environment with python version >= 3.11.9 . (Recommended)

Known data quirk — image 2199200615.jpg has only 4 captions instead of 5 in the raw captions.txt ( one blank row ) . This is handled automatically in notebook 02 by dropping the empty row before splitting .

On Windows — set NUM_WORKERS = 0 in the DataLoader . Using num_workers > 0 on Windows causes a multiprocessing deadlock because Windows uses spawn instead of fork . This is already handled automatically in the notebooks via platform.system() detection .


# What Has Been Done So Far

### 01 — Data Exploration ( notebooks/data_exploration.ipynb )
- Dataset downloaded via kagglehub into src/data/
- Confirmed 31,783 unique images and 158,915 captions ( 5 per image , exactly )
- Confirmed 0 missing image files
- Caption length analysis — mean ~11 words , 99th percentile ~23 words → max_length = 64 tokens decided
- Vocabulary analysis — ~15,000 unique words → GPT-2 BPE tokenizer chosen ( no UNK problem )
- Image dimension check — mostly non-square , all convertible to RGB → ViTImageProcessor handles resize to 224x224

### 02 — Data Preprocessing ( notebooks/02_data_preprocessing.ipynb )
- Loaded and cleaned data ( dropped 1 NaN caption row )
- Created train / val / test split at image level ( no data leakage )
  - train.csv — ~30,950 images
  - val.csv   — ~1,085 images
  - test.csv  — ~748 images
- Set up ViTImageProcessor from google/vit-base-patch16-224-in21k
- Set up GPT-2 tokenizer with SOS / EOS / PAD special tokens
- Defined Flickr30kDataset PyTorch class ( returns pixel_values , input_ids , attention_mask , labels )
- Created DataLoaders ( batch_size=32 )
- Verified batch shapes — pixel_values (32,3,224,224) , input_ids (32,64) , labels (32,64)

### Still To Do
- 03 — Model Architecture ( ViT encoder + projection layer + GPT-2 decoder )
- 04 — Training ( 3-phase transfer learning )
- 05 — Evaluation ( BLEU-4 , CIDEr , METEOR , ROUGE-L )
- 06 — Inference demo


# Understanding Directory Structure
/notebooks — Individual Jupyter notebooks , one per pipeline stage .

/src/data — Flickr30k images and captions ( local only , not pushed to GitHub ) . Also stores train.csv / val.csv / test.csv split files and model checkpoints .

/src/checkpoints — Saved model weights during training .

/docs — Will store model documentation and evaluation results .

plan.md — Full pipeline plan with architecture details , training strategy , and evaluation metrics .