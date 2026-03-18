# KmalPred

KmalPred is a protein language model–based framework for **lysine malonylation (Kmal) site prediction**.  
It uses **ProtT5 residue-level embeddings** (sequence-first encoding) and a **BiLSTM classifier** to predict Kmal sites from protein sequences.

This repository contains scripts for:
1) extracting ProtT5 per-residue embeddings,
2) constructing lysine-centered positive and negative windows from site annotations,
3) training and evaluating the prediction model.

## Requirements

- Python 3.9
- PyTorch 2.5.1 
- transformers 4.45.1
- sentencepiece 0.2.1
- huggingface-hub 0.36.2
- numpy 2.0.1
- pandas 2.3.3
- scikit-learn 1.6.1
- scipy 1.13.1
- tqdm 4.67.1

**Data Preparation**
1) FASTA for embeddings
Prepare FASTA files such as:
```bash
data/Train.fasta
data/Test.fasta
```
FASTA headers (protein IDs) must match the IDs used in your window index files.

2) Excel for site annotations
Prepare Excel files such as:
```bash
data/Train.xlsx
data/Test.xlsx
```
Required columns (your scripts will rename them internally):
Uniprot Accession
Position (1-based site position)
Sequence (full protein sequence)

**Step 1: Extract ProtT5 Embeddings (per-residue)**
```bash
python scripts/embed_prott5.py \
  --fasta data/Test.fasta \
  --out data/test.pkl \
  --model_name_or_path Rostlab/prot_t5_xl_uniref50 \
  --device cuda:0 \
  --fp16
```


Do the same for training:

```bash
python scripts/embed_prott5.py \
  --fasta data/Train.fasta \
  --out data/train.pkl \
  --model_name_or_path Rostlab/prot_t5_xl_uniref50 \
  --device cuda:0 \
  --fp16
```
**Step 2: Build Positive / Negative Windows**

Example:
```bash
python scripts/build_negative_windows.py \
  --excel data/Test.xlsx --sheet Sheet1 \
  --out data/test_negative.pkl \
  --win 35 --shuffle --seed 42
```
  Balanced 1:1 negatives (recommended for Setting 1)
If you want negatives to match the number of positives:
```bash
python scripts/build_negative_windows.py \
  --excel data/Train.xlsx --sheet Sheet1 \
  --out data/train_negative_1to1.pkl \
  --win 35 --shuffle --seed 42 \
  --match_positives data/train_positive.pkl
```

**Step 3: Train & Evaluate**
Run training from the repo root:
```bash
python scripts/main.py \
  --train_emb data/train.pkl --train_pos data/train_positive.pkl --train_neg data/train_negative.pkl \
  --test_emb  data/test.pkl  --test_pos  data/test_positive.pkl  --test_neg  data/test_negative.pkl \
  --epochs 50 --batch_size 256 --device cuda:0 \
  --use_focal --pos_weight 4.0 \
  --outdir runs/setting1 --save_best
```
