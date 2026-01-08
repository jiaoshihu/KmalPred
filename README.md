# KmalPred

KmalPred is a protein language model–based framework for **lysine malonylation (Kmal) site prediction**.  
It uses **ProtT5 residue-level embeddings** (sequence-first encoding) and a **BiLSTM classifier** to predict Kmal sites from protein sequences.

This repository contains scripts for:
1) extracting ProtT5 per-residue embeddings,
2) building positive/negative lysine-centered windows from Excel annotations,
3) training and evaluating a BiLSTM predictor.
