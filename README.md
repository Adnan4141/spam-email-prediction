# Spam Email Detection with Deep Learning (Jupyter Notebook)

**A reproducible Jupyter notebook for detecting spam emails using deep learning with TensorFlow/Keras.**  
This repository demonstrates a workflow for building an attention-based BiLSTM model for spam detection.

---

## Table of contents
- [Project overview](#project-overview)  
- [Notebook highlights](#notebook-highlights)  
- [Repository structure](#repository-structure)  
- [Getting started](#getting-started)  
  - [Requirements](#requirements)  
  - [Install](#install)  
  - [Run the notebook (locally / Colab)](#run-the-notebook-locally--colab)  
- [How the pipeline works](#how-the-pipeline-works)  
- [Model architecture](#model-architecture)  
- [Training & evaluation](#training--evaluation)  
- [Reproducing results](#reproducing-results)  
- [Next steps / deployment ideas](#next-steps--deployment-ideas)  
- [Credits & dataset(s)](#credits--datasets)  
- [License](#license)  
- [Contact](#contact)

---

## Project overview
This notebook demonstrates a practical spam email detector using **deep learning**.  
It includes text preprocessing, tokenization, padding sequences, and an **attention-based BiLSTM model**.  
It is intended as a reproducible demo and learning resource; it is **not production-grade by default**.

---

## Notebook highlights
- Text cleaning with NLTK: stopword removal, lemmatization, removing emails/URLs/punctuation
- Tokenization and sequence padding using `tensorflow.keras.preprocessing.text.Tokenizer`
- Train/test split with class imbalance handled using **class weights**
- Deep learning model:
  - Embedding layer
  - Bidirectional LSTM
  - Custom **attention layer**
  - Dropout and Dense output layer with sigmoid activation
- Callbacks: `EarlyStopping` and `ReduceLROnPlateau`
- Model evaluation: accuracy, classification report
- Visualization: training/validation accuracy & loss curves
- Prediction pipeline for new texts

---

