# Customer Sentiment Analysis in Support Chat Using Machine Learning and Deep Learning

This project investigates how AI-based sentiment analysis methods can be used to detect early signs of customer dissatisfaction in customer support conversations. Leveraging weak supervision, traditional ML models (SVM, Random Forest), and deep learning (BiLSTM), the system labels and classifies chat messages from real-world customer service interactions.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Results](#results)
- [Limitations](#limitations)
- [Future Work](#future-work)

## Overview

The goal is to build an AI-driven sentiment classification system that can:

- Detect dissatisfaction early in support chats.
- Compare ML and DL models trained on weakly labeled data.
- Use real-world customer-agent chat logs from Twitter.

This work answers the research question: **"How well do machine learning and deep learning models perform at predicting customer dissatisfaction in support chats using weak supervision techniques?"**

## Dataset

- **Name**: Customer Support on Twitter
- **Source**: Kaggle
- **Link**: [https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- **Description**: Contains over 3 million tweets from customers and companies engaged in support interactions.

Due to file size, the dataset is not included. Please download it from Kaggle and place it in the project root folder as `twcs.csv`.

## Approach

1. **Data Preprocessing**:

   - Removed missing/irrelevant entries.
   - Cleaned text (URLs, punctuation, emojis).
   - Filtered only inbound (customer) messages.

2. **Sentiment Labelling (Weak Supervision)**:

   - VADER (lexicon-based)
   - DistilBERT (contextual transformer)
   - RoBERTa (pretrained on Twitter data)
   - Merged labels using hybrid voting rules.

3. **Manual Review**:

   - 500 messages manually labelled and evaluated.

4. **Model Training**:

   - Traditional: SVM, Random Forest (TF-IDF features)
   - Deep Learning: BiLSTM (tokenized sequences, class weights)

5. **Evaluation Metrics**:
   - Accuracy, Precision, Recall
   - Macro & Weighted F1-score
   - Confusion Matrices

## Project Structure

```
├── data/                            # Data directory (download dataset here)
├── labelled_data_final.csv          # Final labelled dataset used for training
├── labelled_data_with_roberta.pkl   # Serialized labelled DataFrame
├── labelling.ipynb                  # Label generation and cleaning pipeline
├── ml_models.ipynb                  # SVM and Random Forest training
├── dl_models.ipynb                  # BiLSTM training and evaluation
├── evaluation.ipynb                 # Optional notebook for extended evaluation
├── manual_review_sample.csv         # Sample of messages for manual review
├── README.md                        # Project README (you are here)
└── requirements.txt                 # Python dependencies
```

## Requirements

- Python 3.10+
- Anaconda or virtualenv recommended

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

1. Download the dataset and rename it to `twcs.csv`.
2. Open and run `labelling.ipynb` to:

   - Clean the data
   - Label messages with weak supervision
   - Save `labelled_data_final.csv`

3. Run `ml_models.ipynb` to train and evaluate SVM and Random Forest.
4. Run `dl_models.ipynb` to train and evaluate the BiLSTM model.

**Note**: If using limited hardware, consider reducing the dataset size in DL.

## Results

| Model         | Accuracy | Macro F1 | Weighted F1 |
| ------------- | -------- | -------- | ----------- |
| SVM           | 88%      | 0.88     | 0.89        |
| Random Forest | 86%      | 0.85     | 0.87        |
| **BiLSTM**    | **90%**  | **0.90** | **0.90**    |

- **BiLSTM** showed the best contextual understanding.
- SVM remained a strong, efficient baseline.

## Limitations

- Dataset imbalance (more negative samples).
- Weak supervision introduces label noise.
- Sarcasm and polite dissatisfaction are still hard to detect.

## Future Work

- Integrate Explainable AI (e.g., SHAP, LIME) for model transparency.
- Add sarcasm detection using context-aware transformers.
- Apply active learning to improve label quality.
- Deploy the model into a real-time system or dashboard.

---

Developed for the Machine Learning module of the MSc in Artificial Intelligence at the National College of Ireland.
