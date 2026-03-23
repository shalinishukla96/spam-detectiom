# 📧 Spam Detection using Machine Learning

> End-to-end NLP + ML pipeline for SMS/email spam classification | MSc Data Science Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project builds a robust, reproducible spam detection system using classical NLP and machine learning. Three classifiers are compared using proper cross-validation, with full exploratory data analysis, preprocessing, and evaluation visualisations.

| Component | Details |
|---|---|
| **Task** | Binary text classification (spam / ham) |
| **Features** | TF-IDF (unigrams + bigrams, 10k features) |
| **Models** | Multinomial Naive Bayes · Logistic Regression · Random Forest |
| **Evaluation** | Accuracy · Precision · Recall · F1 · AUC-ROC · 5-fold CV |

---

## Project Structure

```
spam-detection/
│
├── spam_detection.py       # ← Main script (all code in one file)
├── requirements.txt
├── README.md
│
├── data/
│   └── SMSSpamCollection   # ← Place your dataset here
│
└── plots/                  # ← Auto-generated figures (if SAVE_PLOTS=True)
    ├── 01_class_distribution.png
    ├── 02_length_distribution.png
    ├── 03_boxplots.png
    ├── 04_top_words.png
    ├── 05_wordclouds.png
    ├── 06_correlation.png
    ├── 07_*_eval.png
    ├── 08_model_comparison.png
    └── 09_*_features.png
```

---

## Dataset

This project is pre-configured for the **UCI SMS Spam Collection** dataset.

**Download:** https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

Place the file at `data/SMSSpamCollection` (tab-separated, no header).

### Using your own dataset

The loader auto-detects common column names. Your CSV just needs:
- A **label column** named one of: `label`, `class`, `category`, `spam`, `v1`
- A **text column** named one of: `text`, `message`, `sms`, `email`, `body`, `content`, `v2`

Then update the path in `main()`:
```python
DATASET_PATH = "data/your_file.csv"
```

---

## Installation

```bash
git clone https://github.com/<your-username>/spam-detection.git
cd spam-detection
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt
```

---

## Usage

```bash
python spam_detection.py
```

To save all plots as PNG files, set `SAVE_PLOTS = True` in `main()`.

---

## EDA Highlights

The pipeline generates 9 figures covering:
1. Class distribution (counts + pie chart)
2. Message length distributions (characters & words)
3. Box plots of numeric features by class
4. Top-20 most frequent words per class
5. Word clouds (ham vs spam)
6. Feature correlation heatmap
7. Per-model confusion matrices & ROC curves
8. Cross-model metric comparison
9. Top TF-IDF features for the best model

---

## Models & Results

All three models share the same TF-IDF vectoriser (10k features, 1–2 grams):

| Model | Typical Accuracy | Typical F1 (spam) |
|---|---|---|
| Multinomial Naive Bayes | ~97–98% | ~96–97% |
| Logistic Regression | ~98–99% | ~97–98% |
| Random Forest | ~97–98% | ~96–97% |

*Results vary slightly depending on dataset and random seed.*

---

## Key Design Decisions

- **TF-IDF over Bag-of-Words** — sublinear TF scaling reduces the impact of very frequent terms.
- **Bigrams included** — captures phrases like "free prize" and "click here".
- **Porter Stemming** — reduces vocabulary size and groups morphological variants.
- **Stratified splits** — preserves class ratio in both train and test folds, critical for imbalanced data.
- **F1 as primary metric** — more informative than accuracy for imbalanced classes.

---

## Extending the Project

Ideas for further work:

- [ ] Add deep learning model (LSTM / BERT via HuggingFace)
- [ ] Hyperparameter tuning with `GridSearchCV` or `Optuna`
- [ ] Deploy as REST API using FastAPI or Flask
- [ ] Add SHAP values for model explainability
- [ ] Experiment with character-level n-grams

---

## License

MIT © [Your Name] — feel free to use and adapt with attribution.
