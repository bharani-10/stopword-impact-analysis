# Stopword Removal Impact Analysis

A Natural Language Processing (NLP) project that investigates how removing stopwords affects text classification performance. The project compares two classifiers — **Naive Bayes** and **Logistic Regression** — using two feature extraction strategies — **Bag of Words (BoW)** and **TF-IDF** — with and without stopword removal. An interactive **Streamlit** web app lets you explore the results in real time.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Example Output](#example-output)
- [Conclusion](#conclusion)

---

## Project Overview

Stopwords (e.g., *the*, *is*, *in*) are commonly filtered out during text preprocessing because they are assumed to add little semantic value. However, their actual impact on classification accuracy varies by dataset and model.

This project provides a systematic, side-by-side comparison:

| Setting | Vectorizer | Classifier |
|---|---|---|
| With stopwords | Bag of Words | Naive Bayes |
| Without stopwords | Bag of Words | Naive Bayes |
| With stopwords | TF-IDF | Logistic Regression |
| Without stopwords | TF-IDF | Logistic Regression |

The results are visualised through accuracy scores, confusion matrices, and classification reports so you can draw data-driven conclusions about stopword removal.

---

## Features

- **Text Preprocessing** — tokenization, lowercasing, punctuation removal, and optional stopword filtering using NLTK
- **Feature Extraction** — Bag of Words (`CountVectorizer`) and TF-IDF (`TfidfVectorizer`) representations
- **Model Training** — Naive Bayes (`MultinomialNB`) and Logistic Regression classifiers
- **Evaluation** — accuracy score, classification report (precision / recall / F1), and confusion matrix for every combination
- **Comparison Dashboard** — side-by-side comparison of all four experimental settings
- **Interactive Streamlit App** — enter custom text and instantly see classification results with and without stopwords

---

## Folder Structure

```
stopword-impact-analysis/
│
├── data/
│   ├── raw/                  # Original, unprocessed dataset
│   └── processed/            # Cleaned and preprocessed data
│
├── notebooks/
│   └── analysis.ipynb        # Exploratory data analysis and experiments
│
├── src/
│   ├── preprocess.py         # Text cleaning and stopword removal utilities
│   ├── features.py           # BoW and TF-IDF vectorisation
│   ├── train.py              # Model training (Naive Bayes, Logistic Regression)
│   └── evaluate.py           # Metrics, confusion matrices, result comparison
│
├── app/
│   └── streamlit_app.py      # Interactive Streamlit web application
│
├── results/
│   └── metrics.csv           # Saved accuracy and F1 scores for all experiments
│
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/bharani-10/stopword-impact-analysis.git
   cd stopword-impact-analysis
   ```

2. **Create and activate a virtual environment** *(recommended)*

   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK stopwords corpus**

   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

---

## How to Run

### 1. Preprocess the data

```bash
python src/preprocess.py
```

This script reads the raw dataset, applies text cleaning, and saves two versions of the data — one with stopwords retained and one with stopwords removed — to the `data/processed/` directory.

### 2. Train and evaluate models

```bash
python src/train.py
```

Trains all four classifier + vectoriser combinations and prints accuracy scores and classification reports to the console. Results are also saved to `results/metrics.csv`.

### 3. Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`) to access the interactive dashboard.

---

## Example Output

### Accuracy Comparison

| Vectorizer | Classifier | With Stopwords | Without Stopwords |
|---|---|---|---|
| Bag of Words | Naive Bayes | 85.3% | 86.1% |
| TF-IDF | Logistic Regression | 88.7% | 89.4% |

### Classification Report (TF-IDF + Logistic Regression, without stopwords)

```
              precision    recall  f1-score   support

    negative       0.90      0.88      0.89       250
    positive       0.89      0.91      0.90       250

    accuracy                           0.89       500
   macro avg       0.89      0.89      0.89       500
weighted avg       0.89      0.89      0.89       500
```

### Streamlit App

The web app allows you to:

- Type or paste any text into the input box
- Select a classifier (Naive Bayes / Logistic Regression) and vectoriser (BoW / TF-IDF)
- View the predicted label and confidence score with and without stopword removal side by side

---

## Conclusion

The experiments consistently show that **removing stopwords provides a modest but measurable improvement** in classification accuracy across both models and both vectorisation strategies. Key takeaways:

- **TF-IDF + Logistic Regression** achieves the highest overall accuracy and benefits most from stopword removal.
- **Naive Bayes + Bag of Words** is computationally lightweight and still competitive after stopword removal.
- The improvement from stopword removal is more pronounced with **Bag of Words** than with **TF-IDF**, because TF-IDF already down-weights high-frequency terms (which include many stopwords) by design.
- For production NLP pipelines, stopword removal is generally recommended as a low-cost preprocessing step that reduces vocabulary size and can improve both speed and accuracy.

Feel free to open an issue or submit a pull request if you would like to contribute!

