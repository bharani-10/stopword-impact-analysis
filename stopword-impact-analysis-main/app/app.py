import streamlit as st
import joblib
import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

# -----------------------------
# FIX NLTK DOWNLOAD ISSUE (IMPORTANT)
# -----------------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# -----------------------------
# LOAD MODELS (DEPLOYMENT SAFE)
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

model_with = joblib.load(os.path.join(BASE_DIR, "models", "model_with_stopwords.pkl"))
model_without = joblib.load(os.path.join(BASE_DIR, "models", "model_without_stopwords.pkl"))

vectorizer_with = joblib.load(os.path.join(BASE_DIR, "models", "vectorizer_with.pkl"))
vectorizer_without = joblib.load(os.path.join(BASE_DIR, "models", "vectorizer_without.pkl"))

# -----------------------------
# FUNCTIONS
# -----------------------------

def tokenize_text(text):
    return word_tokenize(text)

def remove_stopwords(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words]

def predict_with_stopwords(text):
    vec = vectorizer_with.transform([text])
    pred = model_with.predict(vec)

    if pred[0] == 1:
        return "Spam", "Message contains promotional or suspicious keywords."
    else:
        return "Ham", "Message appears normal and conversational."

def predict_without_stopwords(text):
    vec = vectorizer_without.transform([text])
    pred = model_without.predict(vec)

    if pred[0] == 1:
        return "Spam", "Important spam keywords remain even after removing stopwords."
    else:
        return "Ham", "Remaining keywords indicate normal communication."

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Stopword Analysis", layout="centered")

st.title("📩 Stopword Removal Impact Analysis")
st.write("Compare predictions with and without stopwords.")

user_text = st.text_area("Enter your message here")

if st.button("Analyze Text"):

    if user_text.strip() == "":
        st.warning("⚠ Please enter a message")
    else:

        # -----------------------------
        # ORIGINAL TEXT
        # -----------------------------
        st.header("📌 Original Text")
        st.write(user_text)

        # -----------------------------
        # WITH STOPWORDS
        # -----------------------------
        st.header("🔹 With Stopwords")

        text_with = user_text.lower()
        tokens_with = tokenize_text(text_with)

        label_with, reason_with = predict_with_stopwords(text_with)

        st.subheader("Text")
        st.write(text_with)

        st.subheader("Tokenized Words")
        st.write(tokens_with)

        st.subheader("Prediction")
        st.success(label_with)

        st.subheader("Reason")
        st.info(reason_with)

        # -----------------------------
        # WITHOUT STOPWORDS
        # -----------------------------
        st.header("🔹 Without Stopwords")

        filtered_tokens = remove_stopwords(user_text)
        text_without = " ".join(filtered_tokens)

        label_without, reason_without = predict_without_stopwords(text_without)

        st.subheader("Text")
        st.write(text_without)

        st.subheader("Tokenized Words")
        st.write(filtered_tokens)

        st.subheader("Prediction")
        st.success(label_without)

        st.subheader("Reason")
        st.info(reason_without)

        # -----------------------------
        # GRAPH
        # -----------------------------
        st.markdown("## 📊 Stopword Impact Graph")

        tokens_with_count = len(tokens_with)
        tokens_without_count = len(filtered_tokens)

        labels = ["With Stopwords", "Without Stopwords"]
        values = [tokens_with_count, tokens_without_count]

        fig, ax = plt.subplots()
        ax.bar(labels, values)

        ax.set_title("Token Count Comparison")
        ax.set_ylabel("Number of Tokens")

        st.pyplot(fig)