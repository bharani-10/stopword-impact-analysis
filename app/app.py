import streamlit as st
import joblib
import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

# -----------------------------
# Download nltk resources
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# -----------------------------
# Load models
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_with = joblib.load(os.path.join(BASE_DIR,"models","model_with_stopwords.pkl"))
model_without = joblib.load(os.path.join(BASE_DIR,"models","model_without_stopwords.pkl"))

vectorizer_with = joblib.load(os.path.join(BASE_DIR,"models","vectorizer_with.pkl"))
vectorizer_without = joblib.load(os.path.join(BASE_DIR,"models","vectorizer_without.pkl"))

# -----------------------------
# Functions
# -----------------------------

def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

def remove_stopwords(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    return filtered

def predict_with_stopwords(text):

    vec = vectorizer_with.transform([text])
    pred = model_with.predict(vec)

    if pred[0] == 1:
        label = "Spam"
        reason = "Message contains promotional or suspicious keywords."
    else:
        label = "Ham"
        reason = "Message appears normal and conversational."

    return label, reason

def predict_without_stopwords(text):

    vec = vectorizer_without.transform([text])
    pred = model_without.predict(vec)

    if pred[0] == 1:
        label = "Spam"
        reason = "Important spam keywords remain even after removing stopwords."
    else:
        label = "Ham"
        reason = "Remaining keywords indicate normal communication."

    return label, reason


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("Stopword Removal Impact Analysis")

st.write("This app demonstrates how stopword removal affects NLP predictions.")

user_text = st.text_area("Enter a message")

if st.button("Analyze Text"):

    if user_text.strip() == "":
        st.warning("Please enter a message")
    
    else:

        # Original text
        st.header("Original Text")
        st.write(user_text)

        # -----------------------------
        # WITH STOPWORDS
        # -----------------------------
        st.header("WITH STOPWORDS")

        text_with = user_text.lower()

        st.subheader("Text")
        st.write(text_with)

        tokens_with = tokenize_text(text_with)

        st.subheader("Tokenized Words")
        st.write(tokens_with)

        label_with, reason_with = predict_with_stopwords(text_with)

        st.subheader("Prediction")
        st.write(label_with)

        st.subheader("Reason")
        st.write(reason_with)

        # -----------------------------
        # WITHOUT STOPWORDS
        # -----------------------------
        st.header("WITHOUT STOPWORDS")

        filtered_tokens = remove_stopwords(user_text)

        text_without = " ".join(filtered_tokens)

        st.subheader("Text")
        st.write(text_without)

        st.subheader("Tokenized Words")
        st.write(filtered_tokens)

        label_without, reason_without = predict_without_stopwords(text_without)

        st.subheader("Prediction")
        st.write(label_without)

        st.subheader("Reason")
        st.write(reason_without)

# -----------------------------
# Graph: Stopword Impact
# -----------------------------
        # -----------------------------
        # Graph: Stopword Impact
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