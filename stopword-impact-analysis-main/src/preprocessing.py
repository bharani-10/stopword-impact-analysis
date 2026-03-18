import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load dataset
df = pd.read_csv("data/clean_dataset.csv")

# Download resources if needed
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Function for cleaning
def remove_stopwords(text):

    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = word_tokenize(text)

    filtered = [word for word in tokens if word not in stop_words]

    return " ".join(filtered)

# Text with stopwords
df['text_with_stopwords'] = df['text'].str.lower()

# Text without stopwords
df['text_without_stopwords'] = df['text'].apply(remove_stopwords)

# Save dataset
df.to_csv("data/processed_dataset.csv", index=False)

print("Preprocessing completed")