import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/processed_dataset.csv")

# Remove empty rows
df = df.dropna()

# Convert labels
df['label'] = df['label'].map({'ham':0, 'spam':1})

y = df['label']

# -----------------------------
# WITH STOPWORDS
# -----------------------------

vectorizer1 = TfidfVectorizer()

X1 = vectorizer1.fit_transform(df['text_with_stopwords'])

X_train1, X_test1, y_train, y_test = train_test_split(X1, y, test_size=0.2)

model1 = MultinomialNB()
model1.fit(X_train1, y_train)

joblib.dump(model1, "models/model_with_stopwords.pkl")
joblib.dump(vectorizer1, "models/vectorizer_with.pkl")

# -----------------------------
# WITHOUT STOPWORDS
# -----------------------------

vectorizer2 = TfidfVectorizer()

X2 = vectorizer2.fit_transform(df['text_without_stopwords'])

X_train2, X_test2, y_train, y_test = train_test_split(X2, y, test_size=0.2)

model2 = LogisticRegression(max_iter=200)
model2.fit(X_train2, y_train)

joblib.dump(model2, "models/model_without_stopwords.pkl")
joblib.dump(vectorizer2, "models/vectorizer_without.pkl")

print("Models trained and saved")