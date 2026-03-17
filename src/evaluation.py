import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("data/processed_dataset.csv")

# Remove empty rows
df = df.dropna()

# Replace remaining NaN with empty string
df['text_without_stopwords'] = df['text_without_stopwords'].fillna("")
df['text_with_stopwords'] = df['text_with_stopwords'].fillna("")

# Convert labels
df['label'] = df['label'].map({'ham':0,'spam':1})

y = df['label']

vectorizer = TfidfVectorizer()

# WITH STOPWORDS
X1 = vectorizer.fit_transform(df['text_with_stopwords'])

X_train, X_test, y_train, y_test = train_test_split(X1,y,test_size=0.2)

model1 = joblib.load("models/model_with_stopwords.pkl")

pred1 = model1.predict(X_test)

print("\nWITH STOPWORDS")
print("Accuracy:", accuracy_score(y_test,pred1))
print("Precision:", precision_score(y_test,pred1))
print("Recall:", recall_score(y_test,pred1))
print("F1 Score:", f1_score(y_test,pred1))

# WITHOUT STOPWORDS
X2 = vectorizer.fit_transform(df['text_without_stopwords'])

X_train, X_test, y_train, y_test = train_test_split(X2,y,test_size=0.2)

model2 = joblib.load("models/model_without_stopwords.pkl")

pred2 = model2.predict(X_test)

print("\nWITHOUT STOPWORDS")
print("Accuracy:", accuracy_score(y_test,pred2))
print("Precision:", precision_score(y_test,pred2))
print("Recall:", recall_score(y_test,pred2))
print("F1 Score:", f1_score(y_test,pred2))

import matplotlib.pyplot as plt

models = ['With Stopwords','Without Stopwords']
accuracy = [accuracy_score(y_test,pred1), accuracy_score(y_test,pred2)]

plt.bar(models, accuracy)
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()