import joblib
import nltk
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import clean_text


# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


# Load dataset
df = pd.read_csv("data/spam.csv", encoding="latin-1")

df = df[["v1", "v2"]]
df.columns = ["label", "message"]

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


# Split RAW text before fitting TF-IDF
X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create and fit vectorizer
vectorizer = TfidfVectorizer(
    tokenizer=clean_text,
    token_pattern=None
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Train final model
model = MultinomialNB(alpha=0.1)

model.fit(X_train_tfidf, y_train)


# Evaluate
predictions = model.predict(X_test_tfidf)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))


# Save trained objects
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(model, "spam_model.pkl")

print("\nModel and vectorizer saved successfully.")