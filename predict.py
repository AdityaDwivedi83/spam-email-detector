import joblib

from preprocessing import clean_text


vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("spam_model.pkl")


def predict_spam(message):
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)[0]

    return "SPAM" if prediction == 1 else "HAM"

