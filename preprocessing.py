import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def clean_text(text):
    text = text.lower()

    words = word_tokenize(text)

    words = [
        word for word in words
        if word not in string.punctuation
    ]

    words = [
        word for word in words
        if word not in stopwords.words("english")
    ]

    return words