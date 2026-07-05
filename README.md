# 📧 Spam Message Detector

A machine learning web application that classifies text messages as **Spam** or **Ham (Not Spam)**.

The project uses TF-IDF for text vectorization and a tuned Multinomial Naive Bayes classifier.

## 🚀 Features

- Classifies messages as Spam or Ham
- Text preprocessing using NLTK
- TF-IDF feature extraction
- Multinomial Naive Bayes classification
- Hyperparameter tuning
- Interactive Streamlit web interface

## 🧠 Machine Learning Pipeline

Raw Message
→ Lowercasing
→ Tokenization
→ Punctuation Removal
→ Stopword Removal
→ TF-IDF Vectorization
→ Multinomial Naive Bayes
→ Spam / Ham

## 📊 Model Performance

The final model uses:

- Algorithm: Multinomial Naive Bayes
- Alpha: 0.1
- Accuracy: ~98%
- Spam Precision: 99%
- Spam Recall: 88%
- Spam F1-score: 93%

Confusion Matrix:

| | Predicted Ham | Predicted Spam |
|---|---:|---:|
| Actual Ham | 965 | 1 |
| Actual Spam | 18 | 131 |

The model was compared with Logistic Regression, but the tuned Naive Bayes model achieved significantly better spam recall.

## 🛠️ Technologies Used

- Python
- pandas
- scikit-learn
- NLTK
- Streamlit
- Matplotlib
- Joblib

## 📁 Project Structure

```text
SPAM-EMAIL-DETECTOR/
├── data/
│   └── spam.csv
├── app.py
├── notebook.ipynb
├── predict.py
├── train.py
├── spam_model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md