import streamlit as st

from predict import predict_spam


st.title("📧 Spam Message Detector")

st.write(
    "Enter a message below and the machine learning model "
    "will predict whether it is spam or ham."
)

message = st.text_area(
    "Enter your message:",
    height=180
)

if st.button("Detect Spam"):

    if not message.strip():
        st.warning("Please enter a message.")

    else:
        prediction = predict_spam(message)

        if prediction == "SPAM":
            st.error("🚨 SPAM")
        else:
            st.success("✅ HAM — Not Spam")