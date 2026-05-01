import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import load_model # type: ignore

# -------------------------------
# Load tokenizer and label encoder
# -------------------------------
with open(r"D:\Interview\Twitter_Sentiment_Anallysis\Pickle\tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open(r"D:\Interview\Twitter_Sentiment_Anallysis\Pickle\label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# -------------------------------
# Load trained models
# -------------------------------
rnn_model = load_model(r"D:\Interview\Twitter_Sentiment_Anallysis\Trained_model\rnn_twitter_sentiment_tuned.keras")
lstm_model = load_model(r"D:\Interview\Twitter_Sentiment_Anallysis\Trained_model\lstm_twitter_sentiment_tuned.keras")
gru_model = load_model(r"D:\Interview\Twitter_Sentiment_Anallysis\Trained_model\gru_twitter_sentiment_tuned.keras")

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Twitter Sentiment Analysis")
st.write("Compare predictions from RNN, LSTM, and GRU models.")

tweet = st.text_input("Enter a tweet:")

if tweet:
    # Preprocess input
    seq = tokenizer.texts_to_sequences([tweet])
    pad = pad_sequences(seq, maxlen=60)  # must match training max_len

    # Predict with each model
    rnn_pred = np.argmax(rnn_model.predict(pad), axis=1)[0]
    lstm_pred = np.argmax(lstm_model.predict(pad), axis=1)[0]
    gru_pred = np.argmax(gru_model.predict(pad), axis=1)[0]

    # Decode predictions
    rnn_label = le.inverse_transform([rnn_pred])[0]
    lstm_label = le.inverse_transform([lstm_pred])[0]
    gru_label = le.inverse_transform([gru_pred])[0]

    # Show results
    st.subheader("Predictions")
    st.write(f"**RNN:** {rnn_label}")
    st.write(f"**LSTM:** {lstm_label}")
    st.write(f"**GRU:** {gru_label}")
