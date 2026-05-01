# Twitter Entity-Level Sentiment Analysis

## 📖 Project Overview
This project implements **entity-level sentiment analysis** on Twitter data.  
Given a tweet and an entity, the task is to classify the sentiment expressed toward that entity.  

The dataset contains three sentiment classes:
- **Positive**
- **Negative**
- **Neutral** (includes messages that are irrelevant to the entity)

We train and compare multiple deep learning models (RNN, LSTM, GRU) to evaluate performance on this classification task.

---

## ⚙️ Workflow

### 1. Data Preprocessing
- Remove mentions (`@username`), hashtags, and URLs
- Remove non-alphabetic symbols
- Lowercase normalization
- Stopword removal
- Lemmatization
- Tokenization and padding (max length = 60)

### 2. Label Encoding
- Sentiment labels (`Positive`, `Negative`, `Neutral`, `Irrelevant`) are encoded into integers using `LabelEncoder`.

### 3. Model Architectures
Implemented using TensorFlow/Keras:
- **RNN** (SimpleRNN layers with Dropout)
- **LSTM** (Bidirectional LSTM + stacked LSTM + Dropout)
- **GRU** (Bidirectional GRU + stacked GRU + Dropout)

Each model uses:
- Embedding layer (`vocab_size=5000`, embedding_dim=128)
- Dense output layer with `softmax` activation
- Loss: `sparse_categorical_crossentropy`
- Optimizer: Adam

### 4. Training
- Train/test split (80/20)
- Batch size: 64
- Epochs: 10
- Validation on test set

### 5. Evaluation
- Accuracy score
- Classification report (precision, recall, F1-score per class)

### 6. Model Saving
Models are saved in the modern Keras format:
- `rnn_twitter_sentiment_tuned.keras`
- `lstm_twitter_sentiment_tuned.keras`
- `gru_twitter_sentiment_tuned.keras`

Tokenizer and label encoder are saved with `pickle`:
- `tokenizer.pkl`
- `label_encoder.pkl`

---

## 🚀 Deployment Options

### Streamlit App
Interactive UI for entering tweets and comparing predictions across RNN, LSTM, and GRU models.

---

## 📊 Results
- **RNN Accuracy**: ~74%
- **LSTM Accuracy**: ~82%
- **GRU Accuracy**: ~82%

LSTM and GRU outperform vanilla RNN, with LSTM slightly leading in precision and recall.

---

## 📂 Repository Structure
├── twitter_training.csv          # Dataset
├── tokenizer.pkl                 # Saved tokenizer
├── label_encoder.pkl             # Saved label encoder
├── rnn_twitter_sentiment_tuned.keras
├── lstm_twitter_sentiment_tuned.keras
├── gru_twitter_sentiment_tuned.keras
├── app.py                        # FastAPI REST service
├── streamlit_app.py              # Streamlit UI
└── README.md                     # Project documentation
