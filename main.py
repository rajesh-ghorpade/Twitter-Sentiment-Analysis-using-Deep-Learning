#Import required libraries
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Loading tensorflow model for prediction
model = load_model('model.h5')

with open('tokenizer.pickle','rb') as file:
    tokenizer = pickle.load(file)

st.title("Twitter Tweets Sentiment Analysis")

tweet = st.text_area("Enter your tweet here:")

if st.button("Predict Sentiment") and tweet.strip():
    sequences = tokenizer.texts_to_sequences([tweet])
    sequences = pad_sequences(sequences, maxlen=99, padding='post')

    prediction = model.predict(sequences)
    predicted_class = np.argmax(prediction, axis=1)[0]

    sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

    st.write(f"Predicted Sentiment: {sentiment_mapping[predicted_class]}")
