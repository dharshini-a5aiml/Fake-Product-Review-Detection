import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
data = pd.read_csv("dataset.csv")

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(data["review"])

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.title("Fake Product Review Detection")

# Input box
review = st.text_area("Enter Product Review")

# Predict button
if st.button("Predict"):

    review_vector = vectorizer.transform([review])

    prediction = model.predict(review_vector)

    if prediction[0] == 1:
        st.error("Fake Review")
    else:
        st.success("Genuine Review")