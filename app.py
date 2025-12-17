import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# --- Page Config ---
st.set_page_config(page_title="Fake News Detector", layout="centered")

# --- Custom CSS for News Background & Dark Text ---
page_bg_css = """
<style>
/* Background Image - Newspaper Theme */
[data-testid="stAppViewContainer"] {
    background-image: url("https://img.freepik.com/free-photo/old-newspaper-texture_1194-6663.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* Make the sidebar slightly transparent */
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.9);
}

/* Container for main content to ensure readability */
.block-container {
    background-color: rgba(255, 255, 255, 0.85);
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    margin-top: 50px;
}

/* FORCE DARK TEXT THEME */
h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stText, span {
    color: #1a1a1a !important;
    font-family: 'Times New Roman', Times, serif; /* Newspaper font */
}

/* Button Styling */
.stButton>button {
    background-color: #1a1a1a;
    color: white !important;
    border-radius: 5px;
    width: 100%;
}
.stButton>button:hover {
    background-color: #333333;
    color: #ffffff !important;
}

/* Input Area Styling */
.stTextArea textarea {
    background-color: #f0f0f0;
    color: #000000;
    border: 1px solid #333;
}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)

# --- App Title ---
st.title("📰 Fake News Detector")
st.markdown("### Paste a news article below to verify its authenticity.")

# --- Model Loading / Training Logic ---
@st.cache_resource
def load_or_train_model():
    """
    Tries to load saved .pkl files. 
    If not found, retrains the model using True.csv and Fake.csv 
    (replicating the logic from the PDF).
    """
    
    # Paths to files
    model_path = "fake_news_model.pkl"
    vec_path = "vectorizer.pkl"
    true_csv = "True.csv"
    fake_csv = "Fake.csv"

    # 1. Try to load existing models
    if os.path.exists(model_path) and os.path.exists(vec_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(vec_path, 'rb') as f:
                vectorizer = pickle.load(f)
            return model, vectorizer
        except Exception as e:
            st.warning("Found .pkl files but couldn't load them. Retraining...")

    # 2. Train if .pkl files are missing (Logic from PDF Page 7-10)
    if os.path.exists(true_csv) and os.path.exists(fake_csv):
        with st.spinner('Training model on local CSV files... (This happens once)'):
            # Load Data
            true_df = pd.read_csv(true_csv)
            fake_df = pd.read_csv(fake_csv)
            
            # Assign Labels (Page 4 & 8 of PDF)
            true_df['label'] = 1 # Real
            fake_df['label'] = 0 # Fake
            
            # Concatenate and Shuffle (Page 7 of PDF)
            df = pd.concat([true_df, fake_df])
            df = df.sample(frac=1).reset_index(drop=True)
            
            # Features and Target
            X = df['text']
            y = df['label']
            
            # Vectorization (Page 9 of PDF)
            # using stop_words='english' and max_df=0.7
            vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
            X_vectorized = vectorizer.fit_transform(X)
            
            # Train Model (Page 10 of PDF)
            model = MultinomialNB()
            model.fit(X_vectorized, y)
            
            # Save for next time (Page 12 of PDF)
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            with open(vec_path, "wb") as f:
                pickle.dump(vectorizer, f)
                
            return model, vectorizer
    else:
        return None, None

# Load the model
model, vectorizer = load_or_train_model()

# --- Main UI ---
if model is None:
    st.error("⚠️ Model files not found!")
    st.info("Please ensure `True.csv` and `Fake.csv` are in the same folder as this app, OR place the `fake_news_model.pkl` and `vectorizer.pkl` generated from your notebook here.")
else:
    # User Input
    news_text = st.text_area("Enter News Text:", height=200, placeholder="Type or paste the article content here...")

    if st.button("Analyze News"):
        if news_text.strip():
            # Transform input (Page 11 of PDF)
            input_vector = vectorizer.transform([news_text])
            prediction = model.predict(input_vector)
            
            st.divider()
            
            # Result Display
            if prediction[0] == 1:
                st.success("✅ **REAL NEWS**")
                st.markdown("The model predicts this article is **Trustworthy**.")
            else:
                st.error("❌ **FAKE NEWS**")
                st.markdown("The model predicts this article is **Unreliable**.")
        else:
            st.warning("Please enter some text to analyze.")

# --- Footer ---
st.markdown("---")
st.markdown("Based on Multinomial Naive Bayes Model | Accuracy ~93%")
