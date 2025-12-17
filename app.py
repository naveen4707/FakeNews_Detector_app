import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

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
    background-color: rgba(255, 255, 255, 0.9);
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    margin-top: 30px;
}

/* FORCE DARK TEXT THEME */
h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stText, span, div {
    color: #1a1a1a !important;
    font-family: 'Times New Roman', Times, serif;
}

/* Button Styling */
.stButton>button {
    background-color: #1a1a1a;
    color: white !important;
    border-radius: 5px;
    width: 100%;
    font-weight: bold;
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
st.title("📰 Fake News Probability Detector")
st.markdown("### Paste a news article below to analyze its probability.")

# --- Model Loading / Training Logic ---
@st.cache_resource
def load_or_train_model():
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
        except Exception:
            pass

    # 2. Train if .pkl files are missing
    if os.path.exists(true_csv) and os.path.exists(fake_csv):
        with st.spinner('Training model on local CSV files...'):
            true_df = pd.read_csv(true_csv)
            fake_df = pd.read_csv(fake_csv)
            
            true_df['label'] = 1 # Real
            fake_df['label'] = 0 # Fake
            
            df = pd.concat([true_df, fake_df])
            df = df.sample(frac=1).reset_index(drop=True)
            
            X = df['text']
            y = df['label']
            
            vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
            X_vectorized = vectorizer.fit_transform(X)
            
            model = MultinomialNB()
            model.fit(X_vectorized, y)
            
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            with open(vec_path, "wb") as f:
                pickle.dump(vectorizer, f)
                
            return model, vectorizer
    else:
        return None, None

model, vectorizer = load_or_train_model()

# --- Main UI ---
if model is None:
    st.error("⚠️ Model files not found! Please upload True.csv and Fake.csv.")
else:
    # User Input
    news_text = st.text_area("Enter News Text:", height=150, placeholder="Paste article content here...")

    if st.button("Analyze Probability"):
        if news_text.strip():
            # 1. Transform input
            input_vector = vectorizer.transform([news_text])
            
            # 2. Get Probabilities (Index 0 = Fake, Index 1 = Real)
            probabilities = model.predict_proba(input_vector)[0]
            prob_fake = probabilities[0]
            prob_real = probabilities[1]
            
            # 3. Determine Winner
            prediction = np.argmax(probabilities)
            
            st.divider()
            
            # 4. Display Text Result
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if prediction == 1:
                    st.success(f"✅ **Prediction: REAL NEWS**")
                    st.write(f"Confidence: **{prob_real*100:.2f}%**")
                    st.markdown("The model detected patterns consistent with verified news sources.")
                else:
                    st.error(f"❌ **Prediction: FAKE NEWS**")
                    st.write(f"Confidence: **{prob_fake*100:.2f}%**")
                    st.markdown("The model detected patterns consistent with unverified or fake sources.")

            # 5. Graphing Logic (Donut Chart)
            with col2:
                st.markdown("**Probability Breakdown**")
                
                # Setup Plot
                labels = ['Fake', 'Real']
                sizes = [prob_fake, prob_real]
                colors = ['#ff4b4b', '#4caf50'] # Red for Fake, Green for Real
                explode = (0.05, 0.05) 
                
                fig, ax = plt.subplots(figsize=(3, 3))
                # Transparent background for the figure
                fig.patch.set_alpha(0) 
                
                # Create Pie/Donut
                wedges, texts, autotexts = ax.pie(
                    sizes, 
                    colors=colors, 
                    labels=labels, 
                    autopct='%1.1f%%', 
                    startangle=90, 
                    pctdistance=0.85, 
                    explode=explode,
                    textprops={'color':"black", 'fontsize': 10, 'weight':'bold'}
                )
                
                # Draw a white circle at center to make it a donut
                centre_circle = plt.Circle((0,0),0.70,fc='white')
                fig.gca().add_artist(centre_circle)
                
                ax.axis('equal')  
                st.pyplot(fig, use_container_width=False)
                
        else:
            st.warning("Please enter some text to analyze.")

# --- Footer ---
st.markdown("---")
st.markdown("Probability calculated using Multinomial Naive Bayes `predict_proba`.")
