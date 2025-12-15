import streamlit as st
import torch
import torch.nn as nn
import os
import sys
# if sys.platform == 'win32':
#     torch_lib_path = os.path.join(os.environ['VIRTUAL_ENV'], 'Lib', 'site-packages', 'torch', 'lib')
#     if os.path.isdir(torch_lib_path):
#         os.add_dll_directory(torch_lib_path)
import pickle
import re
import string
import numpy as np
from langdetect import detect, LangDetectException  # Import language detection

# --- 1. Define the Preprocessing Function ---
def wordopt(text):
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# --- 2. Define the Model Structure ---
class SimpleANN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# --- 3. Load Resources ---
@st.cache_resource
def load_resources():
    # Load Vectorizer
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Load Model
    input_dim = len(vectorizer.get_feature_names_out())
    model = SimpleANN(input_dim)
    
    # Load Weights
    model.load_state_dict(torch.load('sentiment_model.pth', map_location=torch.device('cpu')))
    model.eval()
    
    return vectorizer, model

# --- 4. The App Interface ---
st.title("💬 Comment Sentiment Analyzer")
st.write("Enter a comment below to check if it's **Positive** or **Negative** and detect its language.")

try:
    vectorizer, model = load_resources()
    
    # User Input
    user_input = st.text_area("Enter Comment:", height=150)

    if st.button("Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter some text first.")
        else:
            # --- A. Detect Language ---
            try:
                language_code = detect(user_input)
                # Map common codes to full names (optional)
                lang_map = {'en': 'English', 'hi': 'Hindi', 'es': 'Spanish', 'fr': 'French'}
                language_name = lang_map.get(language_code, language_code)
            except LangDetectException:
                language_name = "Unknown"

            # --- B. Preprocess & Predict ---
            cleaned_text = wordopt(user_input)
            vector_input = vectorizer.transform([cleaned_text]).toarray()
            tensor_input = torch.tensor(vector_input, dtype=torch.float32)
            
            with torch.no_grad():
                output = model(tensor_input)
                prediction_score = output.item()
                # Assuming 1 = Positive, 0 = Negative
                prediction_class = 1 if prediction_score > 0.5 else 0

            # --- C. Display Result ---
            st.divider()
            
            # Show Language
            st.info(f"🌐 **Language Detected:** {language_name.upper()}")

            # Show Sentiment
            if prediction_class == 1:
                st.success(f"😊 **POSITIVE COMMENT** (Confidence: {prediction_score:.2%})")
            else:
                st.error(f"😡 **NEGATIVE COMMENT** (Confidence: {1-prediction_score:.2%})")

except FileNotFoundError:
    st.error("Error: Could not find 'sentiment_model.pth' or 'tfidf_vectorizer.pkl'. Please make sure they are in the folder.")