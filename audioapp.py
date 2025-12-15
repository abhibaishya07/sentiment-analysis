import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import pickle
import os
import sys

# --- 1. DLL Fix for Windows (Crucial for your RTX 5060 Ti) ---
if sys.platform == 'win32':
    # Attempt to locate the Torch library path within the virtual environment
    # This prevents the "WinError 1114" DLL initialization error
    venv_path = os.environ.get('VIRTUAL_ENV', '')
    torch_lib_path = os.path.join(venv_path, 'Lib', 'site-packages', 'torch', 'lib')
    
    if os.path.isdir(torch_lib_path) and hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(torch_lib_path)

# --- 2. Configuration (MUST Match Training) ---
SAMPLE_RATE = 22050
DURATION = 3
N_MELS = 64
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
HOP_LENGTH = 512

# --- 3. Define Model Architecture (MUST Match Training) ---
class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16)
        )
        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32)
        )
        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64)
        )
        # Block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128)
        )
        
        # Classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# --- 4. Helper Functions ---
@st.cache_resource
def load_resources():
    # Load Label Encoder
    if not os.path.exists('label_encoder.pkl'):
        st.error("❌ 'label_encoder.pkl' not found! Run the training script first.")
        st.stop()
        
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    
    # Load Model
    if not os.path.exists('audio_sentiment_model.pth'):
        st.error("❌ 'audio_sentiment_model.pth' not found! Run the training script first.")
        st.stop()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(le.classes_)
    
    model = AudioCNN(num_classes)
    model.load_state_dict(torch.load('audio_sentiment_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    return model, le, device

def preprocess_audio(file_path):
    try:
        # Load audio
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Pad or Truncate
        if len(signal) > NUM_SAMPLES:
            signal = signal[:NUM_SAMPLES]
        else:
            padding = NUM_SAMPLES - len(signal)
            signal = np.pad(signal, (0, padding), mode='constant')
            
        # Extract Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=signal, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize (0-1)
        norm_spec = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
        
        # Convert to Tensor (1, 64, 129) -> Add Batch Dim -> (1, 1, 64, 129)
        tensor = torch.tensor(norm_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# --- 5. Main App UI ---
st.title("🎙️ Audio Sentiment Analysis")
st.write("Upload an audio file (.wav) to detect the emotion.")

# Load Model
model, le, device = load_resources()

# File Uploader
uploaded_file = st.file_uploader("Choose a file...", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save temp file
    temp_filename = "temp_audio.wav"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    # Play Audio
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("🔍 Analyze Audio"):
        with st.spinner("Listening and Analyzing..."):
            # Preprocess
            input_tensor = preprocess_audio(temp_filename)
            
            if input_tensor is not None:
                # Move to GPU if available
                input_tensor = input_tensor.to(device)
                
                # Predict
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                
                # Decode Label
                predicted_label = le.inverse_transform([predicted_idx.item()])[0]
                confidence_score = confidence.item() * 100
                
                # Display Result
                st.divider()
                st.success(f"## 🎭 Prediction: **{predicted_label}**")
                st.info(f"Confidence: **{confidence_score:.2f}%**")
                
                # Cleanup
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)