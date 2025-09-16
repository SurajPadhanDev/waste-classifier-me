import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from collections import deque
import threading
import time
import os

# Import utility functions
from utils.model_utils import load_classification_model, predict_image_class
from utils.image_utils import resize_with_padding, preprocess_image

# Page configuration
st.set_page_config(
    page_title="♻️ Smart Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for stunning visuals and animations
st.markdown("""
<style>
    /* Global background with moving gradient */
    .stApp {
        background: linear-gradient(-45deg, #0f0f23, #1a1a2e, #16213e, #0f3460);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Particle effect overlay */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, #00ff41, transparent),
            radial-gradient(2px 2px at 40px 70px, rgba(255,255,255,0.1), transparent),
            radial-gradient(1px 1px at 90px 40px, #00ff41, transparent),
            radial-gradient(1px 1px at 130px 80px, rgba(255,255,255,0.1), transparent),
            radial-gradient(2px 2px at 160px 30px, #00ff41, transparent);
        background-repeat: repeat;
        background-size: 200px 100px;
        animation: sparkle 20s linear infinite;
        pointer-events: none;
        z-index: 1;
    }
    
    @keyframes sparkle {
        from { transform: translateY(0px); }
        to { transform: translateY(-100px); }
    }
    
    /* Main header with enhanced effects */
    .main-header {
        text-align: center;
        padding: 30px 0;
        background: linear-gradient(45deg, #00ff41, #41ff00, #00ff88, #88ff00);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: headerGlow 3s ease-in-out infinite alternate, colorShift 8s ease infinite;
        font-size: 3.5rem;
        font-weight: 900;
        text-shadow: 0 0 30px rgba(0,255,65,0.5);
        position: relative;
        z-index: 10;
    }
    
    @keyframes headerGlow {
        from { 
            filter: brightness(1) drop-shadow(0 0 10px rgba(0,255,65,0.3));
            transform: scale(1);
        }
        to { 
            filter: brightness(1.4) drop-shadow(0 0 20px rgba(0,255,65,0.8));
            transform: scale(1.02);
        }
    }
    
    @keyframes colorShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Enhanced team name with 3D effect */
    .team-name {
        text-align: center;
        font-size: 28px;
        font-weight: 900;
        background: linear-gradient(45deg, #ff6b35, #f7931e, #ffd700, #ff6b35);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 25px 0;
        animation: teamPulse 2.5s infinite, teamShine 6s ease infinite;
        transform-style: preserve-3d;
        position: relative;
        z-index: 10;
    }
    
    @keyframes teamPulse {
        0%, 100% { 
            transform: scale(1) rotateY(0deg);
            filter: drop-shadow(0 0 10px rgba(255,107,53,0.4));
        }
        50% { 
            transform: scale(1.08) rotateY(2deg);
            filter: drop-shadow(0 0 25px rgba(255,107,53,0.8));
        }
    }
    
    @keyframes teamShine {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Enhanced prediction cards with glassmorphism */
    .prediction-card {
        background: linear-gradient(135deg, 
            rgba(30, 60, 114, 0.8) 0%, 
            rgba(42, 82, 152, 0.8) 50%,
            rgba(58, 123, 213, 0.8) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.2);
        padding: 25px;
        border-radius: 20px;
        margin: 15px 0;
        box-shadow: 
            0 8px 32px rgba(0,0,0,0.3),
            inset 0 1px 0 rgba(255,255,255,0.1),
            0 0 20px rgba(0,255,65,0.1);
        position: relative;
        overflow: hidden;
        animation: cardFloat 6s ease-in-out infinite;
        z-index: 5;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(0,255,65,0.1), transparent);
        animation: cardShimmer 3s linear infinite;
        pointer-events: none;
    }
    
    @keyframes cardFloat {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-3px) rotate(0.5deg); }
    }
    
    @keyframes cardShimmer {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Enhanced confidence indicators */
    .confidence-high {
        color: #00ff88;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(0,255,136,0.5);
        animation: confidenceGlow 2s ease-in-out infinite alternate;
    }
    
    .confidence-medium {
        color: #ffaa00;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(255,170,0,0.5);
        animation: confidenceGlow 2s ease-in-out infinite alternate;
    }
    
    .confidence-low {
        color: #ff4444;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(255,68,68,0.5);
        animation: confidenceGlow 2s ease-in-out infinite alternate;
    }
    
    @keyframes confidenceGlow {
        from { filter: brightness(1); }
        to { filter: brightness(1.3); }
    }
    
    /* Enhanced buttons with hover effects */
    .stButton > button {
        background: linear-gradient(45deg, #00ff41, #41ff00) !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 12px 24px !important;
        font-weight: bold !important;
        color: #000 !important;
        box-shadow: 0 4px 15px rgba(0,255,65,0.3) !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.05) !important;
        box-shadow: 0 8px 25px rgba(0,255,65,0.5) !important;
    }
    
    /* Sidebar enhancements */
    .css-1d391kg {
        background: linear-gradient(180deg, 
            rgba(15, 15, 35, 0.95) 0%, 
            rgba(26, 26, 46, 0.95) 100%) !important;
        backdrop-filter: blur(10px) !important;
        border-right: 1px solid rgba(0,255,65,0.2) !important;
    }
    
    /* Content area transparency */
    .block-container {
        background: rgba(255,255,255,0.02) !important;
        backdrop-filter: blur(5px) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        position: relative !important;
        z-index: 2 !important;
    }
    
    /* Progress bar enhancement */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00ff41, #41ff00) !important;
        border-radius: 10px !important;
        box-shadow: 0 0 10px rgba(0,255,65,0.5) !important;
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed rgba(0,255,65,0.3) !important;
        border-radius: 15px !important;
        background: rgba(255,255,255,0.05) !important;
        backdrop-filter: blur(10px) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'pred_buffer' not in st.session_state:
    st.session_state.pred_buffer = deque(maxlen=10)

# Main header
st.markdown('<h1 class="main-header">♻️ Smart Waste Classifier</h1>', unsafe_allow_html=True)
st.markdown('<div class="team-name">🚀 BOLTINNOVATOR 🚀</div>', unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model_cached():
    try:
        model_path = "best_mobilenetv2_model.keras"
        if os.path.exists(model_path):
            model = load_classification_model(model_path)
            return model
        else:
            st.error("❌ Model file 'best_mobilenetv2_model.keras' not found!")
            return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Sidebar with instructions
st.sidebar.markdown("## 📖 Usage Instructions")
st.sidebar.markdown("""
### Live Camera Classification:
1. Click **'Start Camera'** to begin live classification
2. Point your camera at waste items
3. The system will classify in real-time
4. Click **'Stop Camera'** to end the session

### Upload Image Classification:
1. Use the **file uploader** to select an image
2. Supported formats: JPG, JPEG, PNG
3. The system will analyze and classify the waste
4. View results with confidence scores

### Waste Categories:
- 🌱 **Organic**: Food waste, biodegradable items
- ☠️ **Hazardous**: Toxic materials, chemicals
- ♻️ **Recycle**: Plastic, glass, paper, metal
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Classification Tips")
st.sidebar.markdown("""
- Ensure good lighting
- Hold items clearly in view
- Avoid cluttered backgrounds
- Wait for confidence > 70% for reliable results
""")

# Load the model
if st.session_state.model is None:
    with st.spinner("🔄 Loading AI model..."):
        st.session_state.model = load_model_cached()

if st.session_state.model is None:
    st.stop()

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 📹 Live Camera Classification")
    
    # Camera controls
    camera_col1, camera_col2 = st.columns([1, 1])
    
    with camera_col1:
        if st.button("📹 Start Camera", type="primary", use_container_width=True):
            st.session_state.camera_active = True
            st.rerun()
    
    with camera_col2:
        if st.button("⏹️ Stop Camera", type="secondary", use_container_width=True):
            st.session_state.camera_active = False
            st.rerun()
    
    # Camera feed placeholder
    camera_placeholder = st.empty()
    camera_result_placeholder = st.empty()
    
    # Camera functionality
    if st.session_state.camera_active:
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Could not access webcam!")
                st.session_state.camera_active = False
            else:
                # Camera loop
                frame_placeholder = camera_placeholder.empty()
                
                for i in range(30):  # Run for 30 frames
                    if not st.session_state.camera_active:
                        break
                        
                    ret, frame = cap.read()
                    if ret:
                        # Display frame
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                        
                        # Predict
                        prediction, confidence = predict_image_class(
                            st.session_state.model, 
                            frame_rgb, 
                            st.session_state.pred_buffer
                        )
                        
                        # Display result
                        if confidence >= 0.7:
                            conf_class = "confidence-high"
                        elif confidence >= 0.5:
                            conf_class = "confidence-medium"
                        else:
                            conf_class = "confidence-low"
                            
                        camera_result_placeholder.markdown(f"""
                        <div class="prediction-card">
                            <h3>🎯 Live Prediction</h3>
                            <p><strong>Class:</strong> {prediction}</p>
                            <p><strong>Confidence:</strong> <span class="{conf_class}">{confidence*100:.1f}%</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        time.sleep(0.1)
                
                cap.release()
                
        except Exception as e:
            st.error(f"❌ Camera error: {str(e)}")
            st.session_state.camera_active = False
    
    else:
        camera_placeholder.markdown("📷 Camera is stopped. Click 'Start Camera' to begin live classification.")

with col2:
    st.markdown("## 📤 Upload Image Classification")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image of waste for classification"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert PIL to numpy array
        image_array = np.array(image)
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            # Convert RGBA to RGB
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        elif len(image_array.shape) == 3:
            # Already RGB
            pass
        else:
            # Convert grayscale to RGB
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        
        # Predict
        with st.spinner("🔍 Analyzing image..."):
            prediction, confidence = predict_image_class(
                st.session_state.model, 
                image_array, 
                deque(maxlen=1)  # Single prediction for uploaded images
            )
        
        # Display results
        if confidence >= 0.7:
            conf_class = "confidence-high"
            conf_emoji = "✅"
        elif confidence >= 0.5:
            conf_class = "confidence-medium"  
            conf_emoji = "⚠️"
        else:
            conf_class = "confidence-low"
            conf_emoji = "❓"
            
        st.markdown(f"""
        <div class="prediction-card">
            <h3>{conf_emoji} Classification Result</h3>
            <p><strong>Predicted Class:</strong> {prediction}</p>
            <p><strong>Confidence Score:</strong> <span class="{conf_class}">{confidence*100:.1f}%</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence progress bar
        st.progress(confidence, text=f"Confidence: {confidence*100:.1f}%")
        
        # Additional info based on class
        class_info = {
            "Organic": "🌱 Dispose in organic waste bin. This waste is biodegradable.",
            "Hazardous": "☠️ Handle with care! Dispose at designated hazardous waste facility.",
            "Recycle": "♻️ Great! This item can be recycled. Clean before disposing."
        }
        
        if prediction in class_info:
            st.info(class_info[prediction])

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p>🚀 Powered by BOLTINNOVATOR Team | AI Waste Classification System</p>
    <p>Built with ❤️ using Streamlit & TensorFlow</p>
</div>
""", unsafe_allow_html=True)
