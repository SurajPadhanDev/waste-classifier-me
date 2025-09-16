import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from collections import deque

# Class definitions
CLASSES = ["O", "R", "H"]
CLASS_NAMES = {"O": "Hazardous Waste", "R": "Organic Waste", "H": "Inorganic Waste"}
CONFIDENCE_THRESHOLD = 0.7

def load_classification_model(model_path):
    """Load the trained MobileNetV2 model."""
    try:
        model = load_model(model_path)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e

def predict_image_class(model, image_array, pred_buffer):
    """
    Predict the class of an image using the loaded model.
    
    Args:
        model: Loaded Keras model
        image_array: RGB image as numpy array
        pred_buffer: Deque for smoothing predictions
        
    Returns:
        tuple: (class_name, confidence)
    """
    from utils.image_utils import resize_with_padding
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    
    # Preprocess image
    img_resized = resize_with_padding(image_array)
    img_array = np.expand_dims(img_resized.astype(np.float32), axis=0)
    img_array = preprocess_input(img_array)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    pred_buffer.append(predictions)
    
    # Average predictions for smoothing
    avg_pred = np.mean(pred_buffer, axis=0)
    
    # Get predicted class
    predicted_idx = np.argmax(avg_pred)
    predicted_class = CLASSES[predicted_idx]
    confidence = avg_pred[predicted_idx]
    
    # Map to human-readable class name
    class_name = CLASS_NAMES.get(predicted_class, "Unknown")
    
    return class_name, confidence
