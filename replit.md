# Smart Waste Classifier

## Overview

Smart Waste Classifier is an AI-powered waste classification system that uses computer vision to automatically categorize waste materials into three categories: Organic, Hazardous, and Recyclable. The application leverages a fine-tuned MobileNetV2 deep learning model to provide real-time waste classification through both image upload and live camera feed functionality. Built with Streamlit for an intuitive web interface, the system is designed to assist in proper waste management and environmental sustainability efforts.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Web Application**: Single-page application with responsive design and custom CSS animations
- **Real-time Processing**: Supports both static image upload and live camera feed classification
- **Interactive UI**: Wide layout with sidebar navigation and animated visual elements
- **Prediction Smoothing**: Uses buffered predictions to reduce classification noise in video streams

### Backend Architecture
- **Deep Learning Model**: Pre-trained MobileNetV2 architecture fine-tuned for 3-class waste classification
- **Image Processing Pipeline**: Custom preprocessing with aspect ratio preservation and padding
- **Prediction Buffer**: Deque-based smoothing system for stable real-time predictions
- **Modular Design**: Separated utility functions for model operations and image processing

### Data Processing
- **Image Preprocessing**: Resize with padding to maintain aspect ratio, RGB conversion, and MobileNetV2-specific preprocessing
- **Real-time Video Processing**: Frame-by-frame analysis with OpenCV for camera input
- **Prediction Confidence**: Threshold-based classification with confidence scoring
- **Class Mapping**: Three-category system (O=Hazardous, R=Organic, H=Recyclable)

### Model Architecture
- **Base Model**: MobileNetV2 optimized for mobile and edge deployment
- **Input Size**: 224x224 RGB images with preprocessing normalization
- **Output Classes**: 3-class classification (Organic, Hazardous, Recyclable)
- **Prediction Smoothing**: Rolling average over 10 recent predictions for stability

## External Dependencies

### Machine Learning Framework
- **TensorFlow/Keras**: Core deep learning framework for model loading and inference
- **MobileNetV2**: Pre-trained computer vision model for transfer learning

### Computer Vision Libraries
- **OpenCV**: Real-time computer vision processing and camera input handling
- **PIL (Pillow)**: Image manipulation and format conversion
- **NumPy**: Numerical computing for array operations and image processing

### Web Framework
- **Streamlit**: Web application framework for ML model deployment and user interface

### System Libraries
- **Collections**: Deque implementation for prediction buffering
- **Threading**: Concurrent processing support for real-time operations
- **OS**: File system operations and path management