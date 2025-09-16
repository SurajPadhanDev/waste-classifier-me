import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from collections import deque
import tkinter as tk
from PIL import Image, ImageTk
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# Load trained model
# -----------------------------
model_path = "/Users/surajpadhan/Desktop/WasteWiseVision/attached_assets/best_mobilenetv2_model_1758049390069.keras"
model = load_model(model_path)
print("✅ Model loaded successfully!")

# Classes
classes = ["O", "R", "H"]
class_names = {"R": "Organic", "O": "Harzdious", "H": "Recycle"}
confidence_threshold = 0.7
pred_buffer = deque(maxlen=10)

# -----------------------------
# Resize with padding
# -----------------------------
def resize_with_padding(img, target_size=(224,224)):
    h, w = img.shape[:2]
    scale = min(target_size[0]/h, target_size[1]/w)
    nh, nw = int(h*scale), int(w*scale)
    img_resized = cv2.resize(img, (nw, nh))
    top = (target_size[0]-nh)//2
    bottom = target_size[0]-nh-top
    left = (target_size[1]-nw)//2
    right = target_size[1]-nw-left
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=[0,0,0])
    return img_padded

# -----------------------------
# Predict frame
# -----------------------------
def predict_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = resize_with_padding(img_rgb)
    img_array = np.expand_dims(img_resized.astype(np.float32), axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0)[0]
    pred_buffer.append(predictions)
    avg_pred = np.mean(pred_buffer, axis=0)

    predicted_idx = np.argmax(avg_pred)
    predicted_class = classes[predicted_idx]
    confidence = avg_pred[predicted_idx]

    if confidence < confidence_threshold:
        label = f"sorry ({confidence*100:.1f}%)"
    else:
        label = f"{class_names[predicted_class]} ({confidence*100:.1f}%)"
    return label

# -----------------------------
# Tkinter Jarvis-like GUI
# -----------------------------
root = tk.Tk()
root.title("J.A.R.V.I.S Waste Classifier")
root.configure(bg="black")

video_label = tk.Label(root, bg="black")
video_label.pack()

prediction_label = tk.Label(root, text="Initializing...", font=("Consolas", 18, "bold"),
                            fg="cyan", bg="black")
prediction_label.pack(pady=10)

# Change this to your IP Webcam link
cap = cv2.VideoCapture("http://172.59.1.251:4747/video")

def update_frame():
    ret, frame = cap.read()
    if ret:
        label = predict_frame(frame)

        # Update GUI label
        prediction_label.config(text=label)

        # Convert for Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    root.after(10, update_frame)

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()

