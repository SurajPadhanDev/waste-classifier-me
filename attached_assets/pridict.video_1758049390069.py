import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from collections import deque
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = "/Users/surajpadhan/Desktop/WasteWiseVision/attached_assets/best_mobilenetv2_model_1758049390069.keras"

try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    exit()

# Classes
CLASSES = ["O", "R", "H"]
CLASS_NAMES = {"R": "Organic", "O": "Hazardous", "H": "Recycle"}
CONFIDENCE_THRESHOLD = 0.7
pred_buffer = deque(maxlen=10)


# -----------------------------
# Utility functions
# -----------------------------
def resize_with_padding(img, target_size=(224, 224)):
    """Resize while keeping aspect ratio and pad with black borders."""
    h, w = img.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh))
    top = (target_size[0] - nh) // 2
    bottom = target_size[0] - nh - top
    left = (target_size[1] - nw) // 2
    right = target_size[1] - nw - left
    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    return img_padded


def predict_frame(frame):
    """Run prediction on a frame and return label string."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = resize_with_padding(img_rgb)
    img_array = np.expand_dims(img_resized.astype(np.float32), axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0)[0]
    pred_buffer.append(predictions)
    avg_pred = np.mean(pred_buffer, axis=0)

    predicted_idx = np.argmax(avg_pred)
    predicted_class = CLASSES[predicted_idx]
    confidence = avg_pred[predicted_idx]

    if confidence < CONFIDENCE_THRESHOLD:
        return f"❓ Uncertain ({confidence*100:.1f}%)"
    else:
        return f"{CLASS_NAMES[predicted_class]} ({confidence*100:.1f}%)"


# -----------------------------
# Tkinter App Class
# -----------------------------
class BOLT_INOVATOR_WasteClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("♻️ BOLT INOVATOR Waste Classifier")
        self.root.configure(bg="black")

        # ✅ Fullscreen mode
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda e: self.exit_fullscreen())

        # Video panel
        self.video_label = tk.Label(root, bg="black")
        self.video_label.pack(pady=10, expand=True)

        # Prediction text
        self.prediction_label = tk.Label(
            root,
            text="Click ▶ Start to begin...",
            font=("Consolas", 22, "bold"),
            fg="cyan",
            bg="black",
        )
        self.prediction_label.pack(pady=20)

        # Button Frame
        btn_frame = tk.Frame(root, bg="black")
        btn_frame.pack(pady=20)

        # Buttons
        self.start_button = tk.Button(
            btn_frame,
            text="▶ Start",
            font=("Consolas", 16, "bold"),
            fg="white",
            bg="green",
            width=14,
            command=self.start_camera,
        )
        self.start_button.grid(row=0, column=0, padx=15)

        self.stop_button = tk.Button(
            btn_frame,
            text="⏹ Stop",
            font=("Consolas", 16, "bold"),
            fg="white",
            bg="red",
            width=14,
            command=self.stop_camera,
            state="disabled",
        )
        self.stop_button.grid(row=0, column=1, padx=15)

        self.exit_button = tk.Button(
            btn_frame,
            text="❌ Exit",
            font=("Consolas", 16, "bold"),
            fg="white",
            bg="gray",
            width=14,
            command=self.exit_app,
        )
        self.exit_button.grid(row=0, column=2, padx=15)

        # Camera
        self.cap = None
        self.is_running = False

    def start_camera(self):
        """Start webcam stream."""
        self.cap = cv2.VideoCapture(0)  # PC webcam
        if not self.cap.isOpened():
            messagebox.showerror("Error", "❌ Could not access webcam!")
            return

        pred_buffer.clear()
        self.is_running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.update_frame()

    def update_frame(self):
        """Capture frame, run prediction, and update GUI."""
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Prediction
                label = predict_frame(frame)
                self.prediction_label.config(text=label)

                # Convert for Tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)

                # ✅ Resize video to fit fullscreen nicely
                screen_w = self.root.winfo_screenwidth()
                screen_h = self.root.winfo_screenheight()
                img = img.resize((screen_w // 2, screen_h // 2))  

                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        if self.is_running:
            self.root.after(10, self.update_frame)

    def stop_camera(self):
        """Stop webcam, capture last frame, and show its prediction."""
        self.is_running = False
        last_label = "Camera stopped."

        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                last_label = predict_frame(frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                screen_w = self.root.winfo_screenwidth()
                screen_h = self.root.winfo_screenheight()
                img = img.resize((screen_w // 2, screen_h // 2))  
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            self.cap.release()
            self.cap = None

        self.prediction_label.config(text=f"📷 Final: {last_label}")
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

    def exit_app(self):
        """Exit cleanly."""
        self.stop_camera()
        self.root.destroy()

    def exit_fullscreen(self):
        """Disable fullscreen when pressing Esc."""
        self.root.attributes("-fullscreen", False)


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = BOLT_INOVATOR_WasteClassifier(root)
    root.mainloop()

    if app.cap:
        app.cap.release()
    cv2.destroyAllWindows()

