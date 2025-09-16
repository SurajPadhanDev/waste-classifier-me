import cv2
import numpy as np

def resize_with_padding(img, target_size=(224, 224)):
    """
    Resize image while maintaining aspect ratio and add padding.
    
    Args:
        img: Input image as numpy array
        target_size: Target size tuple (height, width)
        
    Returns:
        numpy array: Resized and padded image
    """
    h, w = img.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    
    # Resize image
    img_resized = cv2.resize(img, (nw, nh))
    
    # Calculate padding
    top = (target_size[0] - nh) // 2
    bottom = target_size[0] - nh - top
    left = (target_size[1] - nw) // 2
    right = target_size[1] - nw - left
    
    # Add padding
    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    
    return img_padded

def preprocess_image(image):
    """
    Preprocess image for model prediction.
    
    Args:
        image: Input image as numpy array or PIL Image
        
    Returns:
        numpy array: Preprocessed image ready for model input
    """
    # Convert PIL to numpy if needed
    if hasattr(image, 'mode'):
        image = np.array(image)
    
    # Ensure RGB format
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3:  # Already RGB
            pass
    else:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize with padding
    processed_image = resize_with_padding(image)
    
    return processed_image
