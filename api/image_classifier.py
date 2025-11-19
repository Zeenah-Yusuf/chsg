# api/image_classifier.py
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image, UnidentifiedImageError

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load once
processor = AutoImageProcessor.from_pretrained("google/efficientnet-b0")
model = AutoModelForImageClassification.from_pretrained("google/efficientnet-b0").to(device)

def classify_image(image_path: str) -> dict:
    """
    Classify an image using EfficientNet.
    Returns dict with label and confidence.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        return {"error": f"Image file not found: {image_path}"}
    except UnidentifiedImageError:
        return {"error": f"Invalid image format: {image_path}"}

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    pred_id = logits.argmax(-1).item()
    confidence = torch.softmax(logits, dim=-1)[0][pred_id].item()

    return {"label": model.config.id2label[pred_id], "confidence": confidence}
