# image_classifier.py
# Fast image classification for "safe water" vs "unsafe water"
# Uses MobileNetV2 backbone (lightweight, faster inference than CLIP/EfficientNet)

import os
from typing import Dict, Any
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import mobilenet_v2

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MobileNetV2 once (pretrained on ImageNet)
_model = mobilenet_v2(weights="IMAGENET1K_V1").to(DEVICE)
_model.eval()

# Define preprocessing (resize + normalize)
_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])

# Custom labels for zero-shot style mapping
CUSTOM_LABELS = {
    "safe water": ["lake", "river", "waterfall", "clear water"],
    "unsafe water": ["sewage", "swamp", "polluted water", "flood"]
}


def _ensure_image(path: str) -> Image.Image:
    """Open an image safely and convert to RGB."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    try:
        return Image.open(path).convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError(f"Invalid image format: {path}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to open image: {e}") from e


def classify_image(image_path: str) -> Dict[str, Any]:
    """
    Classify image as 'safe water' or 'unsafe water'.
    Returns dict with predicted_label, confidence, and raw top-5 ImageNet classes.
    """
    try:
        image = _ensure_image(image_path)
        inputs = _preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = _model(inputs)
            probs = torch.softmax(outputs, dim=1)[0]

        # Get top-5 ImageNet predictions
        top5 = torch.topk(probs, 5)
        top_indices = top5.indices.cpu().numpy()
        top_scores = top5.values.cpu().numpy()

        # Map predictions to safe/unsafe categories
        # (simple heuristic: if any unsafe keyword appears in top classes → unsafe)
        # You can refine this mapping with a custom dataset later.
        predicted_label = "safe water"
        confidence = float(top_scores[0])

        # Convert indices to labels (ImageNet class names)
        # Torchvision provides a mapping file
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        class_names = weights.meta["categories"]

        top_classes = [
            {"label": class_names[idx], "score": float(score)}
            for idx, score in zip(top_indices, top_scores)
        ]

        # Check unsafe keywords
        for unsafe_kw in CUSTOM_LABELS["unsafe water"]:
            if any(unsafe_kw in c["label"].lower() for c in top_classes):
                predicted_label = "unsafe water"
                break

        return {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "top_classes": top_classes,
            "device": str(DEVICE)
        }

    except Exception as e:
        return {"error": f"Classification failed: {e}"}


if __name__ == "__main__":
    # Example usage
    test_path = "example_water.jpg"
    result = classify_image(test_path)
    print(result)
