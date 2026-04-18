"""Single-image inference for the dragon fruit disease classifier.

Loads a trained ViT checkpoint and predicts the disease class for a given
image, printing the predicted label and confidence score.

Usage:
    python predict.py --image path/to/image.jpg --config config.yaml
    python predict.py --image path/to/image.jpg --checkpoint results/best_vit.pth
"""

import argparse
import os

import torch
import yaml
from PIL import Image
from torchvision import transforms

from models.vit import get_vit_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def predict(image_path: str, checkpoint_path: str, class_names: list) -> tuple:
    """Run inference on a single image.

    Args:
        image_path (str): Path to the input image file.
        checkpoint_path (str): Path to the .pth model checkpoint.
        class_names (list): Ordered list of class names.

    Returns:
        tuple: (predicted_class, confidence_percent)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_vit_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probs, dim=0)

    return class_names[predicted_idx.item()], confidence.item() * 100


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict dragon fruit disease from an image.")
    parser.add_argument("--image",      required=True,          help="Path to input image")
    parser.add_argument("--config",     default="config.yaml",  help="Path to config.yaml")
    parser.add_argument("--checkpoint", default=None,           help="Path to .pth checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    class_names     = cfg["labels"]
    checkpoint_path = args.checkpoint or os.path.join(cfg["output"]["results_dir"], "best_vit.pth")

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    predicted_class, confidence = predict(args.image, checkpoint_path, class_names)

    print("\n" + "=" * 40)
    print("  Prediction Result")
    print("=" * 40)
    print(f"  Image      : {os.path.basename(args.image)}")
    print(f"  Class      : {predicted_class}")
    print(f"  Confidence : {confidence:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()
