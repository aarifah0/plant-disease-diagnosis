"""Prediction pipeline: load model once, run inference, attach disease info."""
import argparse
import time

import torch
import torch.nn.functional as F

from src.model import load_model
from src.preprocess import load_image, preprocess_image
from src.utils import (
    DEFAULT_DISEASE_DB_PATH,
    DEFAULT_WEIGHTS_PATH,
    confidence_level,
    format_class_name,
    get_device,
    load_class_names,
    load_disease_db,
)


class Predictor:
    """Loads the trained model once and serves predictions for input images."""

    def __init__(self, weights_path=DEFAULT_WEIGHTS_PATH, disease_db_path=DEFAULT_DISEASE_DB_PATH, device=None):
        self.device = device or get_device()
        self.class_names = load_class_names()
        self.disease_db = load_disease_db(disease_db_path)
        self.model = load_model(weights_path, num_classes=len(self.class_names), device=self.device)

    def predict(self, image, top_k: int = 3) -> dict:
        """Run inference on a single PIL image. Returns top-k predictions with disease info."""
        start = time.perf_counter()
        tensor = preprocess_image(image).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        top_k = min(top_k, len(self.class_names))
        top_probs, top_indices = torch.topk(probs, top_k)

        predictions = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            class_name = self.class_names[idx]
            predictions.append({
                "class_name": class_name,
                "label": format_class_name(class_name),
                "confidence": prob,
                "confidence_level": confidence_level(prob),
                "info": self.disease_db.get(class_name, {}),
            })

        return {
            "top_prediction": predictions[0],
            "predictions": predictions,
            "inference_seconds": time.perf_counter() - start,
        }

    def predict_batch(self, images: list, top_k: int = 3) -> list:
        """Run prediction.predict() over a list of PIL images, preserving order."""
        return [self.predict(image, top_k=top_k) for image in images]


def main():
    parser = argparse.ArgumentParser(description="Run plant disease prediction on an image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS_PATH), help="Path to model weights.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to show.")
    args = parser.parse_args()

    predictor = Predictor(weights_path=args.weights)
    image = load_image(args.image)
    result = predictor.predict(image, top_k=args.top_k)

    top = result["top_prediction"]
    print(f"Prediction: {top['label']} ({top['confidence'] * 100:.2f}%)")
    if top["info"]:
        print(f"Description: {top['info'].get('description', 'N/A')}")
        print(f"Treatment: {top['info'].get('treatment', 'N/A')}")

    print(f"\nInference time: {result['inference_seconds'] * 1000:.0f} ms")
    print("Top predictions:")
    for pred in result["predictions"]:
        print(f"  {pred['label']}: {pred['confidence'] * 100:.2f}%")


if __name__ == "__main__":
    main()
