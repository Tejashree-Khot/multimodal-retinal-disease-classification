# Minimalistic inference
import argparse
import sys
from pathlib import Path

import torch
from termcolor import colored

from dataloader.data_preprocessing import load_image
from transformers import BertTokenizer
from src.models.multimodel import MultiModalModel
from models.efficient_net import get_efficientnet_model


classes_dict = [
    "Clinically_Significant_Macular_Edema",
    "No_DR",
    "normal",
    "Mild_Moderate_NPDR",
    "Severe_PDR",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def predict(model, image_path: Path, text: str, multimodal: bool):
    model.eval()
    with torch.no_grad():
        # apply same transforms and tokenization as training
        image = load_image(image_path)
        if image.dim == 3:
            image = image.unsqueeze(0)
        image = image.to(device)
        if multimodal:
            encoding = tokenizer(
                text, padding="max_length", truncation=True, max_length=128, return_tensors="pt"
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            outputs = model(image, input_ids, attention_mask)

        else:
            outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = classes_dict[predicted.item()]
        return predicted_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for retinal disease classification")
    parser.add_argument("--image_path", type=str, help="Path to the image file.")
    parser.add_argument("--text", type=str, default="", help="Associated text for the image.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model file."
    )
    parser.add_argument(
        "--multimodal", action="store_true", help="Enable multimodal (image+text) inference"
    )
    args = parser.parse_args()
    image_path = Path(args.image_path)
    text = args.text
    model_path = Path(args.model_path)
    multimodal = args.multimodal
    # Load appropriate model
    if multimodal:
        model = MultiModalModel(num_classes=len(classes_dict), model_name="efficientnet-b0")
    else:
        model = get_efficientnet_model(num_classes=len(classes_dict), model_name="efficientnet-b0")[
            0
        ]
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    predicted_class = predict(model, image_path, text, multimodal)
    print(colored(f"Predicted class: {predicted_class}", "green"))
    sys.exit(0)
