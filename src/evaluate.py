"""Evaluate a trained model on a test dataset."""

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from termcolor import colored
from tqdm import tqdm
import argparse
from pathlib import Path
from dataloader.data_loader import get_data_loader
from models.efficient_net import get_efficientnet_model
from dataloader.data_preprocessing import tokenize_text
from multimodel import MultiModalModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, data_loader, multimodal=False):
    """Evaluate the model on the test dataset"""
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        if multimodal:
            for images, texts, attention_masks, labels in tqdm(data_loader, desc="Evaluating"):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                # tokenize on the fly
                encoding = tokenize_text(texts)
                texts = encoding["input_ids"].to(DEVICE)
                attention_masks = encoding["attention_mask"].to(DEVICE)
                outputs = model(images, texts, attention_masks)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        else:
            for images, labels in tqdm(data_loader, desc="Evaluating"):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1_score_val = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return accuracy, precision, recall, f1_score_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a test dataset")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model file."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the test dataset directory"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--image_size", type=int, default=224, help="Image size for resizing")
    parser.add_argument(
        "--multimodal", action="store_true", help="Enable multimodal(image+ text) evaluation"
    )
    args = parser.parse_args()
    print(colored(f"Loading model from {args.model_path}", "cyan"))
    if args.multimodal:
        model = MultiModalModel(num_classes=5, model_name="efficientnet-b0")
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        model.to(DEVICE)
        print(colored("Model loaded successfully.", "green"))
        test_loader = get_data_loader(
            dataset_path=Path(args.data_dir),
            size=(args.image_size, args.image_size),
            batch_size=args.batch_size,
            augment=False,
            tokenizer=None,
            use_weighted_sampler=False,
            multimodal=args.multimodal,
        )
        print(colored(f"Evaluating on {len(test_loader.dataset)} test samples...", "cyan"))
        accuracy, precision, recall, f1 = evaluate_model(model, test_loader, multimodal=True)
        print(colored(f"Accuracy: {accuracy:.4f}", "green"))
        print(colored(f"Precision: {precision:.4f}", "green"))
        print(colored(f"Recall: {recall:.4f}", "green"))
        print(colored(f"F1 Score: {f1:.4f}", "green"))
    else:
        model = get_efficientnet_model(num_classes=5, pretrained=False, multimodal=args.multimodal)
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        model.to(DEVICE)
        print(colored("Model loaded successfully.", "green"))
        test_loader = get_data_loader(
            dataset_path=Path(args.data_dir),
            size=(args.image_size, args.image_size),
            batch_size=args.batch_size,
            augment=False,
            tokenizer=None,
            use_weighted_sampler=False,
            multimodal=args.multimodal,
        )
        print(colored(f"Evaluating on {len(test_loader.dataset)} test samples...", "cyan"))
        accuracy, precision, recall, f1 = evaluate_model(model, test_loader)
        print(colored(f"Accuracy: {accuracy:.4f}", "green"))
        print(colored(f"Precision: {precision:.4f}", "green"))
        print(colored(f"Recall: {recall:.4f}", "green"))
        print(colored(f"F1 Score: {f1:.4f}", "green"))
