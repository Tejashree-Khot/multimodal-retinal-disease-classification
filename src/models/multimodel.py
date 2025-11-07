"""Multimodal model combining an EfficientNet image encoder and BERT text encoder.

This module defines a lightweight fusion model that accepts image tensors and
tokenized text inputs. The two branches (image and text) are projected to the
same embedding size and concatenated before the final classification head.

The model is intentionally simple and designed to be used in experiments where
both image and textual captions or metadata are available.
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
from transformers import BertModel


class MultiModalModel(nn.Module):
    """Multimodal classifier combining image and text encoders.

    Args:
        num_classes (int): Number of target classes for classification.
        model_name (str): EfficientNet variant to use for the image branch.

    Inputs to forward:
        images: torch.Tensor of shape [batch, 3, H, W]
        input_ids: torch.Tensor of token ids for BERT [batch, seq_len]
        attention_mask: torch.Tensor attention masks [batch, seq_len]

    Returns:
        logits: torch.Tensor of shape [batch, num_classes]
    """

    def __init__(self, num_classes, model_name="efficientnet-b0"):
        super().__init__()

        # Image branch (EfficientNet features only)
        if model_name == "efficientnet-b0":
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            img_feature_dim = 1280
        elif model_name == "efficientnet-b4":
            base_model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
            img_feature_dim = 1792
        elif model_name == "efficientnet-b7":
            base_model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)
            img_feature_dim = 2560
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
        self.image_encoder = base_model.features  # feature extractor
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # pool HxW → 1x1
        self.img_fc = nn.Linear(img_feature_dim, 512)  # reduce dim

        # Text branch (BERT)
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.txt_fc = nn.Linear(768, 512)  # reduce dim

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 + 512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        """Forward pass for the multimodal model.

        Args:
            images (torch.Tensor): Image batch [B, 3, H, W].
            input_ids (torch.Tensor): Token ids for text [B, L].
            attention_mask (torch.Tensor): Attention masks [B, L].

        Returns:
            torch.Tensor: Logits of shape [B, num_classes].
        """
        # Image branch
        img_feats = self.image_encoder(images)  # [batch, 1280, H, W]
        img_feats = self.global_pool(img_feats)  # [batch, 1280, 1, 1]
        img_feats = img_feats.view(img_feats.size(0), -1)  # flatten → [batch, 1280]
        img_feats = self.img_fc(img_feats)  # [batch, 512]

        # Text branch
        txt_feats = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feats = self.txt_fc(txt_feats.pooler_output)  # [batch, 512]

        # Combine branches and classify
        combined = torch.cat((img_feats, txt_feats), dim=1)  # [batch, 1024]
        output = self.classifier(combined)  # [batch, num_classes]
        return output
