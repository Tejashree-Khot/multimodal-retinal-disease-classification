"""A simple CNN model used for experiments and embedding visualization.

This model is intentionally small and not intended for production use. It is
useful for quick tests and extracting intermediate convolutional embeddings
for visualization with t-SNE.
"""

from torch import nn


class SimpleCNN(nn.Module):
    """Small convolutional classifier.

    Architecture:
        - Two convolutional blocks with ReLU + MaxPool
        - A small fully-connected head producing logits for 5 classes
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 54 * 54, 64), nn.ReLU(), nn.Linear(64, 5)
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input image tensor of shape [B, 3, H, W].

        Returns:
            torch.Tensor: Logits of shape [B, 5].
        """
        x = self.conv(x)
        x = self.fc(x)
        return x
