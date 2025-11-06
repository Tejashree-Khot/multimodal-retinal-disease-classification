"""Small helper script to download the IDRiD dataset from Kaggle.

This script was added for convenience during development. It uses
`kagglehub` to download the dataset and prints the path to the
downloaded files.

Note: the script is a minimal example and may require authentication
or configuration of Kaggle credentials to run successfully.
"""

import kagglehub


# Download latest version
path = kagglehub.dataset_download(
    "mohamedabdalkader/indian-diabetic-retinopathy-image-dataset-idrid"
)

print("Path to dataset files:", path)
