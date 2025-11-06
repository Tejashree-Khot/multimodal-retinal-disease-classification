"""Constants for class names and mapping used across the dataloader.

This module exposes two constants:

- CLASSES: ordered list of class names used by the dataset.
- CLASSES_DICT: mapping from class name (string) to integer label.

Keeping these definitions in a single module prevents hard-coded strings
scattered across the codebase and makes it easy to update label ordering.
"""

CLASSES = [
    "Clinically_Significant_Macular_Edema",
    "No_DR",
    "normal",
    "Mild_Moderate_NPDR",
    "Severe_PDR",
]

CLASSES_DICT = {
    "Clinically_Significant_Macular_Edema": 0,
    "No_DR": 1,
    "normal": 2,
    "Mild_Moderate_NPDR": 3,
    "Severe_PDR": 4,
}
