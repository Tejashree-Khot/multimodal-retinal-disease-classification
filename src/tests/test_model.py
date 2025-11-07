# tests/test_model.py
from models.simple_model import SimpleCNN
from src.models.multimodel import MultiModalModel


def test_multimodal_model():
    """Test the MultiModalModel"""
    # Initialize the model
    model = MultiModalModel(num_classes=5, model_name="efficientnet-b0")
    assert model is not None, "Model initialization failed"
    assert isinstance(model, MultiModalModel), "Model is not an instance of MutliModalModel"
    assert len(model.image_encoder) > 0, "Image encoder layers are not defined"
    assert len(list(model.text_encoder.parameters())) > 0, "Text encoder layers are not defined"
    assert len(model.classifier) > 0, "Classifier layers are not defined"


def test_simple_cnn():
    """Test the SimpleCNN model."""
    # Initialize the model
    model = SimpleCNN()
    assert model is not None, "Model initialization failed"
    assert isinstance(model, SimpleCNN), "Model is not an instance of SimpleCNN"
    assert len(model.conv) > 0, "Convolutional layers are not defined"
    assert len(model.fc) > 0, "Fully connected layers are not defined"


if __name__ == "__main__":
    test_multimodal_model()
    test_simple_cnn()
    print("All tests passed!")
