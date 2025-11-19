# tests/test_model.py
from models.simple_model import SimpleCNN
from src.models.multimodel import MultiModalModel
from src.models.efficientnet_model import get_efficientnet_model


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


def test_efficientnet_model():
    """Test the EfficientNet model loading."""
    model = get_efficientnet_model(num_classes=5, model_name="efficientnet-b0")
    assert model is not None, "EfficientNet model initialization failed"
    assert len(list(model.parameters())) > 0, "Model parameters are not defined"


if __name__ == "__main__":
    test_multimodal_model()
    test_simple_cnn()
    test_efficientnet_model()
    print("All tests passed!")
