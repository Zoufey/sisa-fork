from classification_models.keras import Classifiers
import torch
import pytorch_keras

# Lade das Keras-Modell
ResNet18, _ = Classifiers.get('resnet18')
keras_model = ResNet18(input_shape=(32, 32, 3), num_classes=10) 

# Konvertiere zu PyTorch (falls pytorch-keras verfügbar ist)
pytorch_model = pytorch_keras.convert(keras_model)

# Speichere als PyTorch-Modell
torch.save(pytorch_model.state_dict(), 'resnet18_pytorch.pt')

# Erstelle eine Wrapper-Klasse für SISA
class Model(torch.nn.Module):
    def __init__(self, input_shape, nb_classes, dropout_rate=0.0):
        super(Model, self).__init__()
        self.model = pytorch_model  # Geladenes Modell

    def forward(self, x):
        return self.model(x)