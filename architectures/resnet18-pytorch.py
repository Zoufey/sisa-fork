import torch
import torch.nn as nn
from torchvision.models import resnet18

#Wird ueber Argumente in Aufruf gesetzt
BATCH_SIZE = 64  # --batch_size
EPOCHS = 30 # --epochs
LEARNING_RATE = 0.001 # --learning_rate
OPTIMIZER = "Adam" # --optimizer adam


IMAGE_SIZE = 32 
NUM_CLASSES = 10

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

class Model(nn.Module):
    def __init__(self, input_shape, nb_classes, dropout_rate=0.0):
        super(Model, self).__init__()
        #ResNet18-Modell von torchvision
        self.model = resnet18(pretrained=False)  # pretrained=False, da wir von Grund auf trainieren

        # Passe die erste Convolutional Layer an die Eingabegröße an
        # Standardmäßig erwartet ResNet18 3 Kanäle und 224x224 Bilder
        if input_shape[0] != 3 or input_shape[1:] != (224, 224):
            # Ersetze die erste Conv-Layer, falls die Eingabegröße nicht 3x224x224 ist
            self.model.conv1 = nn.Conv2d(
                input_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Passe die letzte Layer an die Anzahl der Klassen an
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, nb_classes)

    def forward(self, x):
        return self.model(x)