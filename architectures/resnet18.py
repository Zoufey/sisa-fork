import torch
import torch.nn as nn
from torchvision.models import resnet18
import numpy as np

#Wird ueber Argumente in Aufruf gesetzt
BATCH_SIZE = 64  # --batch_size
EPOCHS = 30 # --epochs
LEARNING_RATE = 0.001 # --learning_rate
OPTIMIZER = "adam" # --optimizer adam


IMAGE_SIZE = 32 
NUM_CLASSES = 10

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

class Model(nn.Module):
    def __init__(self, input_shape, nb_classes, dropout_rate=0.0): # from config file
        super(Model, self).__init__()
        #ResNet18-Modell von torchvision
        self.model = resnet18(weights=False)  # pretrained=False, da von Grund auf trainieren, wie in Baseline
        # --- Anpassung für CIFAR-10 (32x32 Bilder) ---
        # Standard-ResNet18 erwartet 3x224x224, aber CIFAR10 3x32x32.
        # Deswegen muss InputLayer = erste Conv-Layer angepasst werden:
        # - kernel_size=3 (statt 7), um die räumliche Dimension nicht zu stark zu reduzieren
        # - stride=1 (statt 2), um die Bildgröße beizubehalten
        # - padding=1 (statt 3), da kernel_size=3
        if input_shape[0] != 3 or input_shape[1:] != (224, 224):
            self.model.conv1 = nn.Conv2d(
                in_channels=input_shape[0],  # 3 für CIFAR-10
                out_channels=64,
                kernel_size=3,  # 3x3 statt 7x7
                stride=1,       # stride=1 statt 2 (kein Downsampling am Anfang)
                padding=1,      # padding=1 für kernel_size=3
                bias=False
            )

        # Entfernung des ersten MaxPooling (stride=2 würde 32x32 → 16x16 reduzieren)
        self.model.maxpool = nn.Identity() 

       # --- Anpassung der letzten Layer ---
        #  letzte Fully Connected Layer an die Anzahl der Klassen anpassen
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, nb_classes)


    def forward(self, x):
        return self.model(x)