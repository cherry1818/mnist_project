#plik jest definicją sieci konwolucyjnej
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module): # Definicja klasy dziedziczącej z nn.Module (bazowa klasa dla modeli w PyTorch)
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 warstwa konwolucyjna - 1 kanał wejściowy, 32 wyjściowe
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Warstwa spoolingująca (zmniejsza rozmiar obrazu o połowę)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Pierwsza warstwa w pełni połączona - 64 kanały * 7x7 po spooling
        self.fc2 = nn.Linear(128, 10)  # Wyjście z 10 neuronami (10 klas MNIST)

    def forward(self, x): # Funkcja opisująca przepływ danych przez model
        x = self.pool(F.relu(self.conv1(x)))  # Warstwa konwolucyjna + ReLU + pooling
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Spłaszczenie
        x = F.relu(self.fc1(x))  # Warstwa w pełni połączona + ReLU
        x = self.fc2(x) # Wyjście z 10 klasami
        return x
