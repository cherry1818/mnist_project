#plik jest definicją sieci konwolucyjnej
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # zmniejsza rozmiar obrazu o połowę
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Pierwsza warstwa w pełni połączona
        self.fc2 = nn.Linear(128, 10)  # Wyjście z 10 neuronami

    # Funkcja opisująca przepływ danych przez model
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Warstwa konwolucyjna + ReLU + pooling
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Spłaszczenie
        x = F.relu(self.fc1(x))  # Warstwa w pełni połączona + ReLU
        x = self.fc2(x)
        return x
