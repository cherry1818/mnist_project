#Ładowanie danych
import torch
from torchvision import datasets, transforms


def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Konwersja obrazu na tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalizacja danych (średnia = 0, odchylenie standardowe = 1)
    ])
    train_dataset = datasets.MNIST(
        root='./data', train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, transform=transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(  # Stworzenie loadera dla danych treningowych
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(  # Stworzenie loadera dla danych testowych
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader

