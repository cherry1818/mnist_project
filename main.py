import torch
from model import CNN
from utils import load_data
from train import train
from test import test

def main():
    batch_size = 64 #liczba przykładów przetwarzanych jednoczesnie
    epochs = 5 #ile razy przepuskamy zbiór danych przez model
    learning_rate = 0.001

    # Wykrywanie urządzenia
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ładowanie danych
    train_loader, test_loader = load_data(batch_size)

    # Tworzenie modelu
    model = CNN().to(device)

    # Optymalizator
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Trening
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        accuracy = test(model, test_loader, device)
        print(f"Epoch {epoch}: Loss={train_loss:.4f}, Accuracy={accuracy:.2f}%")

if __name__ == "__main__":
    main()
