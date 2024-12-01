import torch
import torch.nn.functional as F

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0  # Inicjalizacja całkowitej straty
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) # Przesłanie danych na GPU (jeśli dostępne)
        optimizer.zero_grad() # Zerowanie gradientów
        output = model(data)
        loss = F.cross_entropy(output, target) # Obliczenie straty (cross-entropy)
        loss.backward() # Backpropagation
        optimizer.step() # Aktualizacja wag modelu
        total_loss += loss.item() # Sumowanie strat
    return total_loss / len(train_loader)
