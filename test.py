import torch

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # Wyłączenie gradientów (niepotrzebne podczas testowania)
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # Przesłanie danych na GPU
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) # Wybranie klasy z najwyższym prawdopodobieństwem
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    accuracy = 100. * correct / total
    return accuracy
