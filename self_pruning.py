import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# 1. Prunable Linear Layer
# ==============================
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features) - 3)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores * 3)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

# ==============================
# 2. Neural Network
# ==============================
class PrunableNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = PrunableLinear(32*32*3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ==============================
# 3. Sparsity Loss
# ==============================
def compute_sparsity_loss(model):
    loss = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            loss += torch.mean(torch.abs(gates))
    return loss

# ==============================
# 4. Sparsity Calculation
# ==============================
def calculate_sparsity(model, threshold=0.1):
    total = 0
    pruned = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            total += gates.numel()
            pruned += (gates < threshold).sum().item()

    return 100 * pruned / total

# ==============================
# 5. Evaluation Function
# ==============================
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

# ==============================
# 6. Visualization Functions
# ==============================

# Training Loss Plot
def plot_loss(losses, lambda_val):
    plt.figure()
    plt.plot(losses)
    plt.title(f"Training Loss (Lambda={lambda_val})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

# Accuracy vs Lambda
def plot_accuracy_vs_lambda(results):
    lambdas = [r[0] for r in results]
    accuracies = [r[1] for r in results]

    plt.figure()
    plt.plot(lambdas, accuracies, marker='o')
    plt.xscale('log')
    plt.title("Accuracy vs Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("Accuracy (%)")
    plt.grid()
    plt.show()

# Sparsity vs Lambda
def plot_sparsity_vs_lambda(results):
    lambdas = [r[0] for r in results]
    sparsities = [r[2] for r in results]

    plt.figure()
    plt.plot(lambdas, sparsities, marker='o')
    plt.xscale('log')
    plt.title("Sparsity vs Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("Sparsity (%)")
    plt.grid()
    plt.show()

# Gate Distribution
def plot_gates(model):
    all_gates = []

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy()
            all_gates.extend(gates.flatten())

    plt.figure()
    plt.hist(all_gates, bins=50)
    plt.title("Gate Value Distribution (Pruning Effect)")
    plt.xlabel("Gate Value (0 = pruned, 1 = active)")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

# ==============================
# 7. Data Loading (CIFAR-10)
# ==============================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ==============================
# 8. Training Function
# ==============================
def train_model(lambda_val):
    model = PrunableNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 15
    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss_val = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            classification_loss = F.cross_entropy(outputs, labels)
            sparsity_loss = compute_sparsity_loss(model)

            loss = classification_loss + lambda_val * sparsity_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_val += loss.item()

        epoch_losses.append(total_loss_val)
        print(f"Lambda {lambda_val} | Epoch {epoch+1} | Loss: {total_loss_val:.2f}")

    accuracy = evaluate(model, test_loader)
    sparsity = calculate_sparsity(model)

    return model, accuracy, sparsity, epoch_losses

# ==============================
# 9. Main Execution
# ==============================
if __name__ == "__main__":

    lambda_values = [0.5, 1.0, 5.0, 20.0]
    results = []

    best_model = None
    best_acc = 0

    for lam in lambda_values:
        print(f"\nTraining with lambda = {lam}")

        model, acc, sparsity, losses = train_model(lam)

        plot_loss(losses, lam)  # 🔥 Loss graph

        print(f"Lambda: {lam} | Accuracy: {acc:.2f}% | Sparsity: {sparsity:.2f}%")

        results.append((lam, acc, sparsity))

        if acc > best_acc:
            best_acc = acc
            best_model = model

# Results Table
print("\nFinal Results:")
print("Lambda\tAccuracy\tSparsity")
for r in results:
    print(f"{r[0]}\t{r[1]:.2f}%\t\t{r[2]:.2f}%")

# Final Visualizations
plot_accuracy_vs_lambda(results)
plot_sparsity_vs_lambda(results)
plot_gates(best_model)