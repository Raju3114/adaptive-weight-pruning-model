import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# =====================================================
# 1. CUSTOM PRUNABLE LINEAR LAYER
# =====================================================
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        # Normal trainable weights
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01
        )

        self.bias = nn.Parameter(
            torch.zeros(out_features)
        )

        # Learnable gate scores (same shape as weight)
        self.gate_scores = nn.Parameter(
            torch.randn(out_features, in_features)
        )

    def forward(self, x):
        # Convert gate scores to values between 0 and 1
        gates = torch.sigmoid(self.gate_scores)

        # Apply pruning effect
        pruned_weights = self.weight * gates

        return F.linear(x, pruned_weights, self.bias)

    def get_gate_values(self):
        return torch.sigmoid(self.gate_scores)


# =====================================================
# 2. SELF PRUNING NETWORK
# =====================================================
class SelfPruningNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = PrunableLinear(32 * 32 * 3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def sparsity_loss(self):
        """
        L1-style sparsity penalty using gate values
        """
        loss = 0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                loss += module.get_gate_values().sum()
        return loss

    def calculate_sparsity(self, threshold=1e-2):
        """
        Percentage of gates effectively pruned
        """
        total = 0
        pruned = 0

        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates = module.get_gate_values()
                total += gates.numel()
                pruned += (gates < threshold).sum().item()

        return 100 * pruned / total

    def get_all_gate_values(self):
        values = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                values.extend(
                    module.get_gate_values()
                    .detach()
                    .cpu()
                    .flatten()
                    .tolist()
                )
        return values


# =====================================================
# 3. EVALUATION FUNCTION
# =====================================================
def evaluate_model(model, test_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# =====================================================
# 4. TRAINING FUNCTION
# =====================================================
def train_model(lambda_value=0.01, epochs=5, batch_size=128, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    # CIFAR-10 Dataset
    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = SelfPruningNet().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining with lambda = {lambda_value}\n")

    for epoch in range(epochs):
        model.train()
        total_epoch_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            classification_loss = criterion(outputs, labels)
            sparsity_penalty = model.sparsity_loss()

            # Main required formula
            total_loss = (
                classification_loss
                + lambda_value * sparsity_penalty
            )

            total_loss.backward()
            optimizer.step()

            total_epoch_loss += total_loss.item()

        avg_loss = total_epoch_loss / len(train_loader)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {avg_loss:.4f}"
        )

    # Final evaluation
    accuracy = evaluate_model(model, test_loader, device)
    sparsity = model.calculate_sparsity()

    print("\nFINAL RESULTS")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Sparsity Level: {sparsity:.2f}%")

    return model, accuracy, sparsity


# =====================================================
# 5. PLOT GATE DISTRIBUTION
# =====================================================
def plot_gate_distribution(model, lambda_value):
    values = model.get_all_gate_values()

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=50)

    plt.title(
        f"Gate Value Distribution (lambda={lambda_value})"
    )
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")

    filename = f"gate_distribution_lambda_{lambda_value}.png"
    plt.savefig(filename)
    plt.show()

    print(f"Plot saved as: {filename}")


# =====================================================
# 6. MAIN RUNNER
# =====================================================
if __name__ == "__main__":
    lambda_values = [0.001, 0.01, 0.1]

    results = []

    for lam in lambda_values:
        model, accuracy, sparsity = train_model(
            lambda_value=lam,
            epochs=5
        )

        results.append((lam, accuracy, sparsity))

        plot_gate_distribution(model, lam)

    print("\nSUMMARY TABLE")
    print("Lambda | Accuracy | Sparsity %")
    print("-" * 40)

    for row in results:
        print(
            f"{row[0]:<7} | "
            f"{row[1]:<8.2f} | "
            f"{row[2]:<10.2f}"
        )