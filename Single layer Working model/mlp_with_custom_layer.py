import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from custom_layer import CustomLinear
from tuner_model import load_model_and_scaler, predict_best_block_size
import time
import csv
import os

class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, tuner_path="tuner_model.pt"):
        super(MLP, self).__init__()

        # Load tuner model and scaler
        if os.path.exists(tuner_path):
            tuner_model, scaler = load_model_and_scaler(tuner_path, input_dim=12)
            print(f"Loaded tuner model and scaler from {tuner_path}")
        else:
            raise FileNotFoundError(f"Tuner model not found at {tuner_path}")

        self.tuner_model = tuner_model
        self.scaler = scaler
        self.tuner_model.eval()

        self.layers = nn.Sequential(
            CustomLinear(input_size, hidden_size, self.tuner_model, self.scaler),
            nn.ReLU(),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.layers(x)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 784
hidden_size = 256
num_classes = 10
num_epochs = 5
batch_size = 64
learning_rate = 0.001

# Load MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# Initialize model
model = MLP(input_size, hidden_size, tuner_path="tuner_model.pt").to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with timing and CSV logging
csv_path = "training_log.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "batch", "loss", "time_ms"])

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            batch_start = time.time()
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_end = time.time()

            writer.writerow([epoch+1, batch_idx+1, loss.item(), (batch_end - batch_start)*1000])

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy on test set: {100 * correct / total:.2f}%')

# Save the model
torch.save(model.state_dict(), "mlp_with_custom_layer.pt")
print("Model saved to mlp_with_custom_layer.pt")
