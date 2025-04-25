import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import csv
import os
import argparse
import torch.cuda

# Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Original logic

# Add command-line argument parsing
parser = argparse.ArgumentParser(description='Train a standard MLP on MNIST')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                    choices=['cuda', 'cpu'], help='Device to run the training on (cuda or cpu)') # Added device argument
args = parser.parse_args()

# Set device based on argument
device = torch.device(args.device)
print(f"Using device: {device}") # Added print statement for confirmation

# Update hyperparameters
input_size = 784  # 28x28 images
hidden_size1 = 256
hidden_size2 = 128
num_classes = 10
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr

# Load dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# Define MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, num_classes)
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.layers(x)

model = MLP().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with timing and CSV logging
csv_path = "standard_mlp_training_log.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "batch", "loss", "time_ms"])

# Add timing variables
start_time = time.time()
epoch_times = []
forward_times = []
backward_times = []

for epoch in range(num_epochs):
    epoch_start = time.time()
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Time forward pass
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        outputs = model(images)
        loss = criterion(outputs, labels)
        end_event.record()
        
        torch.cuda.synchronize()
        forward_time = start_event.elapsed_time(end_event)
        forward_times.append(forward_time)
        
        # Time backward pass
        start_event.record()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end_event.record()
        
        torch.cuda.synchronize()
        backward_time = start_event.elapsed_time(end_event)
        backward_times.append(backward_time)
        
        # Write to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, batch_idx+1, loss.item(), forward_time + backward_time])

    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start
    epoch_times.append(epoch_time)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Time: {epoch_time:.2f}s")

# After training, add timing statistics
total_time = time.time() - start_time
avg_epoch_time = sum(epoch_times) / len(epoch_times)
avg_forward_time = sum(forward_times) / len(forward_times)
avg_backward_time = sum(backward_times) / len(backward_times)

print("\n===== STANDARD MLP PERFORMANCE =====")
print(f"Total training time: {total_time:.2f} seconds")
print(f"Average epoch time: {avg_epoch_time:.2f} seconds")
print(f"Average forward pass time: {avg_forward_time:.2f} ms")
print(f"Average backward pass time: {avg_backward_time:.2f} ms")

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
torch.save(model.state_dict(), "standard_mlp.pt")
print("Model saved to standard_mlp.pt")
