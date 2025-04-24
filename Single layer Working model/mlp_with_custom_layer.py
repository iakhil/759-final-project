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
import argparse
import torch.cuda

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, tuner_model=None, scaler=None):
        super(MLP, self).__init__()
        
        # Initialize the custom linear layer with the tuner model
        self.layer1 = CustomLinear(input_size, hidden_size, tuner_model=tuner_model, scaler=scaler)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)  # Regular PyTorch linear layer
        self.input_size = input_size
    
    def forward(self, x):
        # Reshape input to ensure correct dimensions
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten all dimensions except batch
        
        # Ensure input has the correct number of features
        if x.size(1) != self.input_size:
            raise ValueError(f"Expected input with {self.input_size} features, got {x.size(1)}")
            
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add command-line argument parsing
parser = argparse.ArgumentParser(description='Train an MLP with custom layer on MNIST')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--block_x', type=int, default=16, help='Block size x dimension')
parser.add_argument('--block_y', type=int, default=16, help='Block size y dimension')
args = parser.parse_args()

# Update hyperparameters
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
block_x = args.block_x
block_y = args.block_y

# Load MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# Initialize model
model = MLP(input_size=784, hidden_size=256, output_size=10, tuner_model=None, scaler=None).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with timing and CSV logging
csv_path = "training_log.csv"

# Open the CSV file and keep it open for the duration of the training loop
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    # Write the header row
    writer.writerow(["epoch", "batch", "loss", "time_ms"])

    # Add timing variables
    start_time = time.time()
    epoch_times = []
    forward_times = []
    backward_times = []

    # Main training loop - now inside the 'with' block
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
            forward_times.append(start_event.elapsed_time(end_event))
            
            # Time backward pass
            start_event.record()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_event.record()
            
            torch.cuda.synchronize()
            backward_times.append(start_event.elapsed_time(end_event))

            # Write data for the current batch to the CSV file
            writer.writerow([epoch+1, batch_idx+1, loss.item(), (end_event.elapsed_time(start_event)*1000)])

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Time: {epoch_time:.2f}s")

# The 'with' block ends here, and the CSV file 'f' is automatically closed.

# After training, calculate and print timing statistics
# These calculations happen *after* the file is closed, which is fine.
total_time = time.time() - start_time
avg_epoch_time = sum(epoch_times) / len(epoch_times)
avg_forward_time = sum(forward_times) / len(forward_times)
avg_backward_time = sum(backward_times) / len(backward_times)

print("\n===== CUSTOM MLP PERFORMANCE =====")
print(f"Block size configuration: ({block_x}, {block_y})")
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
torch.save(model.state_dict(), "mlp_with_custom_layer.pt")
print("Model saved to mlp_with_custom_layer.pt")
