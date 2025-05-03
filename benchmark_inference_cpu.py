import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Device configuration - PyTorch on CPU, Custom on GPU
cpu_device = torch.device("cpu")
gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch will run on: {cpu_device}")
print(f"Custom model will run on: {gpu_device}")

if gpu_device.type == "cpu":
    print("WARNING: CUDA not available, both models will run on CPU!")

# Add the "Single layer Working model" directory to the path
single_layer_dir = os.path.join(os.getcwd(), "Single layer Working model")
sys.path.insert(0, single_layer_dir)

# Import custom layer
from custom_layer import CustomLinear, MatMulFunction

# Import tuner model
import importlib.util
spec = importlib.util.spec_from_file_location(
    "tuner_model_working", 
    os.path.join(single_layer_dir, "tuner_model.py")
)
tuner_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tuner_module)

# Get the functions we need
load_model_and_scaler = tuner_module.load_model_and_scaler
predict_best_block_size = tuner_module.predict_best_block_size

# Training parameters
INPUT_SIZE = 784  # MNIST image size (28x28)
HIDDEN_SIZE1 = 256
HIDDEN_SIZE2 = 128
NUM_CLASSES = 10
BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 0.001

# Standard PyTorch MLP model (will run on CPU)
class PyTorchMLP(nn.Module):
    def __init__(self):
        super(PyTorchMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE1),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE1, HIDDEN_SIZE2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE2, NUM_CLASSES)
        )

    def forward(self, x):
        x = x.view(-1, INPUT_SIZE)
        return self.layers(x)

# Custom MLP model with GPU custom layer
class CustomMLP(nn.Module):
    def __init__(self, tuner_model=None, scaler=None):
        super(CustomMLP, self).__init__()
        self.layers = nn.Sequential(
            CustomLinear(INPUT_SIZE, HIDDEN_SIZE1, tuner_model, scaler),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE1, HIDDEN_SIZE2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE2, NUM_CLASSES)
        )

    def forward(self, x):
        x = x.view(-1, INPUT_SIZE)
        return self.layers(x)

def load_data():
    """Load MNIST dataset for training and testing"""
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

def initialize_models():
    """Initialize and load both PyTorch and Custom models"""
    
    # Load tuner model for block size prediction
    print("Loading tuner model...")
    model_path = os.path.join(single_layer_dir, "tuner_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Tuner model file {model_path} not found")
    
    tuner_model, scaler = load_model_and_scaler(model_path, input_dim=12)
    tuner_model = tuner_model.to(gpu_device)  # Move tuner model to GPU
    tuner_model.eval()
    
    # Initialize models on respective devices
    pytorch_model = PyTorchMLP().to(cpu_device)  # PyTorch on CPU
    custom_model = CustomMLP(tuner_model, scaler).to(gpu_device)  # Custom on GPU
    
    return pytorch_model, custom_model, tuner_model, scaler

def train_model(model_name, model, train_loader, test_loader, device):
    """Train the model with detailed timing measurements"""
    
    print(f"\n{'='*20} Training {model_name} on {device.type.upper()} {'='*20}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Timing variables
    total_train_start = time.time()
    epoch_times = []
    forward_pass_times = []
    backward_pass_times = []
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        # Training
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass timing
            forward_start = time.time()
            outputs = model(data)
            loss = criterion(outputs, target)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            forward_end = time.time()
            forward_pass_times.append(forward_end - forward_start)
            
            # Backward pass timing
            backward_start = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            backward_end = time.time()
            backward_pass_times.append(backward_end - backward_start)
            
            running_loss += loss.item()
            batch_count += 1
            
            # Print batch progress for every 100 batches
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Avg Forward: {np.mean(forward_pass_times[-100:]) * 1000:.2f}ms, "
                      f"Avg Backward: {np.mean(backward_pass_times[-100:]) * 1000:.2f}ms")
        
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)
        
        # Validate after each epoch
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
        val_loss /= total
        accuracy = 100. * correct / total
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} completed in {epoch_time:.2f}s | "
              f"Loss: {running_loss/batch_count:.4f} | "
              f"Validation Accuracy: {accuracy:.2f}%")
    
    # Calculate final test accuracy
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_loss /= total
    test_accuracy = 100. * correct / total
    
    total_train_end = time.time()
    total_train_time = total_train_end - total_train_start
    avg_epoch_time = np.mean(epoch_times)
    avg_forward_time = np.mean(forward_pass_times) * 1000  # Convert to ms
    avg_backward_time = np.mean(backward_pass_times) * 1000  # Convert to ms
    
    # Print the final results
    print(f"\n{'='*20} {model_name} Results {'='*20}")
    print(f"Total Training Time: {total_train_time:.2f} seconds")
    print(f"Average Epoch Time: {avg_epoch_time:.2f} seconds")
    print(f"Average Forward Pass Time: {avg_forward_time:.4f} ms")
    print(f"Average Backward Pass Time: {avg_backward_time:.4f} ms")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    return {
        'name': model_name,
        'device': device.type,
        'total_train_time': total_train_time,
        'avg_epoch_time': avg_epoch_time,
        'avg_forward_time': avg_forward_time,
        'avg_backward_time': avg_backward_time, 
        'test_accuracy': test_accuracy,
        'test_loss': test_loss
    }

def compare_results(pytorch_results, custom_results):
    """Compare and display the results of both models"""
    print("\n" + "="*75)
    print(f"{'Metric':<25} | {'PyTorch (CPU)':<15} | {'Custom (GPU)':<15} | {'Speedup':<10}")
    print("-"*75)
    
    # Calculate speedups (pytorch_time / custom_time)
    metrics = [
        ('Total Training Time (s)', pytorch_results['total_train_time'], custom_results['total_train_time']),
        ('Avg Epoch Time (s)', pytorch_results['avg_epoch_time'], custom_results['avg_epoch_time']),
        ('Avg Forward Pass (ms)', pytorch_results['avg_forward_time'], custom_results['avg_forward_time']),
        ('Avg Backward Pass (ms)', pytorch_results['avg_backward_time'], custom_results['avg_backward_time'])
    ]
    
    for metric, pytorch_val, custom_val in metrics:
        speedup = pytorch_val / custom_val if custom_val > 0 else float('inf')
        print(f"{metric:<25} | {pytorch_val:<15.4f} | {custom_val:<15.4f} | {speedup:<10.2f}x")
    
    # Accuracy doesn't have a speedup, so display separately
    print(f"{'Test Accuracy (%)':<25} | {pytorch_results['test_accuracy']:<15.2f} | {custom_results['test_accuracy']:<15.2f} | {'-':<10}")
    print(f"{'Test Loss':<25} | {pytorch_results['test_loss']:<15.4f} | {custom_results['test_loss']:<15.4f} | {'-':<10}")
    
    print("="*75)
    
    # Save results to CSV
    import csv
    with open('training_benchmark_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'PyTorch (CPU)', 'Custom (GPU)', 'Speedup'])
        for metric, pytorch_val, custom_val in metrics:
            speedup = pytorch_val / custom_val if custom_val > 0 else float('inf')
            writer.writerow([metric, pytorch_val, custom_val, speedup])
        writer.writerow(['Test Accuracy (%)', pytorch_results['test_accuracy'], custom_results['test_accuracy'], '-'])
        writer.writerow(['Test Loss', pytorch_results['test_loss'], custom_results['test_loss'], '-'])
    
    print("Training benchmark results saved to 'training_benchmark_results.csv'")
    
    # Create bar chart comparing training metrics
    plt.figure(figsize=(12, 8))
    
    # Convert data for plotting
    metric_names = [m[0] for m in metrics]
    pytorch_values = [m[1] for m in metrics]
    custom_values = [m[2] for m in metrics]
    
    # Calculate speedup text
    speedup_text = [f"{pytorch_values[i]/custom_values[i]:.1f}x" for i in range(len(metric_names))]
    
    # Plotting
    x = np.arange(len(metric_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width/2, pytorch_values, width, label='PyTorch (CPU)')
    bars2 = ax.bar(x + width/2, custom_values, width, label='Custom (GPU)')
    
    # Add speedup text above bars
    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        speed = pytorch_values[i] / custom_values[i] if custom_values[i] > 0 else float('inf')
        height = max(b1.get_height(), b2.get_height())
        ax.text(i, height * 1.05, f"{speed:.1f}x faster", 
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.set_ylabel('Time (seconds/milliseconds)')
    ax.set_title('Training Performance: PyTorch on CPU vs Custom on GPU')
    
    plt.tight_layout()
    plt.savefig('training_benchmark_plot.png')
    print("Training benchmark plot saved as 'training_benchmark_plot.png'")

if __name__ == "__main__":
    print("Starting training benchmark: PyTorch on CPU vs Custom MLP on GPU")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Get the predicted block sizes for the batch size
    print(f"Loading tuner to predict optimal block sizes for batch size {BATCH_SIZE}...")
    tuner_model, scaler = load_model_and_scaler(os.path.join(single_layer_dir, "tuner_model.pt"), input_dim=12)
    tuner_model = tuner_model.to(gpu_device)
    block_x, block_y, _ = predict_best_block_size(
        M=BATCH_SIZE, 
        K=INPUT_SIZE, 
        N=HIDDEN_SIZE1, 
        model=tuner_model, 
        scaler=scaler, 
        device=gpu_device
    )
    print(f"Using block sizes: ({block_x}, {block_y}) for batch size {BATCH_SIZE}")
    
    # Load MNIST data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_data()
    
    # Initialize models
    pytorch_model, custom_model, _, _ = initialize_models()
    
    # Train and evaluate models
    pytorch_results = train_model("PyTorch MLP", pytorch_model, train_loader, test_loader, cpu_device)
    custom_results = train_model("Custom MLP", custom_model, train_loader, test_loader, gpu_device)
    
    # Compare results
    compare_results(pytorch_results, custom_results) 