import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# Add the "Single layer Working model" directory to the path
single_layer_dir = os.path.join(os.getcwd(), "Single layer Working model")
sys.path.insert(0, single_layer_dir)  # Prioritize this path over others

# Import CPU matrix multiplication function
from cpu_custom_layer import CPUMatMulFunction

# Import from Single layer Working model/tuner_model.py
# We use a direct import with the full module path to avoid ambiguity
import importlib.util
spec = importlib.util.spec_from_file_location(
    "tuner_model_working", 
    os.path.join(single_layer_dir, "tuner_model.py")
)
tuner_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tuner_module)

# Get the functions we need from the module
load_model_and_scaler = tuner_module.load_model_and_scaler
predict_best_block_size = tuner_module.predict_best_block_size

# Force CPU usage
device = torch.device("cpu")

# Training parameters
BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
INPUT_FEATURES = 784  # MNIST input size (28x28)
HIDDEN_FEATURES = 256  # Hidden layer size
NUM_CLASSES = 10

# Load tuner model to predict block sizes
print("Loading tuner model...")
model_path = os.path.join(single_layer_dir, "tuner_model.pt")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found. Please train the model first.")

tuner_model, scaler = load_model_and_scaler(model_path, input_dim=12)
tuner_model.eval()

# Custom Linear layer that uses the CPUMatMulFunction but manages its own block size prediction
class CPUCustomLinear(nn.Module):
    def __init__(self, in_features, out_features, tuner_model=None, scaler=None, block_x=16, block_y=16):
        super(CPUCustomLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.tuner_model = tuner_model
        self.scaler = scaler
        self.block_x = block_x  # Default block size x
        self.block_y = block_y  # Default block size y
        
        # Save dimensions for prediction
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        # Validate input dimensions
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor, got {x.dim()}D tensor")
        if x.size(1) != self.in_features:
            raise ValueError(f"Expected input feature dimension {self.in_features}, got {x.size(1)}")
            
        x = x.contiguous()  # Ensure x is contiguous
        weight = self.weight.contiguous()  # Ensure weight is contiguous
        
        batch_size = x.size(0)  # M dimension for matmul and tuner input
        
        # Determine block sizes
        current_block_x = self.block_x
        current_block_y = self.block_y

        if self.tuner_model is not None and self.scaler is not None:
            # If tuner model and scaler are provided, predict the best block size
            try:
                # Predict best block size using the tuner model
                predicted_bx, predicted_by, _ = predict_best_block_size(
                    M=batch_size, 
                    K=self.in_features, 
                    N=self.out_features, 
                    model=self.tuner_model, 
                    scaler=self.scaler, 
                    device='cpu'  # We're on CPU
                )
                current_block_x = predicted_bx
                current_block_y = predicted_by
                
            except Exception as e:
                # Fallback to default block sizes if prediction fails
                print(f"Error predicting block size: {e}. Using default block sizes ({self.block_x}, {self.block_y}).")
        
        # Apply the custom matrix multiplication function using determined block sizes
        out = CPUMatMulFunction.apply(x, weight, current_block_x, current_block_y)
        
        # Add bias
        return out + self.bias

# Define a standard PyTorch MLP
class PyTorchMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(PyTorchMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define a custom MLP with our optimized matmul layer
class CustomMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, tuner_model, scaler):
        super(CustomMLP, self).__init__()
        self.custom_layer = CPUCustomLinear(input_size, hidden_size, tuner_model, scaler)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.custom_layer(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

def train_model(model_name, model, train_loader, test_loader):
    """Train the model with detailed timing measurements"""
    
    print(f"\n{'='*20} Training {model_name} {'='*20}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
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
            data = data.view(-1, INPUT_FEATURES)  # Flatten the images
            
            # Forward pass timing
            forward_start = time.time()
            outputs = model(data)
            loss = criterion(outputs, target)
            forward_end = time.time()
            forward_pass_times.append(forward_end - forward_start)
            
            # Backward pass timing
            backward_start = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(-1, INPUT_FEATURES)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
        val_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} completed in {epoch_time:.2f}s | "
              f"Loss: {running_loss/batch_count:.4f} | "
              f"Validation Accuracy: {accuracy:.2f}%")
    
    # Calculate final test accuracy
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, INPUT_FEATURES)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    
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
        'total_train_time': total_train_time,
        'avg_epoch_time': avg_epoch_time,
        'avg_forward_time': avg_forward_time,
        'avg_backward_time': avg_backward_time, 
        'test_accuracy': test_accuracy,
        'test_loss': test_loss
    }

def compare_results(pytorch_results, custom_results):
    """Compare and display the results of both models"""
    print("\n" + "="*60)
    print(f"{'Metric':<25} | {'PyTorch':<15} | {'Custom':<15} | {'Ratio':<10}")
    print("-"*60)
    
    # Calculate ratios (custom/pytorch)
    metrics = [
        ('Total Training Time (s)', pytorch_results['total_train_time'], custom_results['total_train_time']),
        ('Avg Epoch Time (s)', pytorch_results['avg_epoch_time'], custom_results['avg_epoch_time']),
        ('Avg Forward Pass (ms)', pytorch_results['avg_forward_time'], custom_results['avg_forward_time']),
        ('Avg Backward Pass (ms)', pytorch_results['avg_backward_time'], custom_results['avg_backward_time']),
        ('Test Accuracy (%)', pytorch_results['test_accuracy'], custom_results['test_accuracy']),
        ('Test Loss', pytorch_results['test_loss'], custom_results['test_loss'])
    ]
    
    for metric, pytorch_val, custom_val in metrics:
        ratio = custom_val / pytorch_val if pytorch_val != 0 else float('inf')
        print(f"{metric:<25} | {pytorch_val:<15.4f} | {custom_val:<15.4f} | {ratio:<10.4f}")
    
    print("="*60)
    
    # Save results to CSV
    with open('detailed_benchmark_results.csv', 'w') as f:
        f.write('Metric,PyTorch,Custom,Ratio\n')
        for metric, pytorch_val, custom_val in metrics:
            ratio = custom_val / pytorch_val if pytorch_val != 0 else float('inf')
            f.write(f"{metric},{pytorch_val},{custom_val},{ratio}\n")
    
    print("Detailed results saved to 'detailed_benchmark_results.csv'")

if __name__ == "__main__":
    print(f"Starting benchmark on CPU: PyTorch vs Custom MLP with optimal block sizes")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    
    # Get the block sizes predicted by the model
    print("Predicting optimal block sizes for benchmark batch size...")
    block_x, block_y, _ = predict_best_block_size(
        M=BATCH_SIZE, 
        K=INPUT_FEATURES, 
        N=HIDDEN_FEATURES, 
        model=tuner_model, 
        scaler=scaler, 
        device=device
    )
    print(f"Using block sizes: ({block_x}, {block_y}) for batch size {BATCH_SIZE}")
    
    # Load MNIST data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_data()
    
    # Initialize models
    pytorch_model = PyTorchMLP(INPUT_FEATURES, HIDDEN_FEATURES, NUM_CLASSES).to(device)
    custom_model = CustomMLP(INPUT_FEATURES, HIDDEN_FEATURES, NUM_CLASSES, tuner_model, scaler).to(device)
    
    # Train and evaluate models
    pytorch_results = train_model("PyTorch MLP", pytorch_model, train_loader, test_loader)
    custom_results = train_model("Custom MLP", custom_model, train_loader, test_loader)
    
    # Compare results
    compare_results(pytorch_results, custom_results) 