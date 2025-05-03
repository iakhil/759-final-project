import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the "Single layer Working model" directory to the path
single_layer_dir = os.path.join(os.getcwd(), "Single layer Working model")
sys.path.insert(0, single_layer_dir)  # Prioritize this path over others

# Import CPU custom layer
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

# Configure the device - we'll run both on CPU for fair comparison
device = torch.device("cpu")

# Constants for the benchmark
BATCH_SIZES = [1, 8, 16, 32, 64, 128, 256, 512]
INPUT_FEATURES = 784  # MNIST input size (28x28)
HIDDEN_FEATURES = 256  # Hidden layer size
NUM_CLASSES = 10
NUM_RUNS = 10  # Number of runs for each configuration to get an average

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

def benchmark_models():
    results = {'batch_size': [], 'pytorch_time': [], 'custom_time': [], 'speedup': [], 'block_x': [], 'block_y': []}
    
    for batch_size in BATCH_SIZES:
        print(f"Benchmarking with batch size: {batch_size}")
        
        # Create input data
        x = torch.randn(batch_size, INPUT_FEATURES, device=device)
        
        # Initialize the models
        pytorch_model = PyTorchMLP(INPUT_FEATURES, HIDDEN_FEATURES, NUM_CLASSES).to(device)
        custom_model = CustomMLP(INPUT_FEATURES, HIDDEN_FEATURES, NUM_CLASSES, tuner_model, scaler).to(device)
        
        # Record the block sizes predicted by the model
        block_x, block_y, _ = predict_best_block_size(
            M=batch_size, 
            K=INPUT_FEATURES, 
            N=HIDDEN_FEATURES, 
            model=tuner_model, 
            scaler=scaler, 
            device=device
        )
        
        # Warm-up runs
        for _ in range(3):
            _ = pytorch_model(x)
            _ = custom_model(x)
        
        # Benchmark PyTorch model
        pytorch_times = []
        for _ in range(NUM_RUNS):
            start = time.time()
            _ = pytorch_model(x)
            end = time.time()
            pytorch_times.append((end - start) * 1000)  # ms
        
        # Benchmark Custom model
        custom_times = []
        for _ in range(NUM_RUNS):
            start = time.time()
            _ = custom_model(x)
            end = time.time()
            custom_times.append((end - start) * 1000)  # ms
        
        # Calculate averages
        avg_pytorch_time = np.mean(pytorch_times)
        avg_custom_time = np.mean(custom_times)
        speedup = avg_pytorch_time / avg_custom_time if avg_custom_time > 0 else float('inf')
        
        # Store results
        results['batch_size'].append(batch_size)
        results['pytorch_time'].append(avg_pytorch_time)
        results['custom_time'].append(avg_custom_time)
        results['speedup'].append(speedup)
        results['block_x'].append(block_x)
        results['block_y'].append(block_y)
        
        print(f"  PyTorch time: {avg_pytorch_time:.4f} ms")
        print(f"  Custom time: {avg_custom_time:.4f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Block sizes: ({block_x}, {block_y})")
        print()
    
    return results

def plot_results(results):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot 1: Runtime comparison
    ax1.plot(results['batch_size'], results['pytorch_time'], 'o-', label='PyTorch CPU')
    ax1.plot(results['batch_size'], results['custom_time'], 's-', label='Custom with Predicted Blocks')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Runtime Comparison: PyTorch vs Custom Implementation')
    ax1.grid(True, which='both', linestyle='--', alpha=0.6)
    ax1.legend()
    
    # Plot 2: Speedup and block sizes
    ax2.bar(range(len(results['batch_size'])), results['speedup'], label='Speedup')
    ax2.set_xticks(range(len(results['batch_size'])))
    ax2.set_xticklabels([f"{bs}\n({bx},{by})" for bs, bx, by in 
                         zip(results['batch_size'], results['block_x'], results['block_y'])])
    ax2.set_xlabel('Batch Size (with Block Sizes)')
    ax2.set_ylabel('Speedup (x times)')
    ax2.set_title('Speedup with Predicted Block Sizes')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    print("Results plot saved as 'benchmark_results.png'")
    
    # Also display the numerical results in a readable format
    print("\n----- Benchmark Results Summary -----")
    print(f"{'Batch Size':<10} | {'PyTorch (ms)':<12} | {'Custom (ms)':<12} | {'Speedup':<8} | {'Block Sizes':<10}")
    print("-" * 60)
    for i in range(len(results['batch_size'])):
        print(f"{results['batch_size'][i]:<10} | {results['pytorch_time'][i]:<12.4f} | "
              f"{results['custom_time'][i]:<12.4f} | {results['speedup'][i]:<8.2f} | "
              f"({results['block_x'][i]}, {results['block_y'][i]})")

if __name__ == "__main__":
    print("Starting benchmark: PyTorch CPU vs Custom MLP with optimal block sizes")
    results = benchmark_models()
    plot_results(results) 