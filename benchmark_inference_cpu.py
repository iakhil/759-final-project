import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Force CPU usage
device = torch.device("cpu")
print(f"Using device: {device}")

# Add the "Single layer Working model" directory to the path
single_layer_dir = os.path.join(os.getcwd(), "Single layer Working model")
sys.path.insert(0, single_layer_dir)

# Import CPU custom layer
from cpu_custom_layer import CPUMatMulFunction

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

# Hyperparameters
INPUT_SIZE = 784  # MNIST image size (28x28)
HIDDEN_SIZE1 = 256
HIDDEN_SIZE2 = 128
NUM_CLASSES = 10
BATCH_SIZES = [1, 8, 16, 32, 64, 128, 256, 512]
NUM_RUNS = 100  # Number of inference runs to average

# Custom Linear layer implementation for CPU
class CPUCustomLinear(nn.Module):
    def __init__(self, in_features, out_features, tuner_model=None, scaler=None, block_x=16, block_y=16):
        super(CPUCustomLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.tuner_model = tuner_model
        self.scaler = scaler
        self.block_x = block_x
        self.block_y = block_y
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor, got {x.dim()}D tensor")
        if x.size(1) != self.in_features:
            raise ValueError(f"Expected input feature dimension {self.in_features}, got {x.size(1)}")
            
        x = x.contiguous()
        weight = self.weight.contiguous()
        
        batch_size = x.size(0)
        
        # Determine block sizes
        current_block_x = self.block_x
        current_block_y = self.block_y

        if self.tuner_model is not None and self.scaler is not None:
            try:
                predicted_bx, predicted_by, _ = predict_best_block_size(
                    M=batch_size, 
                    K=self.in_features, 
                    N=self.out_features, 
                    model=self.tuner_model, 
                    scaler=self.scaler, 
                    device='cpu'
                )
                current_block_x = predicted_bx
                current_block_y = predicted_by
            except Exception as e:
                print(f"Error predicting block size: {e}. Using default block sizes.")
        
        # Apply custom matrix multiplication
        out = CPUMatMulFunction.apply(x, weight, current_block_x, current_block_y)
        
        # Add bias
        return out + self.bias

# Standard PyTorch MLP model
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

# Custom MLP model with CPU custom layer
class CustomMLP(nn.Module):
    def __init__(self, tuner_model=None, scaler=None):
        super(CustomMLP, self).__init__()
        self.layers = nn.Sequential(
            CPUCustomLinear(INPUT_SIZE, HIDDEN_SIZE1, tuner_model, scaler),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE1, HIDDEN_SIZE2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE2, NUM_CLASSES)
        )

    def forward(self, x):
        x = x.view(-1, INPUT_SIZE)
        return self.layers(x)

def load_test_data():
    """Load MNIST test dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Create dict of DataLoaders for different batch sizes
    test_loaders = {}
    for batch_size in BATCH_SIZES:
        test_loaders[batch_size] = DataLoader(
            dataset=test_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
    
    return test_loaders

def initialize_models():
    """Initialize and load both PyTorch and Custom models"""
    
    # Load tuner model for block size prediction
    print("Loading tuner model...")
    model_path = os.path.join(single_layer_dir, "tuner_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Tuner model file {model_path} not found")
    
    tuner_model, scaler = load_model_and_scaler(model_path, input_dim=12)
    tuner_model.eval()
    
    # Initialize models
    pytorch_model = PyTorchMLP().to(device)
    custom_model = CustomMLP(tuner_model, scaler).to(device)
    
    # Try to load pre-trained weights if available
    try:
        model_path = "mlp_with_custom_layer.pt"
        if os.path.exists(model_path):
            print(f"Loading pre-trained model from {model_path}")
            # Need to handle the fact that keys will be different between models
            # Just initialize with random weights for benchmarking inference
            # custom_model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Could not load pre-trained model: {e}")
        print("Using randomly initialized models for benchmarking")
    
    return pytorch_model, custom_model, tuner_model, scaler

def benchmark_inference():
    """Benchmark inference time for both models across different batch sizes"""
    
    # Load test data for different batch sizes
    test_loaders = load_test_data()
    
    # Initialize models
    pytorch_model, custom_model, tuner_model, scaler = initialize_models()
    
    # Put models in eval mode
    pytorch_model.eval()
    custom_model.eval()
    
    # Results dictionary
    results = {
        'batch_size': [],
        'pytorch_time': [],
        'custom_time': [],
        'speedup': [],
        'block_x': [],
        'block_y': []
    }
    
    print("\n" + "="*60)
    print(f"{'Batch Size':<10} | {'PyTorch (ms)':<15} | {'Custom (ms)':<15} | {'Ratio':<8} | {'Block Sizes':<12}")
    print("-"*60)
    
    # Benchmark each batch size
    for batch_size in BATCH_SIZES:
        dataloader = test_loaders[batch_size]
        
        # Get one batch to benchmark
        data, _ = next(iter(dataloader))
        data = data.to(device)
        
        # Record block sizes predicted for this batch size
        block_x, block_y, _ = predict_best_block_size(
            M=batch_size,
            K=INPUT_SIZE,
            N=HIDDEN_SIZE1,
            model=tuner_model,
            scaler=scaler,
            device=device
        )
        
        # Warm-up runs
        for _ in range(10):
            with torch.no_grad():
                _ = pytorch_model(data)
                _ = custom_model(data)
        
        # Benchmark PyTorch model
        pytorch_times = []
        torch.cuda.synchronize() if device.type == 'cuda' else None
        for _ in range(NUM_RUNS):
            with torch.no_grad():
                start = time.time()
                _ = pytorch_model(data)
                end = time.time()
                pytorch_times.append((end - start) * 1000)  # ms
        
        # Benchmark Custom model
        custom_times = []
        torch.cuda.synchronize() if device.type == 'cuda' else None
        for _ in range(NUM_RUNS):
            with torch.no_grad():
                start = time.time()
                _ = custom_model(data)
                end = time.time()
                custom_times.append((end - start) * 1000)  # ms
        
        # Calculate averages
        avg_pytorch_time = np.mean(pytorch_times)
        avg_custom_time = np.mean(custom_times)
        ratio = avg_custom_time / avg_pytorch_time if avg_pytorch_time > 0 else float('inf')
        
        # Store results
        results['batch_size'].append(batch_size)
        results['pytorch_time'].append(avg_pytorch_time)
        results['custom_time'].append(avg_custom_time)
        results['speedup'].append(1/ratio if ratio > 0 else 0)
        results['block_x'].append(block_x)
        results['block_y'].append(block_y)
        
        print(f"{batch_size:<10} | {avg_pytorch_time:<15.4f} | {avg_custom_time:<15.4f} | "
              f"{ratio:<8.2f} | ({block_x}, {block_y})")
    
    print("="*60)
    return results

def plot_results(results):
    """Create visualizations from benchmark results"""
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Inference time comparison
    ax1.plot(results['batch_size'], results['pytorch_time'], 'o-', label='PyTorch (CPU)')
    ax1.plot(results['batch_size'], results['custom_time'], 's-', label='Custom with Predicted Blocks')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Inference Time (ms)')
    ax1.set_title('Inference Time Comparison: PyTorch vs Custom Implementation on CPU')
    ax1.grid(True, which='both', linestyle='--', alpha=0.6)
    ax1.legend()
    
    # Plot 2: Ratios and block sizes
    ax2.bar(range(len(results['batch_size'])), 
            [custom/pytorch for custom, pytorch in zip(results['custom_time'], results['pytorch_time'])], 
            label='Custom/PyTorch Ratio')
    ax2.set_xticks(range(len(results['batch_size'])))
    ax2.set_xticklabels([f"{bs}\n({bx},{by})" for bs, bx, by in 
                        zip(results['batch_size'], results['block_x'], results['block_y'])])
    ax2.set_xlabel('Batch Size (with Block Sizes)')
    ax2.set_ylabel('Time Ratio (Custom/PyTorch)')
    ax2.set_title('Performance Ratio with Predicted Block Sizes')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('inference_benchmark_results.png')
    print("Results plot saved as 'inference_benchmark_results.png'")
    
    # Save raw data to CSV
    import csv
    with open('inference_benchmark_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Batch Size', 'PyTorch Time (ms)', 'Custom Time (ms)', 
                        'Custom/PyTorch Ratio', 'Block X', 'Block Y'])
        for i in range(len(results['batch_size'])):
            writer.writerow([
                results['batch_size'][i],
                results['pytorch_time'][i],
                results['custom_time'][i],
                results['custom_time'][i] / results['pytorch_time'][i],
                results['block_x'][i],
                results['block_y'][i]
            ])
    print("Results saved to 'inference_benchmark_results.csv'")

if __name__ == "__main__":
    print("Starting inference benchmark: PyTorch vs Custom MLP on CPU")
    results = benchmark_inference()
    plot_results(results) 