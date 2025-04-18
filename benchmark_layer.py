
import torch
import time
import csv
import itertools
from torch.utils.data import DataLoader, TensorDataset
from mlp_with_custom_layer import MLP

# Constants
K, N = 784, 256
M_values = [1, 8, 16, 32, 64, 128, 256]
BLOCK_LIMIT = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate synthetic MNIST-like data (28x28 grayscale images, labels 0–9)
def generate_fake_mnist_data(num_samples):
    images = torch.rand(num_samples, 1, 28, 28)  # Shape: (B, C, H, W)
    labels = torch.randint(0, 10, (num_samples,))  # Shape: (B,)
    return images, labels

results = []

for M in M_values:
    # Create fake dataset and loader
    images, labels = generate_fake_mnist_data(M)
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=M, shuffle=False)

    # Get a single batch
    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    for block_x, block_y in itertools.product(range(1, 33), repeat=2):
        if block_x * block_y >= BLOCK_LIMIT:
            continue

        # Reinitialize model every time to avoid warm starts
        model = MLP(K, N, block_x, block_y).to(device)
        model.train()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        results.append([M, K, N, block_x, block_y, elapsed_ms])
        print(f"M={M}, block=({block_x},{block_y}) -> {elapsed_ms:.2f} ms")

# Save results
with open("benchmark_results_light.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["M", "K", "N", "block_x", "block_y", "runtime_ms"])
    writer.writerows(results)

print("Done! CSV saved as benchmark_results_light.csv")
