import torch
import time
import csv
import os

from matmul_cuda import launch_matmul

K, N = 784, 256  # fixed
M_values = [1, 8, 16, 32, 64, 128, 256]
MAX_THREADS = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_file = "matmul_results.csv"
with open(output_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['M', 'K', 'N', 'block_x', 'block_y', 'time_taken_ms'])

    for M in M_values:
        A = torch.randn(M, K, device=device, dtype=torch.float32)
        B = torch.randn(K, N, device=device, dtype=torch.float32)
        C = torch.zeros(M, N, device=device, dtype=torch.float32)

        for block_x in range(1, 33):
            for block_y in range(1, 33):
                if block_x * block_y > MAX_THREADS:
                    continue

                try:
                    # Warm-up
                    launch_matmul(A, B, C, M, K, N, block_x, block_y)
                    torch.cuda.synchronize()

                    # Timing
                    start = time.time()
                    launch_matmul(A, B, C, M, K, N, block_x, block_y)
                    torch.cuda.synchronize()
                    end = time.time()
                    
                    elapsed = (end - start) * 1000  # ms
                    writer.writerow([M, K, N, block_x, block_y, elapsed])
                except Exception as e:
                    print(f"Failed M={M}, Bx={block_x}, By={block_y}: {e}")

