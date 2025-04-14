import torch
import time
from itertools import product
from torch.utils.cpp_extension import load

matmul = load(
    name="matmul_cuda",
    sources=["matmul_kernel.cu"], 
    verbose=True
)

M, K, N = 1024, 1024, 1024  
A = torch.randn(M, K, device='cuda')
B = torch.randn(K, N, device='cuda')
C = torch.zeros(M, N, device='cuda')

configs = [(bx, by) for bx, by in product([8, 16, 32], [8, 16, 32])]

results = []

for block_x, block_y in configs:
    torch.cuda.synchronize()
    start = time.time()

    matmul.matmul_launcher(A, B, C, block_x, block_y)

    torch.cuda.synchronize()
    end = time.time()

    elapsed = (end - start) * 1000  # milliseconds
    results.append(((block_x, block_y), elapsed))
    print(f"Block ({block_x}, {block_y}) => {elapsed:.3f} ms")

results.sort(key=lambda x: x[1])
print("\nTop 5 Fastest Configurations:")
for (bx, by), t in results[:5]:
    print(f"Block ({bx}, {by}): {t:.3f} ms")
