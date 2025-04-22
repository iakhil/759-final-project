#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                              int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // index in M dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x; // index in N dimension

    if (row < M && col < N) {
        float value = 0.0f;
        for (int i = 0; i < K; ++i) {
            value += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = value;
    }
}

// Kernel launcher
void launch_matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                   int M, int K, int N, int block_x, int block_y) {
    const dim3 threads(block_x, block_y);
    const dim3 blocks((N + threads.x - 1) / threads.x,
                      (M + threads.y - 1) / threads.y);

    matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );
}

// PyTorch binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_matmul", &launch_matmul, "Custom CUDA MatMul kernel");
}

