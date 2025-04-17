#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>  // for CUDA stream handling

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Host launcher
void matmul_launcher(torch::Tensor A, torch::Tensor B, torch::Tensor C, int block_x, int block_y) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.device().is_cuda(), "C must be a CUDA tensor");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    dim3 threadsPerBlock(block_x, block_y);
    dim3 numBlocks((N + block_x - 1) / block_x, (M + block_y - 1) / block_y);

    matmul_kernel<<<numBlocks, threadsPerBlock>>>(A_ptr, B_ptr, C_ptr, M, K, N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}

// Add this block at the end of the file
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_launcher", &matmul_launcher, "Matrix Multiply with configurable blocks");
}

