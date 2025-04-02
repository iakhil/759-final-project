// cuda_mlp_mnist.cu

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <cuda_runtime.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define BATCH_SIZE 64
#define EPOCHS 5
#define LEARNING_RATE 0.01f

__device__ float relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ float relu_deriv(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

__device__ float softmax_denom(const float* input, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++)
        if (input[i] > max_val) max_val = input[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
        sum += expf(input[i] - max_val);
    return sum;
}

__global__ void forward_pass(
    float* input, float* W1, float* W2,
    float* hidden, float* output, int batch_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    float local_hidden[HIDDEN_SIZE];
    float local_output[OUTPUT_SIZE];

    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < INPUT_SIZE; ++k) {
            sum += input[i * INPUT_SIZE + k] * W1[k * HIDDEN_SIZE + j];
        }
        local_hidden[j] = relu(sum);
        hidden[i * HIDDEN_SIZE + j] = local_hidden[j];
    }

    float denom = 0.0f;
    for (int j = 0; j < OUTPUT_SIZE; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < HIDDEN_SIZE; ++k) {
            sum += local_hidden[k] * W2[k * OUTPUT_SIZE + j];
        }
        local_output[j] = expf(sum);
        denom += local_output[j];
    }

    for (int j = 0; j < OUTPUT_SIZE; ++j) {
        output[i * OUTPUT_SIZE + j] = local_output[j] / denom;
    }
}

__global__ void backward_pass(
    float* input, float* hidden, float* output, int* labels,
    float* W1, float* W2, int batch_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    float d_output[OUTPUT_SIZE];
    for (int j = 0; j < OUTPUT_SIZE; ++j) {
        int target = (labels[i] == j) ? 1 : 0;
        d_output[j] = output[i * OUTPUT_SIZE + j] - target;
    }

    float d_hidden[HIDDEN_SIZE] = {0};
    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        float grad = 0.0f;
        for (int k = 0; k < OUTPUT_SIZE; ++k) {
            grad += d_output[k] * W2[j * OUTPUT_SIZE + k];
        }
        d_hidden[j] = grad * relu_deriv(hidden[i * HIDDEN_SIZE + j]);
    }

    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        for (int k = 0; k < OUTPUT_SIZE; ++k) {
            atomicAdd(&W2[j * OUTPUT_SIZE + k], -LEARNING_RATE * hidden[i * HIDDEN_SIZE + j] * d_output[k]);
        }
    }

    for (int j = 0; j < INPUT_SIZE; ++j) {
        for (int k = 0; k < HIDDEN_SIZE; ++k) {
            atomicAdd(&W1[j * HIDDEN_SIZE + k], -LEARNING_RATE * input[i * INPUT_SIZE + j] * d_hidden[k]);
        }
    }
}

void initialize_weights(std::vector<float>& W, int in_size, int out_size) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    for (int i = 0; i < in_size * out_size; ++i)
        W[i] = dist(gen);
}

// Dummy loader (you'd replace this with actual MNIST reading)
void load_dummy_data(std::vector<float>& X, std::vector<int>& Y, int count) {
    X.resize(count * INPUT_SIZE);
    Y.resize(count);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> pix(0.0f, 1.0f);
    std::uniform_int_distribution<int> cls(0, 9);
    for (int i = 0; i < count * INPUT_SIZE; ++i)
        X[i] = pix(gen);
    for (int i = 0; i < count; ++i)
        Y[i] = cls(gen);
}

int main() {
    std::vector<float> h_X, h_W1(INPUT_SIZE * HIDDEN_SIZE), h_W2(HIDDEN_SIZE * OUTPUT_SIZE);
    std::vector<int> h_Y;
    load_dummy_data(h_X, h_Y, BATCH_SIZE);
    initialize_weights(h_W1, INPUT_SIZE, HIDDEN_SIZE);
    initialize_weights(h_W2, HIDDEN_SIZE, OUTPUT_SIZE);

    float *d_X, *d_W1, *d_W2, *d_hidden, *d_output;
    int* d_Y;

    cudaMalloc(&d_X, sizeof(float) * BATCH_SIZE * INPUT_SIZE);
    cudaMalloc(&d_W1, sizeof(float) * INPUT_SIZE * HIDDEN_SIZE);
    cudaMalloc(&d_W2, sizeof(float) * HIDDEN_SIZE * OUTPUT_SIZE);
    cudaMalloc(&d_hidden, sizeof(float) * BATCH_SIZE * HIDDEN_SIZE);
    cudaMalloc(&d_output, sizeof(float) * BATCH_SIZE * OUTPUT_SIZE);
    cudaMalloc(&d_Y, sizeof(int) * BATCH_SIZE);

    cudaMemcpy(d_X, h_X.data(), sizeof(float) * BATCH_SIZE * INPUT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, h_W1.data(), sizeof(float) * INPUT_SIZE * HIDDEN_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2.data(), sizeof(float) * HIDDEN_SIZE * OUTPUT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y.data(), sizeof(int) * BATCH_SIZE, cudaMemcpyHostToDevice);

    int threads = 64;
    int blocks = (BATCH_SIZE + threads - 1) / threads;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        forward_pass<<<blocks, threads>>>(d_X, d_W1, d_W2, d_hidden, d_output, BATCH_SIZE);
        backward_pass<<<blocks, threads>>>(d_X, d_hidden, d_output, d_Y, d_W1, d_W2, BATCH_SIZE);
        cudaDeviceSynchronize();
        std::cout << "Epoch " << epoch + 1 << " done.\n";
    }

    cudaFree(d_X);
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_Y);

    return 0;
}
