import torch
from torch import nn
from torch.autograd import Function
import matmul_cuda  # Your custom CUDA kernel
from tuner_model_copy import predict_best_block_size  # Make sure this is imported

class MatMulFunction(Function):
    @staticmethod
    def forward(ctx, A, B, block_x, block_y):
        ctx.save_for_backward(A, B)
        ctx.block_x = block_x
        ctx.block_y = block_y

        M, K = A.shape
        N = B.shape[1]
        C = torch.zeros((M, N), device=A.device, dtype=A.dtype)

        matmul_cuda.matmul_launcher(A, B, C, block_x, block_y)
        return C

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = grad_output @ B.T
        grad_B = A.T @ grad_output
        return grad_A, grad_B, None, None

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, tuner_model, scaler):
        super(CustomLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.tuner_model = tuner_model
        self.scaler = scaler
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        # Get matrix dimensions
        M = x.shape[0]
        K = self.in_features
        N = self.out_features

        # Use the tuner to get best block sizes for this (M, K, N)
        block_x, block_y, _ = predict_best_block_size(M, K, N, model=self.tuner_model, scaler=self.scaler)
        block_x = int(block_x)
        block_y = int(block_y)

        # Call the CUDA kernel with tuned block sizes
        out = MatMulFunction.apply(x, self.weight, block_x, block_y)
        return out + self.bias
