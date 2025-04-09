import torch
from torch import nn
from torch.autograd import Function
import matmul_cuda  # Compiled with CUDA kernel

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
    def __init__(self, in_features, out_features, block_x=16, block_y=16):
        super(CustomLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.block_x = block_x
        self.block_y = block_y

    def forward(self, x):
        out = MatMulFunction.apply(x, self.weight, self.block_x, self.block_y)
        return out + self.bias