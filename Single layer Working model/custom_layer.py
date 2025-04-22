import torch
from torch import nn
from torch.autograd import Function
import matmul_cuda  # Compiled with CUDA kernel

class MatMulFunction(Function):
    @staticmethod
    def forward(ctx, A, B, block_x, block_y):
        assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only float32 supported"
        assert A.is_contiguous() and B.is_contiguous(), "Inputs must be contiguous"

        ctx.save_for_backward(A, B)
        ctx.block_x = block_x
        ctx.block_y = block_y

        M, K = A.shape
        K_B, N = B.shape
        assert K == K_B, f"Shape mismatch: A is {A.shape}, B is {B.shape}"

        C = torch.zeros((M, N), device=A.device, dtype=A.dtype)

        matmul_cuda.matmul_launcher(A, B, C, block_x, block_y)
        return C

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = grad_output @ B.T
        grad_B = A.T @ grad_output
        return grad_A, grad_B, None, None  # block_x and block_y don't require gradients

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, tuner_model=None, default_block_x=16, default_block_y=16):
        super(CustomLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.tuner_model = tuner_model
        self.default_block_x = default_block_x
        self.default_block_y = default_block_y
        
        # Fixed properties for the tuner
        self.in_features = in_features
        self.out_features = out_features

    def get_block_sizes(self, batch_size):
        if self.tuner_model is not None:
            # Create input tensor: [batch_size, in_features, out_features]
            tuner_input = torch.tensor([batch_size, self.in_features, self.out_features], 
                                     dtype=torch.float32, device=self.weight.device).unsqueeze(0)
            
            # Get predicted block sizes
            with torch.no_grad():
                block_sizes = self.tuner_model(tuner_input).squeeze()
            
            # Convert to integers and ensure they're positive
            block_x = max(1, int(block_sizes[0].item()))
            block_y = max(1, int(block_sizes[1].item()))
            return block_x, block_y
        return self.default_block_x, self.default_block_y

    def forward(self, x):
        x = x.contiguous()  # Make sure x is contiguous
        weight = self.weight.contiguous()
        
        # Get batch size (M dimension)
        batch_size = x.size(0)
        
        # Get optimal block sizes for this batch
        block_x, block_y = self.get_block_sizes(batch_size)
        
        out = MatMulFunction.apply(x, weight, block_x, block_y)
        return out + self.bias