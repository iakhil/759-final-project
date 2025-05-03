import torch
from torch.autograd import Function

class CPUMatMulFunction(Function):
    @staticmethod
    def forward(ctx, A, B, block_x, block_y):
        assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only float32 supported"
        assert A.is_contiguous() and B.is_contiguous(), "Inputs must be contiguous"

        ctx.save_for_backward(A, B)
        
        M, K = A.shape
        K_B, N = B.shape
        assert K == K_B, f"Shape mismatch: A is {A.shape}, B is {B.shape}"

        # Initialize output tensor
        C = torch.zeros((M, N), device=A.device, dtype=A.dtype)
        
        # Simple blocked matrix multiplication
        for i in range(0, M, block_x):
            for j in range(0, N, block_y):
                i_end = min(i + block_x, M)
                j_end = min(j + block_y, N)
                
                # Block matrix multiplication - use PyTorch's native matmul
                # This is a straightforward implementation - in a real scenario
                # we would optimize with vectorized operations or NumPy
                C[i:i_end, j:j_end] = A[i:i_end, :] @ B[:, j:j_end]
        
        return C

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = grad_output @ B.T
        grad_B = A.T @ grad_output
        return grad_A, grad_B, None, None  # block_x and block_y don't require gradients 