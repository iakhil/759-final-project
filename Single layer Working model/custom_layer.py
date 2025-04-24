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
    def __init__(self, in_features, out_features, tuner_model=None, scaler=None, block_x=16, block_y=16):
        super(CustomLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.tuner_model = tuner_model
        self.scaler = scaler
        self.block_x = block_x # Default block size x
        self.block_y = block_y # Default block size y
        
        # Save dimensions for prediction
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        # Validate input dimensions
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor, got {x.dim()}D tensor")
        if x.size(1) != self.in_features:
            raise ValueError(f"Expected input feature dimension {self.in_features}, got {x.size(1)}")
            
        x = x.contiguous()  # Ensure x is contiguous
        weight = self.weight.contiguous() # Ensure weight is contiguous
        
        batch_size = x.size(0) # M dimension for matmul and tuner input
        
        # Determine block sizes for CUDA kernel
        current_block_x = self.block_x
        current_block_y = self.block_y

        if self.tuner_model is not None and self.scaler is not None:
            # If tuner model and scaler are provided, predict the best block size
            try:
                # Import locally to avoid potential circular dependency issues
                from tuner_model import predict_best_block_size 
                
                # Predict best block size using the tuner model
                # Passes M, K, N correctly to the prediction function
                predicted_bx, predicted_by, _ = predict_best_block_size(
                    M=batch_size, 
                    K=self.in_features, 
                    N=self.out_features, 
                    model=self.tuner_model, 
                    scaler=self.scaler, 
                    device=x.device # Ensure prediction runs on the same device
                )
                current_block_x = predicted_bx
                current_block_y = predicted_by
                
            except ImportError:
                 print("Warning: tuner_model.py not found or predict_best_block_size not available. Using default block sizes.")
            except Exception as e:
                # Fallback to default block sizes if prediction fails
                print(f"Error predicting block size: {e}. Using default block sizes ({self.block_x}, {self.block_y}).")
        
        # Apply the custom matrix multiplication function using determined block sizes
        out = MatMulFunction.apply(x, weight, current_block_x, current_block_y)
        
        # Add bias
        return out + self.bias