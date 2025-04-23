import torch
import os
from tuner_model import load_model_and_scaler, predict_best_block_size

# Matrix dimensions for the first layer
M = 64  # batch size
K = 784  # input size (28x28)
N = 256  # hidden layer size

print(f"Predicting optimal block sizes for matrix dimensions: M={M}, K={K}, N={N}")

# Load tuner model and scaler
try:
    model_path = "tuner_model.pt"
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found. Checking in alternative locations...")
        
        # Try to find the model in subdirectories
        for root, _, files in os.walk("."):
            if "tuner_model.pt" in files:
                model_path = os.path.join(root, "tuner_model.pt")
                print(f"Found model at {model_path}")
                break
        else:
            raise FileNotFoundError(f"Could not find tuner_model.pt in any subdirectory")
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and scaler
    model, scaler = load_model_and_scaler(model_path, input_dim=12)
    model = model.to(device)
    print("Successfully loaded tuner model and scaler")
    
    # Predict best block size
    best_bx, best_by, predicted_runtime = predict_best_block_size(
        M, K, N, model, scaler, device=device
    )
    print(f"Predicted optimal block sizes: block_x={best_bx}, block_y={best_by}")
    print(f"Predicted runtime: {predicted_runtime:.4f} ms")
    
except Exception as e:
    print(f"Error in prediction: {e}")
    # Default block sizes if prediction fails
    best_bx, best_by = 16, 16
    print(f"Using default block sizes: block_x={best_bx}, block_y={best_by}") 