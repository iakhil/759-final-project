import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class KernelTuner(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=128):
        super(KernelTuner, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Predicting runtime
        )

    def forward(self, x):
        return self.model(x)

def train_and_save_model(csv_path='benchmark_results_light.csv', model_path='tuner_model.pt'):
    df = pd.read_csv(csv_path)
    # Feature engineering
    eps = 1e-8
    df['M_block_x'] = df['M'] * df['block_x']
    df['M_block_y'] = df['M'] * df['block_y']
    df['block_x_block_y'] = df['block_x'] * df['block_y']
    df['M_div_block_x'] = df['M'] / (df['block_x'] + eps)
    df['M_div_block_y'] = df['M'] / (df['block_y'] + eps)
    df['block_x_div_block_y'] = df['block_x'] / (df['block_y'] + eps)
    df['block_y_div_block_x'] = df['block_y'] / (df['block_x'] + eps)

    features = ['M', 'K', 'N', 'block_x', 'block_y',
                'M_block_x', 'M_block_y', 'block_x_block_y',
                'M_div_block_x', 'M_div_block_y',
                'block_x_div_block_y', 'block_y_div_block_x']
    X = df[features].values
    y = df['runtime_ms'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KernelTuner(input_dim=len(features)).to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(200):  # More epochs
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    # Testing/Validation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        abs_errors = torch.abs(test_outputs - y_test)
        mae = abs_errors.mean().item()
        # Accuracy within ±0.1 ms
        accuracy = (abs_errors <= 0.1).sum().item() / len(y_test)

        print(f'\nTest Results:')
        print(f'Test Loss: {test_loss.item():.4f}')
        print(f'Mean Absolute Error: {mae:.4f} ms')
        print(f'Accuracy (±0.1 ms): {accuracy*100:.2f}%')
        print('\nSample predictions:')
        for i in range(min(5, len(X_test))):
            input_values = X_test[i].cpu().numpy()
            predicted_runtime = test_outputs[i].item()
            actual_runtime = y_test[i].item()
            print(f'Input: {input_values} => Predicted Runtime: {predicted_runtime:.4f} ms, Actual Runtime: {actual_runtime:.4f} ms')

    # Save only the state_dict and scaler parameters
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_
    }, model_path)
    print(f'\nTuner model and scaler saved to {model_path}')

def load_model_and_scaler(model_path='tuner_model.pt', input_dim=12):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = KernelTuner(input_dim=input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    scaler = StandardScaler()
    scaler.mean_ = checkpoint['scaler_mean']
    scaler.scale_ = checkpoint['scaler_scale']
    return model, scaler

def predict_best_block_size(M, K, N, model, scaler, block_sizes=[4, 8, 16, 32], device="cpu"):
    candidates = []
    eps = 1e-8
    for block_x in block_sizes:
        for block_y in block_sizes:
            features = np.array([[M, K, N, block_x, block_y,
                                  M*block_x, M*block_y, block_x*block_y,
                                  M/(block_x+eps), M/(block_y+eps),
                                  block_x/(block_y+eps), block_y/(block_x+eps)]])
            features_scaled = scaler.transform(features)
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
            with torch.no_grad():
                predicted_runtime = model(features_tensor).item()
            candidates.append((predicted_runtime, block_x, block_y))
    best_runtime, best_block_x, best_block_y = min(candidates)
    return best_block_x, best_block_y, best_runtime

if __name__ == '__main__':
    train_and_save_model()

    # Use constant K and N from your dataset
    K = 784
    N = 256
    M = 128  # Change as needed

    model, scaler = load_model_and_scaler('tuner_model.pt', input_dim=12)
    best_bx, best_by, best_runtime = predict_best_block_size(M, K, N, model, scaler)
    print(f"\nFor M={M}, K={K}, N={N}, the best block sizes are: block_x={best_bx}, block_y={best_by} (predicted runtime: {best_runtime:.4f} ms)")
