import torch.nn as nn
import torch

class KernelTuner(nn.Module):
    def __init__(self):
        super(KernelTuner, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Outputs: [block_size_x, block_size_y]
        )

    def forward(self, x):
        return self.fc(x)


# Directory: training/train_tuner.py
import torch
from models.tuner_model import KernelTuner

# Dummy dataset: [input_size, hidden_size, matrix_size] -> [block_x, block_y]
data = torch.tensor([[784, 256, 64], [784, 512, 128]], dtype=torch.float32)
labels = torch.tensor([[16, 16], [32, 32]], dtype=torch.float32)

model = KernelTuner()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

for epoch in range(100):
    output = model(data)
    loss = loss_fn(output, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), "tuner_model.pt")
