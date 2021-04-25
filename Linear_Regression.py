import numpy as np
import torch
import torch.nn as nn


# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')

inputs = torch.from_numpy(inputs)

targets = torch.from_numpy(targets)
print(inputs)
print(targets)


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(in_features=3, out_features=2)

    def forward(self, x):
        y = self.linear(x)
        return y


model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

loss_list = []

for epoch in range(100):
    # Calculate predictions
    predictions = model(inputs)

    # Calculate loss
    loss = criterion(predictions, targets)

    loss_list.append(loss.item())

    # Zeroing gradients
    optimizer.zero_grad()

    # Compute gradients
    loss.backward()

    # Update the weights
    optimizer.step()

    print(f' epoch {epoch}, loss {loss.item()}')

with torch.no_grad():
    outputs = model(inputs)

print(outputs.numpy())

