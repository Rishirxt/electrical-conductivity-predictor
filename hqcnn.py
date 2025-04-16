import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

# Set up quantum device
dev = qml.device("default.qubit", wires=2)

# Define quantum circuit
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=[0, 1])
    qml.StronglyEntanglingLayers(weights, wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

# QNode
theta_shape = (3, 2, 3)
qnode = qml.QNode(quantum_circuit, dev, interface="torch")

# Quantum Layer
class QuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_weights = nn.Parameter(0.01 * torch.randn(theta_shape))

    def forward(self, x):
        return torch.stack([qnode(x_i, self.q_weights) for x_i in x]).to(torch.float32)

# Hybrid Neural Network
class HybridQNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_layer = QuantumLayer()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        x = self.q_layer(x)
        x = x.view(-1, 1)
        return self.fc(x)

# Load and normalize dataset
def load_material_data(csv_file):
    df = pd.read_csv(csv_file)
    materials = df["Material"].values
    X = torch.tensor(df[["Feature1", "Feature2"]].values, dtype=torch.float32)
    y = torch.tensor(df["Electrical Conductivity (MS/m)"].values, dtype=torch.float32).view(-1, 1)

    # Normalize input features
    X_min, X_max = X.min(0)[0], X.max(0)[0]
    X_norm = ((X - X_min) / (X_max - X_min)).float()

    # Normalize targets
    y_min, y_max = y.min(), y.max()
    y_norm = (y - y_min) / (y_max - y_min)

    return materials, X_norm, y_norm, X_min, X_max, y_min, y_max

# Denormalize
def denormalize(tensor, min_val, max_val):
    return tensor * (max_val - min_val) + min_val

# Load data
csv_file = "materials.csv"
materials, X_train, y_train, X_min, X_max, y_min, y_max = load_material_data(csv_file)

# Model, optimizer, loss
model = HybridQNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

# Prediction
y_pred = model(X_train).detach()
y_pred_denorm = denormalize(y_pred, y_min, y_max).numpy()
y_true_denorm = denormalize(y_train, y_min, y_max).numpy()

# Print results in table format
print("\nPredicted vs Actual Electrical Conductivity (MS/m):")
print(f"{'Material':<15}{'Actual Value':<20}{'Predicted Value':<20}")
for mat, actual, predicted in zip(materials, y_true_denorm, y_pred_denorm):
    print(f"{mat:<15}{actual[0]:<20.2f}{predicted[0]:<20.2f}")

# Save to CSV
df_results = pd.DataFrame({
    "Material": materials,
    "Actual Electrical Conductivity (MS/m)": y_true_denorm.flatten(),
    "Predicted Electrical Conductivity (MS/m)": y_pred_denorm.flatten()
})
df_results.to_csv("predicted_electrical_conductivity.csv", index=False)
print("\nResults saved to 'predicted_electrical_conductivity.csv'")

# Plotting the comparison between actual and predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_true_denorm, y_pred_denorm, color='blue', label='Predicted vs Actual', alpha=0.7)
plt.plot([min(y_true_denorm), max(y_true_denorm)], [min(y_true_denorm), max(y_true_denorm)], 'k--', lw=2, label='Perfect fit')
plt.xlabel("Actual Electrical Conductivity (MS/m)")
plt.ylabel("Predicted Electrical Conductivity (MS/m)")
plt.title("Predicted vs Actual Electrical Conductivity")
plt.legend()
plt.grid(True)
plt.show()
