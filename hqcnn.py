import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# HQCNN Model (example)
class HQCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),  # Increased neurons
            nn.BatchNorm1d(64),  # Batch Normalization
            nn.LeakyReLU(0.01),  # LeakyReLU instead of ReLU
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# Load and normalize dataset
def load_material_data(csv_file):
    df = pd.read_csv(csv_file)
    materials = df["Material"].values
    features = df[["Feature1", "Feature2"]].values
    targets = df["Electrical Conductivity (MS/m)"].values.reshape(-1, 1)

    # Standardization (mean=0, std=1)
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = X_scaler.fit_transform(features)
    y_scaled = y_scaler.fit_transform(targets)

    X = torch.tensor(X_scaled, dtype=torch.float32)
    y = torch.tensor(y_scaled, dtype=torch.float32)

    return materials, X, y, X_scaler, y_scaler

# Inverse transform
def denormalize(tensor, scaler):
    return scaler.inverse_transform(tensor.detach().numpy())

# Load data
csv_file = "expanded_materials.csv"
materials, X_train, y_train, X_scaler, y_scaler = load_material_data(csv_file)

# Model, optimizer, loss
model = HQCNN()
optimizer = optim.AdamW(model.parameters(), lr=0.001)  # AdamW instead of Adam
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)  # Learning rate decay
loss_fn = nn.SmoothL1Loss()  # Huber loss

# Training loop
epochs = 1000  # More epochs
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

# Prediction
model.eval()
with torch.no_grad():
    y_pred = model(X_train)

y_pred_denorm = denormalize(y_pred, y_scaler)
y_true_denorm = denormalize(y_train, y_scaler)

# Print results
print("\nPredicted Electrical Conductivity:")
for mat, cond in zip(materials, y_pred_denorm):
    print(f"{mat}: Electrical Conductivity = {cond[0]:.2f} MS/m")

# Save to CSV
df_results = pd.DataFrame({
    "Material": materials,
    "Actual Electrical Conductivity (MS/m)": y_true_denorm.flatten(),
    "Predicted Electrical Conductivity (MS/m)": y_pred_denorm.flatten()
})
df_results.to_csv("improved_predicted_electrical_conductivity.csv", index=False)
print("\nResults saved to 'improved_predicted_electrical_conductivity.csv'")

# Plot comparison
plt.figure(figsize=(8, 6))
plt.scatter(y_true_denorm, y_pred_denorm, color='purple', label='Predicted vs Actual', alpha=0.7)
plt.plot([min(y_true_denorm), max(y_true_denorm)], [min(y_true_denorm), max(y_true_denorm)], 'k--', lw=2, label='Perfect fit')
plt.xlabel("Actual Electrical Conductivity (MS/m)")
plt.ylabel("Predicted Electrical Conductivity (MS/m)")
plt.title("Improved HQCNN: Predicted vs Actual Electrical Conductivity")
plt.legend()
plt.grid(True)
plt.show()
