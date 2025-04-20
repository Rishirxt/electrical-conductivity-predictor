import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

# Classical MLP Model
class ClassicalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.model(x)

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
csv_file = r"C:\Users\Rishi\Desktop\physics\expanded_materials.csv"  # UPDATED PATH
materials, X_train, y_train, X_min, X_max, y_min, y_max = load_material_data(csv_file)

# Model, optimizer, loss
model = ClassicalMLP()
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

# Print results
print("\nPredicted vs Actual Electrical Conductivity (MS/m):")
print(f"{'Material':<20}{'Actual':<20}{'Predicted':<20}")
for mat, actual, pred in zip(materials, y_true_denorm, y_pred_denorm):
    print(f"{mat:<20}{actual[0]:<20.2f}{pred[0]:<20.2f}")

# Save to CSV
df_results = pd.DataFrame({
    "Material": materials,
    "Actual Electrical Conductivity (MS/m)": y_true_denorm.flatten(),
    "Predicted Electrical Conductivity (MS/m)": y_pred_denorm.flatten()
})
df_results.to_csv(r"C:\Users\Rishi\Desktop\physics\classical_predicted_electrical_conductivity.csv", index=False)
print("\nResults saved to 'classical_predicted_electrical_conductivity.csv'")

# Plot comparison
plt.figure(figsize=(8, 6))
plt.scatter(y_true_denorm, y_pred_denorm, color='green', label='Predicted vs Actual', alpha=0.7)
plt.plot([min(y_true_denorm), max(y_true_denorm)], [min(y_true_denorm), max(y_true_denorm)], 'k--', lw=2, label='Perfect fit')
plt.xlabel("Actual Electrical Conductivity (MS/m)")
plt.ylabel("Predicted Electrical Conductivity (MS/m)")
plt.title("Classical MLP: Predicted vs Actual Electrical Conductivity")
plt.legend()
plt.grid(True)
plt.show()
