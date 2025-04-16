import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Load predictions
df_hq = pd.read_csv("predicted_electrical_conductivity.csv")
df_cl = pd.read_csv("classical_predicted_electrical_conductivity.csv")

# Combine predictions
df_compare = df_hq.copy()
df_compare["MLP Predicted (MS/m)"] = df_cl["Predicted Electrical Conductivity (MS/m)"]

# Save comparison
df_compare.to_csv("comparison_results.csv", index=False)
print("\n--- Combined Results ---")
print(df_compare)

# Metrics
actual = df_compare["Actual Electrical Conductivity (MS/m)"]
pred_hq = df_compare["Predicted Electrical Conductivity (MS/m)"]
pred_cl = df_compare["MLP Predicted (MS/m)"]

mse_hq = mean_squared_error(actual, pred_hq)
mse_cl = mean_squared_error(actual, pred_cl)

r2_hq = r2_score(actual, pred_hq)
r2_cl = r2_score(actual, pred_cl)

print(f"\nHQCNN MSE: {mse_hq:.4f}, R²: {r2_hq:.4f}")
print(f"Classical MLP MSE: {mse_cl:.4f}, R²: {r2_cl:.4f}")

# Plotting
plt.figure(figsize=(12, 5))

# HQCNN plot
plt.subplot(1, 2, 1)
plt.scatter(actual, pred_hq, color='blue', label='HQCNN', alpha=0.7)
plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("HQCNN Predictions")
plt.grid(True)

# Classical plot
plt.subplot(1, 2, 2)
plt.scatter(actual, pred_cl, color='green', label='Classical MLP', alpha=0.7)
plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Classical MLP Predictions")
plt.grid(True)

plt.tight_layout()
plt.show()
