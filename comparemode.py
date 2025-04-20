import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
# Load predictions
df_hq = pd.read_csv(r"C:\Users\Rishi\Desktop\physics\improved_predicted_electrical_conductivity.csv")
df_cl = pd.read_csv(r"C:\Users\Rishi\Desktop\physics\classical_predicted_electrical_conductivity.csv")  # Updated path

# Combine predictions
df_compare = df_hq.copy()
df_compare["MLP Predicted (MS/m)"] = df_cl["Predicted Electrical Conductivity (MS/m)"]

# Save combined results
df_compare.to_csv(r"C:\Users\Rishi\Desktop\physics\comparison_results.csv", index=False)
print("\n--- Combined Results ---")
print(df_compare)

# Metrics
actual = df_compare["Actual Electrical Conductivity (MS/m)"].values
pred_hq = df_compare["Predicted Electrical Conductivity (MS/m)"].values
pred_cl = df_compare["MLP Predicted (MS/m)"].values

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
plt.scatter(actual, pred_hq, color='blue', label=f'HQCNN\nMSE: {mse_hq:.4f}\nR²: {r2_hq:.4f}', alpha=0.7)
plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', label="Perfect Fit")
plt.xlabel("Actual Electrical Conductivity (MS/m)")
plt.ylabel("Predicted Electrical Conductivity (MS/m)")
plt.title("HQCNN Predictions")
plt.legend()
plt.grid(True)

# Classical MLP plot
plt.subplot(1, 2, 2)
plt.scatter(actual, pred_cl, color='green', label=f'Classical MLP\nMSE: {mse_cl:.4f}\nR²: {r2_cl:.4f}', alpha=0.7)
plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', label="Perfect Fit")
plt.xlabel("Actual Electrical Conductivity (MS/m)")
plt.ylabel("Predicted Electrical Conductivity (MS/m)")
plt.title("Classical MLP Predictions")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
errors_hq = df_compare["Actual Electrical Conductivity (MS/m)"] - df_compare["Predicted Electrical Conductivity (MS/m)"]
errors_cl = df_compare["Actual Electrical Conductivity (MS/m)"] - df_compare["MLP Predicted (MS/m)"]

# Plot distribution
plt.figure(figsize=(12, 6))

# HQCNN Errors
sns.histplot(errors_hq, kde=True, color='blue', label='HQCNN Errors', stat="density", bins=20, alpha=0.6)

# Classical MLP Errors
sns.histplot(errors_cl, kde=True, color='green', label='Classical MLP Errors', stat="density", bins=20, alpha=0.6)

plt.xlabel("Prediction Error (MS/m)")
plt.ylabel("Density")
plt.title("Distribution of Prediction Errors")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
