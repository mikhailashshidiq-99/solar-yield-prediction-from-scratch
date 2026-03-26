"""
02_nonlinear_regression.py
==========================
Compares the old Linear Regression vs the new Non-Linear Regression
(with cosine SZA transform + polynomial features).

Run from the project root:
    venv\\Scripts\\python.exe notebooks/02_nonlinear_regression.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from src.data_loader import get_nrel_data
from src.preprocessing import Z_Standardization, bias_column, time_series_split, preprocess_nonlinear
from src.model import CustomLinearRegression


# ============================================================
#  HELPER: Compute R² Score
# ============================================================
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


# ============================================================
#  LOAD DATA
# ============================================================
print("=" * 60)
print("  SOLAR YIELD PREDICTION — LINEAR vs NON-LINEAR COMPARISON")
print("=" * 60)

df = get_nrel_data(source="csv", file_path="data/phoenix_2024.csv")
print(f"\nDataset shape: {df.shape}")
print(f"Features: {list(df.columns)}")


# ============================================================
#  PIPELINE 1: OLD LINEAR MODEL (Baseline)
# ============================================================
print("\n" + "=" * 60)
print("  PIPELINE 1: LINEAR REGRESSION (Baseline)")
print("=" * 60)

X_linear, y_linear, mu_lin, sigma_lin = Z_Standardization(df)
X_linear = bias_column(X_linear)

X_train_lin, X_test_lin, y_train_lin, y_test_lin = time_series_split(X_linear, y_linear)

print(f"Training samples: {X_train_lin.shape[0]}")
print(f"Test samples:     {X_test_lin.shape[0]}")
print(f"Features (with bias): {X_train_lin.shape[1]}")

model_linear = CustomLinearRegression(learning_rate=0.01, epochs=1000)
model_linear.fit(X_train_lin, y_train_lin)

y_pred_lin = model_linear.predict(X_test_lin)
r2_linear = r2_score(y_test_lin, y_pred_lin)
print(f"\n>>> LINEAR MODEL R² = {r2_linear:.4f} ({r2_linear*100:.2f}%)")


# ============================================================
#  PIPELINE 2: NON-LINEAR MODEL (Cosine SZA + Polynomial)
# ============================================================
print("\n" + "=" * 60)
print("  PIPELINE 2: NON-LINEAR REGRESSION (cos(SZA) + Poly Degree 2)")
print("=" * 60)

X_nonlin, y_nonlin, mu_nl, sigma_nl = preprocess_nonlinear(df, poly_degree=2)
X_nonlin = bias_column(X_nonlin)

X_train_nl, X_test_nl, y_train_nl, y_test_nl = time_series_split(X_nonlin, y_nonlin)

print(f"Training samples: {X_train_nl.shape[0]}")
print(f"Test samples:     {X_test_nl.shape[0]}")
print(f"Features (with bias): {X_train_nl.shape[1]}")

# Lower learning rate for stability with many polynomial features
model_nonlinear = CustomLinearRegression(learning_rate=0.001, epochs=2000)
model_nonlinear.fit(X_train_nl, y_train_nl)

y_pred_nl = model_nonlinear.predict(X_test_nl)
r2_nonlinear = r2_score(y_test_nl, y_pred_nl)
print(f"\n>>> NON-LINEAR MODEL R² = {r2_nonlinear:.4f} ({r2_nonlinear*100:.2f}%)")


# ============================================================
#  RESULTS COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("  FINAL RESULTS COMPARISON")
print("=" * 60)
print(f"  Linear Regression R²:      {r2_linear:.4f}  ({r2_linear*100:.2f}%)")
print(f"  Non-Linear Regression R²:  {r2_nonlinear:.4f}  ({r2_nonlinear*100:.2f}%)")
improvement = (r2_nonlinear - r2_linear) * 100
print(f"  Improvement:               +{improvement:.2f} percentage points")
print("=" * 60)


# ============================================================
#  GENERATE COMPARISON CHART
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Plot 1: R² Bar Comparison ---
bars = axes[0].bar(['Linear\nRegression', 'Non-Linear\nRegression'],
                   [r2_linear * 100, r2_nonlinear * 100],
                   color=['#4A90D9', '#E85D75'], edgecolor='black', width=0.5)
axes[0].set_ylabel('R² Score (%)', fontsize=12)
axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylim(0, 105)
for bar, val in zip(bars, [r2_linear * 100, r2_nonlinear * 100]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.2f}%', ha='center', fontsize=12, fontweight='bold')

# --- Plot 2: Linear — Predicted vs Actual ---
axes[1].scatter(y_test_lin, y_pred_lin, alpha=0.3, s=8, color='#4A90D9')
max_val = max(y_test_lin.max(), y_pred_lin.max())
axes[1].plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Perfect Prediction')
axes[1].set_xlabel('Actual GHI (W/m²)', fontsize=11)
axes[1].set_ylabel('Predicted GHI (W/m²)', fontsize=11)
axes[1].set_title(f'Linear Model (R²={r2_linear*100:.1f}%)', fontsize=13, fontweight='bold')
axes[1].legend()

# --- Plot 3: Non-Linear — Predicted vs Actual ---
axes[2].scatter(y_test_nl, y_pred_nl, alpha=0.3, s=8, color='#E85D75')
max_val = max(y_test_nl.max(), y_pred_nl.max())
axes[2].plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Perfect Prediction')
axes[2].set_xlabel('Actual GHI (W/m²)', fontsize=11)
axes[2].set_ylabel('Predicted GHI (W/m²)', fontsize=11)
axes[2].set_title(f'Non-Linear Model (R²={r2_nonlinear*100:.1f}%)', fontsize=13, fontweight='bold')
axes[2].legend()

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'linear_vs_nonlinear_comparison.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nComparison chart saved to: {output_path}")
print("Done!")
