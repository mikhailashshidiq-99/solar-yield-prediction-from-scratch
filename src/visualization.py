"""Plotting helpers for model evaluation and comparison."""

import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curve(loss_history):
    """Plot the training loss curve."""
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color='purple', linewidth=2)
    plt.title("Training Loss Curve", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def plot_prediction_vs_actual(actual, predicted, r2, window=168):
    """Side-by-side: time series overlay + scatter plot."""
    predicted = np.maximum(0, predicted)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # time series overlay
    axes[0].plot(actual[:window], label='Actual', color='#1f77b4', linewidth=2)
    axes[0].plot(predicted[:window], label='Predicted', color='#ff7f0e', linestyle='--', linewidth=2)
    axes[0].set_title("7-Day Forecast on Unseen Data", fontsize=14)
    axes[0].set_ylabel("Solar Power Output (W/m²)")
    axes[0].set_xlabel("Hours")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # scatter plot
    max_val = max(np.max(actual), np.max(predicted))
    axes[1].scatter(actual, predicted, alpha=0.5, color='teal')
    axes[1].plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    axes[1].set_title(f"Model Accuracy ($R^2$: {r2:.3f})", fontsize=14)
    axes[1].set_xlabel("Actual Power")
    axes[1].set_ylabel("Predicted Power")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_model_comparison(r2_linear, r2_nonlinear, y_test_lin, y_pred_lin, y_test_nl, y_pred_nl):
    """3-panel comparison chart: bar chart + 2 scatter plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # bar chart
    bars = axes[0].bar(
        ['Linear\nRegression', 'Non-Linear\nRegression'],
        [r2_linear * 100, r2_nonlinear * 100],
        color=['#4A90D9', '#E85D75'], edgecolor='black', width=0.5
    )
    axes[0].set_ylabel('R² Score (%)', fontsize=12)
    axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 105)
    for bar, val in zip(bars, [r2_linear * 100, r2_nonlinear * 100]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{val:.2f}%', ha='center', fontsize=12, fontweight='bold')

    # linear scatter
    max_val = max(y_test_lin.max(), y_pred_lin.max())
    axes[1].scatter(y_test_lin, y_pred_lin, alpha=0.3, s=8, color='#4A90D9')
    axes[1].plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Perfect Prediction')
    axes[1].set_xlabel('Actual GHI (W/m²)', fontsize=11)
    axes[1].set_ylabel('Predicted GHI (W/m²)', fontsize=11)
    axes[1].set_title(f'Linear Model (R²={r2_linear * 100:.1f}%)', fontsize=13, fontweight='bold')
    axes[1].legend()

    # non-linear scatter
    max_val = max(y_test_nl.max(), y_pred_nl.max())
    axes[2].scatter(y_test_nl, y_pred_nl, alpha=0.3, s=8, color='#E85D75')
    axes[2].plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Perfect Prediction')
    axes[2].set_xlabel('Actual GHI (W/m²)', fontsize=11)
    axes[2].set_ylabel('Predicted GHI (W/m²)', fontsize=11)
    axes[2].set_title(f'Non-Linear Model (R²={r2_nonlinear * 100:.1f}%)', fontsize=13, fontweight='bold')
    axes[2].legend()

    plt.tight_layout()
    plt.show()
