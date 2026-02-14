"""
Visualization script for model training results.
Creates charts comparing model performance.
"""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def load_results(results_path):
    """Load training results from JSON file."""
    with open(results_path, "r") as f:
        results = json.load(f)
    return results


def plot_model_comparison(results, output_dir="../models"):
    """Create comparison plots for all models."""

    # Extract data
    models = list(results.keys())
    train_rmse = [results[m]["train_rmse"] for m in models]
    test_rmse = [results[m]["test_rmse"] for m in models]
    train_r2 = [results[m]["train_r2"] for m in models]
    test_r2 = [results[m]["test_r2"] for m in models]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")

    # 1. RMSE Comparison
    ax1 = axes[0, 0]
    x = range(len(models))
    width = 0.35
    ax1.bar([i - width / 2 for i in x], train_rmse, width, label="Train RMSE", alpha=0.8)
    ax1.bar([i + width / 2 for i in x], test_rmse, width, label="Test RMSE", alpha=0.8)
    ax1.set_xlabel("Models")
    ax1.set_ylabel("RMSE")
    ax1.set_title("Root Mean Squared Error (RMSE)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. R² Score Comparison
    ax2 = axes[0, 1]
    ax2.bar([i - width / 2 for i in x], train_r2, width, label="Train R²", alpha=0.8)
    ax2.bar([i + width / 2 for i in x], test_r2, width, label="Test R²", alpha=0.8)
    ax2.set_xlabel("Models")
    ax2.set_ylabel("R² Score")
    ax2.set_title("R² Score (Coefficient of Determination)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="r", linestyle="--", alpha=0.5)

    # 3. Test RMSE (focused view, excluding SVR for better scale)
    ax3 = axes[1, 0]
    filtered_models = [m for m in models if results[m]["test_rmse"] < 2000]
    filtered_test_rmse = [results[m]["test_rmse"] for m in filtered_models]
    colors = ["green" if m == "ridge" else "steelblue" for m in filtered_models]
    bars = ax3.bar(filtered_models, filtered_test_rmse, color=colors, alpha=0.7)
    ax3.set_xlabel("Models")
    ax3.set_ylabel("Test RMSE")
    ax3.set_title("Test RMSE (Best Models)")
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 4. Overfitting Analysis (Train vs Test RMSE ratio)
    ax4 = axes[1, 1]
    overfit_ratio = [test_rmse[i] / train_rmse[i] for i in range(len(models))]
    colors_overfit = [
        "green" if ratio < 1.2 else "orange" if ratio < 2 else "red" for ratio in overfit_ratio
    ]
    bars = ax4.bar(models, overfit_ratio, color=colors_overfit, alpha=0.7)
    ax4.set_xlabel("Models")
    ax4.set_ylabel("Test RMSE / Train RMSE")
    ax4.set_title("Overfitting Analysis (Lower is Better)")
    ax4.tick_params(axis="x", rotation=45)
    ax4.axhline(y=1.0, color="g", linestyle="--", alpha=0.5, label="Perfect")
    ax4.axhline(y=1.5, color="orange", linestyle="--", alpha=0.5, label="Acceptable")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Comparison plot saved to {output_path}")

    plt.show()


def create_results_table(results, output_dir="../models"):
    """Create a formatted results table."""
    df = pd.DataFrame(results).T

    # Reorder columns
    column_order = ["train_rmse", "test_rmse", "train_mae", "test_mae", "train_r2", "test_r2"]
    df = df[column_order]

    # Round values
    df = df.round(2)

    # Add best model indicator
    best_model = df["test_rmse"].idxmin()
    df["best"] = df.index == best_model

    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 80)
    print(df.to_string())
    print("\n" + "=" * 80)
    print(f"Best Model: {best_model} (Lowest Test RMSE: {df.loc[best_model, 'test_rmse']:.2f})")
    print("=" * 80)

    # Save to CSV
    csv_path = os.path.join(output_dir, "results_table.csv")
    df.to_csv(csv_path)
    print(f"\nResults table saved to {csv_path}")

    return df


def main():
    """Main function."""
    results_path = "../models/training_results.json"

    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        print("Please train models first using train_model.py")
        return

    # Load results
    results = load_results(results_path)

    # Create results table
    df = create_results_table(results)

    # Create visualizations
    print("\nGenerating visualizations...")
    plot_model_comparison(results)


if __name__ == "__main__":
    main()
