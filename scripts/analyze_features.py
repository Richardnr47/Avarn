"""
Feature importance analysis for fire alarm testing price prediction model.
Extracts and visualizes feature importance from the trained model.
"""

import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def load_model_and_preprocessor(model_path, preprocessor_path):
    """Load the trained model and preprocessor."""
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    model = model_data["model"]
    model_name = model_data.get("model_name", "unknown")

    with open(preprocessor_path, "rb") as f:
        preprocessor_data = pickle.load(f)
    feature_columns = preprocessor_data["feature_columns"]

    return model, model_name, feature_columns


def get_feature_importance(model, feature_names):
    """
    Extract feature importance from the model.
    Handles different model types (tree-based vs linear).
    """
    importance_dict = {}

    # Tree-based models (Random Forest, Gradient Boosting)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        for name, importance in zip(feature_names, importances):
            importance_dict[name] = float(importance)

    # Linear models (coefficient magnitudes)
    elif hasattr(model, "coef_"):
        coef = model.coef_
        # For multi-output, take mean
        if coef.ndim > 1:
            coef = np.mean(np.abs(coef), axis=0)
        else:
            coef = np.abs(coef)

        # Normalize to sum to 1
        coef = coef / coef.sum()

        for name, importance in zip(feature_names, coef):
            importance_dict[name] = float(importance)

    else:
        print("Warning: Model type not supported for feature importance extraction")
        return None

    return importance_dict


def plot_feature_importance(importance_dict, output_path=None):
    """Create visualization of feature importance."""
    if importance_dict is None:
        print("No feature importance data available")
        return

    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    features, importances = zip(*sorted_features)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    fig.suptitle("Feature Importance Analysis", fontsize=16, fontweight="bold")

    # 1. Horizontal bar chart
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    bars = ax1.barh(features, importances, color=colors)
    ax1.set_xlabel("Importance Score", fontsize=12)
    ax1.set_title(
        "Feature Importance (Horizontal Bar Chart)", fontsize=14, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, importances)):
        ax1.text(imp, i, f" {imp:.4f}", va="center", fontsize=10)

    # 2. Pie chart (top features)
    ax2 = axes[1]
    # Show top 8 features in pie chart
    top_n = min(8, len(features))
    top_features = features[:top_n]
    top_importances = importances[:top_n]
    other_importance = sum(importances[top_n:]) if len(features) > top_n else 0

    if other_importance > 0:
        pie_features = list(top_features) + ["Other"]
        pie_importances = list(top_importances) + [other_importance]
    else:
        pie_features = top_features
        pie_importances = top_importances

    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(pie_features)))
    wedges, texts, autotexts = ax2.pie(
        pie_importances,
        labels=pie_features,
        autopct="%1.1f%%",
        colors=colors_pie,
        startangle=90,
    )
    ax2.set_title(
        f"Feature Importance Distribution (Top {top_n})", fontsize=14, fontweight="bold"
    )

    # Improve text readability
    for autotext in autotexts:
        autotext.set_color("black")
        autotext.set_fontweight("bold")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Feature importance plot saved to {output_path}")

    plt.show()


def analyze_feature_importance(importance_dict, feature_names):
    """Provide detailed analysis of feature importance."""
    if importance_dict is None:
        return

    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    # Calculate statistics
    importances = list(importance_dict.values())
    total_importance = sum(importances)

    print(f"\nTotal Features: {len(feature_names)}")
    print(f"Total Importance: {total_importance:.4f}")
    print(f"Mean Importance: {np.mean(importances):.4f}")
    print(f"Std Importance: {np.std(importances):.4f}")

    # Categorize features
    high_importance = [f for f, imp in sorted_features if imp > 0.1]
    medium_importance = [f for f, imp in sorted_features if 0.05 < imp <= 0.1]
    low_importance = [f for f, imp in sorted_features if imp <= 0.05]

    print(f"\n{'='*80}")
    print("FEATURE CATEGORIZATION")
    print("=" * 80)
    print(f"\nHigh Importance (>10%): {len(high_importance)} features")
    for feat, imp in sorted_features:
        if imp > 0.1:
            print(f"  - {feat}: {imp:.4f} ({imp*100:.2f}%)")

    print(f"\nMedium Importance (5-10%): {len(medium_importance)} features")
    for feat, imp in sorted_features:
        if 0.05 < imp <= 0.1:
            print(f"  - {feat}: {imp:.4f} ({imp*100:.2f}%)")

    print(f"\nLow Importance (<5%): {len(low_importance)} features")
    for feat, imp in sorted_features:
        if imp <= 0.05:
            print(f"  - {feat}: {imp:.4f} ({imp*100:.2f}%)")

    # Top 5 features
    print(f"\n{'='*80}")
    print("TOP 5 MOST IMPORTANT FEATURES")
    print("=" * 80)
    for i, (feat, imp) in enumerate(sorted_features[:5], 1):
        print(f"{i}. {feat}: {imp:.4f} ({imp*100:.2f}%)")

    # Cumulative importance
    print(f"\n{'='*80}")
    print("CUMULATIVE IMPORTANCE")
    print("=" * 80)
    cumulative = 0
    for i, (feat, imp) in enumerate(sorted_features, 1):
        cumulative += imp
        if i <= 5 or cumulative <= 0.8:
            print(f"Top {i} features: {cumulative:.4f} ({cumulative*100:.2f}%)")
            if cumulative >= 0.8:
                break

    print("=" * 80)

    return sorted_features


def save_importance_report(importance_dict, output_path):
    """Save feature importance to JSON file."""
    if importance_dict is None:
        return

    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    report = {
        "feature_importance": importance_dict,
        "sorted_features": [
            {"feature": f, "importance": float(imp), "percentage": float(imp * 100)}
            for f, imp in sorted_features
        ],
        "total_features": len(importance_dict),
        "total_importance": float(sum(importance_dict.values())),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nFeature importance report saved to {output_path}")


def main():
    """Main function."""
    model_path = "../models/best_model.pkl"
    preprocessor_path = "../models/preprocessor.pkl"
    output_dir = "../models"

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please train models first using train_model.py")
        return

    if not os.path.exists(preprocessor_path):
        print(f"Preprocessor file not found: {preprocessor_path}")
        return

    # Load model and preprocessor
    print("Loading model and preprocessor...")
    model, model_name, feature_names = load_model_and_preprocessor(
        model_path, preprocessor_path
    )
    print(f"Loaded model: {model_name}")
    print(f"Features: {len(feature_names)}")

    # Get feature importance
    print("\nExtracting feature importance...")
    importance_dict = get_feature_importance(model, feature_names)

    if importance_dict is None:
        print("Could not extract feature importance from this model type")
        return

    # Analyze
    sorted_features = analyze_feature_importance(importance_dict, feature_names)

    # Visualize
    print("\nGenerating visualizations...")
    plot_path = os.path.join(output_dir, "feature_importance.png")
    plot_feature_importance(importance_dict, plot_path)

    # Save report
    report_path = os.path.join(output_dir, "feature_importance.json")
    save_importance_report(importance_dict, report_path)

    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()
