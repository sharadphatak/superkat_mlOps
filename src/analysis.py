"""Dataset exploration and post-training evaluation figures; prints grounded insights."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.common import NUMERIC, TARGET


def explore(data="data/SupeKart.csv", output="reports"):
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(data)
    frame.describe(include="all").T.to_csv(output / "data_descriptive_statistics.csv")
    print("Shape:", frame.shape)
    print("Missingness:\n", frame.isna().sum().to_string())
    print("Duplicate rows:", frame.duplicated().sum())
    print("Target distribution:\n", frame[TARGET].describe().to_string())
    print("Product categories:\n", frame.Product_Type.value_counts().to_string())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(frame[TARGET], bins=35, ax=axes[0])
    sns.scatterplot(data=frame, x="Product_MRP", y=TARGET, alpha=.4, ax=axes[1])
    axes[0].set_title("Revenue distribution (source units)")
    axes[1].set_title("MRP versus revenue")
    fig.tight_layout(); fig.savefig(figures / "revenue_and_mrp.png", dpi=140); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=frame, x="Store_Type", y=TARGET, ax=axes[0])
    axes[0].tick_params(axis="x", rotation=35)
    sns.heatmap(frame[NUMERIC + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1])
    fig.tight_layout(); fig.savefig(figures / "store_and_correlations.png", dpi=140); plt.close(fig)
    print("Descriptive correlations with revenue (association, not causation):\n",
          frame[NUMERIC + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False).to_string())
    print("Mean revenue by store type:\n", frame.groupby("Store_Type")[TARGET].agg(["count", "mean", "median"]).to_string())
    print("Missing features are imputed within CV; descriptive charts use observed values only.")
    if (output / "evaluation.json").exists():
        evaluation = json.loads((output / "evaluation.json").read_text())
        pred = pd.read_csv(output / "holdout_predictions.csv")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.scatterplot(data=pred, x="actual", y="predicted", alpha=.5, ax=axes[0])
        lim = [min(pred.actual.min(), pred.predicted.min()), max(pred.actual.max(), pred.predicted.max())]
        axes[0].plot(lim, lim, "r--")
        sns.histplot(pred.residual, bins=30, ax=axes[1])
        fig.tight_layout(); fig.savefig(figures / "evaluation.png", dpi=140); plt.close(fig)
        model, baseline = evaluation["model"], evaluation["baseline"]
        print(f"Selected algorithm: {evaluation['selected_algorithm']}; holdout RMSE {model['RMSE']:.2f}, MAE {model['MAE']:.2f}, R² {model['R2']:.3f}.")
        improvement = (1 - model["RMSE"] / baseline["RMSE"]) * 100 if baseline["RMSE"] else None
        print("RMSE change against mean baseline (% reduction):", improvement)
        print("No timestamps: these results do not establish future-period forecast quality.\n",
              "Top encoded-feature importances (impurity-based; may favor high-cardinality features):\n",
              pd.read_csv(output / "feature_importance.csv").head(12).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/SupeKart.csv")
    p.add_argument("--output", default="reports")
    a = p.parse_args(); explore(a.data, a.output)
