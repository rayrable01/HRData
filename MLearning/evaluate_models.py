"""
Оценка положительного эффекта моделей на ретроспективных данных.

Логика:
  - Модель НЕ видела final_score при обучении
  - Модель учитась на is_hired (сырые признаки -> нанят/нет)
  - После обучения модель выдаёт model_score (вероятность найма)
  - Сравниваем ранжирование модели с ранжированием по final_score

Метрики:
  1. Precision@K — доля реально нанятых в top-K
  2. Recall@K — какую долю нанятых нашли в top-K
  3. NDCG@K — качество ранжирования относительно is_hired
  4. Spearman correlation — насколько модель согласна с final_score
  5. Overlap@K — сколько кандидатов из top-K модели совпадают с top-K по final_score
  6. Uplift — насколько модель лучше baseline по Precision@K
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.resolve()
MODELS_DIR = SCRIPT_DIR / "saved_models"
RESULTS_DIR = SCRIPT_DIR / "evaluation_results"
RESULTS_DIR.mkdir(exist_ok=True)

TOP_K_VALUES = [10, 20, 30, 50, 100]


# ===================================================================
# MLP model (same architecture as in pipeline)
# ===================================================================

class CandidateMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ===================================================================
# Data loading
# ===================================================================

def load_data():
    df = pd.read_csv(SCRIPT_DIR / "candidates_data.csv")

    with open(MODELS_DIR / "metadata.json") as f:
        meta = json.load(f)

    feature_names = meta["feature_names"]
    label_encoders = meta["label_encoders"]

    for col, classes in label_encoders.items():
        if col in df.columns:
            le = LabelEncoder()
            le.classes_ = np.array(classes)
            df[col] = le.transform(df[col].astype(str))

    cols = [c for c in feature_names if c in df.columns]
    X = df[cols].values
    return df, X, feature_names


def load_models(feature_names):
    models = {}

    input_dim = len(feature_names)
    mlp = CandidateMLP(input_dim)
    mlp.load_state_dict(torch.load(MODELS_DIR / "mlp_model.pt", weights_only=True))
    mlp.eval()
    models["neural_ranking_mlp"] = mlp

    with open(MODELS_DIR / "catboost_model.pkl", "rb") as f:
        models["catboost_classification"] = pickle.load(f)

    pw_path = MODELS_DIR / "pairwise_lr_model.pkl"
    if pw_path.exists():
        with open(pw_path, "rb") as f:
            models["pairwise_ranking_lr"] = pickle.load(f)

    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return models, scaler


# ===================================================================
# Scoring functions
# ===================================================================

def score_baseline(df):
    return df["final_score"].values


def score_mlp(models, X, scaler):
    X_s = scaler.transform(X)
    X_t = torch.tensor(X_s, dtype=torch.float32)
    with torch.no_grad():
        logits = models["neural_ranking_mlp"](X_t)
        return torch.sigmoid(logits).cpu().numpy()


def score_catboost(models, X, scaler):
    X_s = scaler.transform(X)
    return models["catboost_classification"].predict_proba(X_s)[:, 1]


def score_pairwise(models, X, scaler):
    X_s = scaler.transform(X)
    return models["catboost_classification"].predict_proba(X_s)[:, 1]


# ===================================================================
# Evaluation
# ===================================================================

def evaluate_model(scores, is_hired, final_scores, k):
    ranked_indices = np.argsort(-scores)
    top_k_indices = ranked_indices[:k]

    top_k_hired = is_hired[top_k_indices]
    top_k_final = final_scores[top_k_indices]

    precision = top_k_hired.mean() if k > 0 else 0.0
    total_hired = is_hired.sum()
    recall = top_k_hired.sum() / total_hired if total_hired > 0 else 0.0
    ndcg = ndcg_score([is_hired], [scores], k=k)
    avg_final_score = top_k_final.mean()

    return {
        "k": k,
        "precision@k": precision,
        "recall@k": recall,
        "ndcg@k": ndcg,
        "avg_final_score": avg_final_score,
        "hired_in_top_k": int(top_k_hired.sum()),
    }


def compute_spearman(scores, final_scores):
    corr, p_value = spearmanr(scores, final_scores)
    return corr, p_value


def compute_overlap(scores_model, scores_baseline, k):
    top_model = set(np.argsort(-scores_model)[:k])
    top_baseline = set(np.argsort(-scores_baseline)[:k])
    overlap = len(top_model & top_baseline)
    return overlap, overlap / k


def run_evaluation():
    logger.info("=" * 60)
    logger.info("MODEL EFFECT EVALUATION ON RETROSPECTIVE DATA")
    logger.info("Model was trained WITHOUT final_score")
    logger.info("=" * 60)

    df, X, feature_names = load_data()
    models, scaler = load_models(feature_names)

    is_hired = df["is_hired"].values
    final_scores = df["final_score"].values

    scoring_fns = {
        "baseline (final_score)": lambda: score_baseline(df),
        "neural_ranking_mlp": lambda: score_mlp(models, X, scaler),
        "catboost_classification": lambda: score_catboost(models, X, scaler),
    }
    if "pairwise_ranking_lr" in models:
        scoring_fns["pairwise_ranking_lr"] = lambda: score_pairwise(models, X, scaler)

    # ---- Main evaluation ----
    all_results = []
    spearman_results = []
    overlap_results = []

    baseline_scores = score_baseline(df)

    for model_name, score_fn in scoring_fns.items():
        logger.info("Evaluating: %s", model_name)
        scores = score_fn()

        for k in TOP_K_VALUES:
            metrics = evaluate_model(scores, is_hired, final_scores, k)
            metrics["model"] = model_name
            all_results.append(metrics)

        # Spearman correlation with final_score
        if model_name != "baseline (final_score)":
            corr, p_val = compute_spearman(scores, final_scores)
            spearman_results.append({
                "model": model_name,
                "spearman_corr": corr,
                "p_value": p_val,
            })
            logger.info("  Spearman vs final_score: %.4f (p=%.6f)", corr, p_val)

            # Overlap with baseline top-K
            for k in TOP_K_VALUES:
                overlap, overlap_pct = compute_overlap(scores, baseline_scores, k)
                overlap_results.append({
                    "model": model_name,
                    "k": k,
                    "overlap": overlap,
                    "overlap_pct": overlap_pct,
                })

    results_df = pd.DataFrame(all_results)
    spearman_df = pd.DataFrame(spearman_results)
    overlap_df = pd.DataFrame(overlap_results)

    # ---- Uplift analysis ----
    logger.info("=" * 60)
    logger.info("UPLIFT vs BASELINE")
    logger.info("=" * 60)

    baseline_prec = results_df[results_df["model"] == "baseline (final_score)"].set_index("k")["precision@k"]

    uplift_rows = []
    for model_name in scoring_fns:
        if model_name == "baseline (final_score)":
            continue
        model_prec = results_df[results_df["model"] == model_name].set_index("k")["precision@k"]
        for k in TOP_K_VALUES:
            if k in baseline_prec.index and k in model_prec.index:
                uplift = model_prec[k] - baseline_prec[k]
                uplift_pct = (uplift / baseline_prec[k] * 100) if baseline_prec[k] > 0 else 0
                uplift_rows.append({
                    "model": model_name,
                    "k": k,
                    "baseline_precision": baseline_prec[k],
                    "model_precision": model_prec[k],
                    "uplift_abs": uplift,
                    "uplift_pct": uplift_pct,
                })

    uplift_df = pd.DataFrame(uplift_rows)
    logger.info("\n%s", uplift_df.to_string(index=False))

    # ---- Spearman summary ----
    logger.info("=" * 60)
    logger.info("SPEARMAN CORRELATION vs final_score")
    logger.info("=" * 60)
    if not spearman_df.empty:
        logger.info("\n%s", spearman_df.to_string(index=False))

    # ---- Overlap summary ----
    logger.info("=" * 60)
    logger.info("TOP-K OVERLAP vs BASELINE")
    logger.info("=" * 60)
    if not overlap_df.empty:
        logger.info("\n%s", overlap_df.to_string(index=False))

    # ---- Summary table ----
    logger.info("=" * 60)
    logger.info("SUMMARY (K=%d)", TOP_K_VALUES[2])
    logger.info("=" * 60)

    summary_k = TOP_K_VALUES[2]
    summary_data = results_df[results_df["k"] == summary_k].copy()
    summary = summary_data.pivot_table(
        index="model",
        values=["precision@k", "recall@k", "ndcg@k", "avg_final_score", "hired_in_top_k"],
    )
    logger.info("\n%s", summary.to_string())

    # ---- Plots ----
    _plot_precision_recall_curve(results_df)
    _plot_uplift_chart(uplift_df)
    _plot_ndcg_comparison(results_df)
    _plot_spearman_chart(spearman_df)
    _plot_overlap_chart(overlap_df)

    # ---- Save results ----
    results_df.to_csv(RESULTS_DIR / "evaluation_results.csv", index=False)
    uplift_df.to_csv(RESULTS_DIR / "uplift_analysis.csv", index=False)
    if not spearman_df.empty:
        spearman_df.to_csv(RESULTS_DIR / "spearman_correlation.csv", index=False)
    if not overlap_df.empty:
        overlap_df.to_csv(RESULTS_DIR / "topk_overlap.csv", index=False)

    # ---- Best model ----
    filtered = results_df[
        (results_df["model"] != "baseline (final_score)") &
        (results_df["k"] == summary_k)
    ]
    best_idx = filtered["ndcg@k"].idxmax()
    best_row = filtered.loc[best_idx]

    logger.info("=" * 60)
    logger.info("BEST MODEL: %s", best_row["model"])
    logger.info("  Precision@%d: %.4f", summary_k, best_row["precision@k"])
    logger.info("  Recall@%d: %.4f", summary_k, best_row["recall@k"])
    logger.info("  NDCG@%d: %.4f", summary_k, best_row["ndcg@k"])
    logger.info("  Avg Final Score in top-%d: %.2f", summary_k, best_row["avg_final_score"])
    logger.info("  Hired in top-%d: %d", summary_k, best_row["hired_in_top_k"])
    logger.info("=" * 60)

    return results_df, uplift_df


# ===================================================================
# Plotting
# ===================================================================

def _plot_precision_recall_curve(results_df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for model_name, color in zip(
        results_df["model"].unique(),
        sns.color_palette("Set2", len(results_df["model"].unique())),
    ):
        model_data = results_df[results_df["model"] == model_name].sort_values("k")
        axes[0].plot(model_data["k"], model_data["precision@k"], marker="o", label=model_name, color=color)
        axes[1].plot(model_data["k"], model_data["recall@k"], marker="o", label=model_name, color=color)

    axes[0].set_title("Precision@K")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Precision")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Recall@K")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Recall")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "precision_recall_curves.png", dpi=150)
    plt.close(fig)


def _plot_uplift_chart(uplift_df):
    if uplift_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(uplift_df))
    bars = ax.bar(x, uplift_df["uplift_pct"], color=sns.color_palette("Set2", len(x)))

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{row['model'][:15]}\n(K={row['k']})" for _, row in uplift_df.iterrows()],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_ylabel("Uplift vs Baseline (%)")
    ax.set_title("Model Uplift over Baseline (Precision@K)")
    ax.axhline(y=0, color="black", linewidth=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f"{height:+.1f}%", ha="center", va="bottom" if height > 0 else "top", fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "uplift_chart.png", dpi=150)
    plt.close(fig)


def _plot_ndcg_comparison(results_df):
    fig, ax = plt.subplots(figsize=(12, 6))

    for model_name, color in zip(
        results_df["model"].unique(),
        sns.color_palette("Set2", len(results_df["model"].unique())),
    ):
        model_data = results_df[results_df["model"] == model_name].sort_values("k")
        ax.plot(model_data["k"], model_data["ndcg@k"], marker="o", label=model_name, color=color)

    ax.set_title("NDCG@K Comparison")
    ax.set_xlabel("K")
    ax.set_ylabel("NDCG")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "ndcg_comparison.png", dpi=150)
    plt.close(fig)


def _plot_spearman_chart(spearman_df):
    if spearman_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(spearman_df["model"], spearman_df["spearman_corr"],
                   color=sns.color_palette("Set2", len(spearman_df)))

    ax.set_xlabel("Spearman Correlation with final_score")
    ax.set_title("Model Agreement with Final Score")
    ax.axvline(x=0, color="black", linewidth=0.5)

    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2.,
                f"{width:.3f}", ha="left", va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "spearman_correlation.png", dpi=150)
    plt.close(fig)


def _plot_overlap_chart(overlap_df):
    if overlap_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for model_name, color in zip(
        overlap_df["model"].unique(),
        sns.color_palette("Set2", len(overlap_df["model"].unique())),
    ):
        model_data = overlap_df[overlap_df["model"] == model_name].sort_values("k")
        ax.plot(model_data["k"], model_data["overlap_pct"], marker="o",
                label=model_name, color=color)

    ax.set_title("Top-K Overlap with Baseline (final_score)")
    ax.set_xlabel("K")
    ax.set_ylabel("Overlap (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "topk_overlap.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    results, uplift = run_evaluation()
