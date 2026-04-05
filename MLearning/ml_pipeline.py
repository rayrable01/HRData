"""
ML Pipeline для ранжирования кандидатов.

ВАЖНО: Модель НЕ видит final_score при обучении.
Она учится на is_hired (сырые признаки -> нанят/нет).

После обучения:
  - Модель выдаёт model_score (вероятность найма)
  - Ранжируем кандидатов по model_score
  - Сравниваем: ранг модели vs ранг по final_score
  - Насколько совпали -> качество модели

3 модели:
  1. Нейросетевое ранжирование (PyTorch MLP) — классификация is_hired
  2. Градиентный бустинг (CatBoost) — классификация is_hired
  3. Парное ранжирование (Logistic Regression) — Learning to Rank
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    ndcg_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.resolve()
MODELS_DIR = SCRIPT_DIR / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)


# ===================================================================
# Config
# ===================================================================

class Config:
    # Сырые признаки — БЕЗ final_score
    FEATURE_COLS = [
        "exp_score", "hard_score", "quest_score", "auth_score",
        "red_flags_score", "avail_score", "add_value_score",
        "experience_level", "education", "n_skills",
        "years_experience", "has_portfolio", "has_certifications",
        "expected_salary", "notice_period_days",
        "position_applied",
    ]
    CATEGORICAL_COLS = ["position_applied", "experience_level", "education"]
    PAIRWISE_SCORE_COLS = [
        "exp_score", "hard_score", "quest_score", "auth_score",
        "red_flags_score", "avail_score", "add_value_score",
    ]
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    TOP_K = 30

    # PyTorch MLP
    MLP_HIDDEN = 128
    MLP_EPOCHS = 150
    MLP_LR = 1e-3
    MLP_BATCH = 64
    MLP_DROPOUT = 0.2

    # CatBoost
    CAT_ITER = 300
    CAT_DEPTH = 6
    CAT_LR = 0.05

    # XGBoost (для CV)
    XGB_N_EST = 200
    XGB_DEPTH = 6
    XGB_LR = 0.05

    # LightGBM (для CV)
    LGB_N_EST = 150
    LGB_DEPTH = 5
    LGB_LR = 0.05

    # Logistic Regression (pairwise)
    LR_MAX_ITER = 1000


# ===================================================================
# 1. Neural Ranking — PyTorch MLP (классификация is_hired)
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


def train_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    cfg: Config,
):
    """MLP учится предсказывать is_hired. Выход — вероятность (model_score)."""
    device = torch.device("cpu")
    model = CandidateMLP(X_train.shape[1], cfg.MLP_HIDDEN, cfg.MLP_DROPOUT).to(device)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=cfg.MLP_BATCH, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.MLP_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.MLP_EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, cfg.MLP_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        scheduler.step()

        if epoch % 25 == 0 or epoch == 1:
            avg_loss = epoch_loss / len(train_ds)
            logger.info("  MLP Epoch %d/%d  loss=%.4f", epoch, cfg.MLP_EPOCHS, avg_loss)

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(X_test_t)
        y_proba = torch.sigmoid(logits).cpu().numpy()

    y_pred = (y_proba >= 0.5).astype(int)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    ndcg = ndcg_score([y_test], [y_proba], k=cfg.TOP_K)

    logger.info("  MLP -> Precision: %.4f | Recall: %.4f | F1: %.4f | AUC: %.4f | NDCG@%d: %.4f",
                precision, recall, f1, auc, cfg.TOP_K, ndcg)

    return model, {"precision": precision, "recall": recall, "f1": f1, "auc_roc": auc, "ndcg@10": ndcg}


# ===================================================================
# Data preparation
# ===================================================================

def load_candidates(data_path: Path, cfg: Config):
    """Загружаем данные. final_score НЕ используется при обучении."""
    df = pd.read_csv(data_path)
    logger.info("Loaded %d candidates", len(df))

    label_encoders = {}
    for col in cfg.CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    cols = [c for c in cfg.FEATURE_COLS if c in df.columns]
    X = df[cols].values
    y_clf = df["is_hired"].values

    # final_score сохраняем ТОЛЬКО для последующей оценки (не для обучения!)
    final_scores = df["final_score"].values

    return X, y_clf, final_scores, cols, label_encoders, df


def load_pairwise_data(cfg: Config):
    pw_path = SCRIPT_DIR / "pairwise_ranking_data.csv"
    if not pw_path.exists():
        logger.warning("pairwise_ranking_data.csv not found")
        return None, None, None

    pw = pd.read_csv(pw_path)
    logger.info("Loaded %d pairwise records", len(pw))

    score_cols = cfg.PAIRWISE_SCORE_COLS
    X_pw = pw[[f"a_{c}" for c in score_cols]].values - pw[[f"b_{c}" for c in score_cols]].values
    y_pw = (pw["label"] > 0).astype(int).values
    return X_pw, y_pw, pw


# ===================================================================
# Plotting
# ===================================================================

def plot_feature_importance(importances: np.ndarray, feature_names: list[str], title: str):
    feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    feat_imp = feat_imp.sort_values("importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feat_imp, x="importance", y="feature", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Feature Importance")
    fig.tight_layout()
    save_path = MODELS_DIR / f"{title.replace(' ', '_').lower()}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot: %s", save_path)


def plot_mlp_weights(model: CandidateMLP, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    idx = 0
    for layer in model.net:
        if isinstance(layer, nn.Linear) and idx < 3:
            weights = layer.weight.detach().cpu().numpy()
            im = axes[idx].imshow(np.abs(weights), aspect="auto", cmap="viridis")
            axes[idx].set_title(f"Layer weights ({weights.shape})")
            fig.colorbar(im, ax=axes[idx])
            idx += 1
    fig.tight_layout()
    save_path = MODELS_DIR / f"{title.replace(' ', '_').lower()}_weights.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ===================================================================
# Main Pipeline
# ===================================================================

def run_pipeline():
    cfg = Config()
    logger.info("=" * 60)
    logger.info("CANDIDATE RANKING PIPELINE")
    logger.info("Model trains on is_hired — final_score is HIDDEN")
    logger.info("=" * 60)

    # ---- Load data ----
    X, y_clf, final_scores, feature_names, label_encoders, raw_df = load_candidates(
        SCRIPT_DIR / "candidates_data.csv", cfg,
    )

    # ---- Train/Test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_clf, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE, stratify=y_clf,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    logger.info("Train: %d | Test: %d", len(X_train_s), len(X_test_s))

    # =================================================================
    # MODEL 1: Neural Ranking (PyTorch MLP — классификация is_hired)
    # =================================================================
    logger.info("=" * 60)
    logger.info("MODEL 1: NEURAL RANKING (PyTorch MLP)")
    logger.info("=" * 60)

    mlp_model, mlp_metrics = train_mlp(X_train_s, y_train, X_test_s, y_test, cfg)
    plot_mlp_weights(mlp_model, "Neural_Ranking_MLP")

    # =================================================================
    # MODEL 2: Gradient Boosting (CatBoost — классификация is_hired)
    # =================================================================
    logger.info("=" * 60)
    logger.info("MODEL 2: GRADIENT BOOSTING (CatBoost Classification)")
    logger.info("=" * 60)

    cat_model = CatBoostClassifier(
        iterations=cfg.CAT_ITER,
        depth=cfg.CAT_DEPTH,
        learning_rate=cfg.CAT_LR,
        random_seed=cfg.RANDOM_STATE,
        verbose=False,
    )
    cat_model.fit(X_train_s, y_train)

    cat_proba = cat_model.predict_proba(X_test_s)[:, 1]
    cat_pred = (cat_proba >= 0.5).astype(int)

    cat_precision = precision_score(y_test, cat_pred, zero_division=0)
    cat_recall = recall_score(y_test, cat_pred, zero_division=0)
    cat_f1 = f1_score(y_test, cat_pred, zero_division=0)
    cat_auc = roc_auc_score(y_test, cat_proba)
    cat_ndcg = ndcg_score([y_test], [cat_proba], k=cfg.TOP_K)

    cat_metrics = {
        "precision": cat_precision, "recall": cat_recall,
        "f1": cat_f1, "auc_roc": cat_auc, "ndcg@10": cat_ndcg,
    }
    logger.info("CatBoost -> Precision: %.4f | Recall: %.4f | F1: %.4f | AUC: %.4f | NDCG@%d: %.4f",
                cat_precision, cat_recall, cat_f1, cat_auc, cfg.TOP_K, cat_ndcg)

    plot_feature_importance(cat_model.feature_importances_, feature_names, "CatBoost_Classification")

    # =================================================================
    # MODEL 3: Pairwise Ranking (Logistic Regression)
    # =================================================================
    logger.info("=" * 60)
    logger.info("MODEL 3: PAIRWISE RANKING (Logistic Regression)")
    logger.info("=" * 60)

    X_pw, y_pw, pw_df = load_pairwise_data(cfg)
    pw_metrics = {}
    lr_model = None

    if X_pw is not None:
        X_train_pw, X_test_pw, y_train_pw, y_test_pw = train_test_split(
            X_pw, y_pw, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE,
        )

        lr_model = LogisticRegression(max_iter=cfg.LR_MAX_ITER, random_state=cfg.RANDOM_STATE)
        lr_model.fit(X_train_pw, y_train_pw)

        lr_proba = lr_model.predict_proba(X_test_pw)[:, 1]
        lr_pred = (lr_proba >= 0.5).astype(int)

        lr_acc = (lr_pred == y_test_pw).mean()
        lr_prec = precision_score(y_test_pw, lr_pred, zero_division=0)
        lr_rec = recall_score(y_test_pw, lr_pred, zero_division=0)
        lr_f1 = f1_score(y_test_pw, lr_pred, zero_division=0)
        lr_auc = roc_auc_score(y_test_pw, lr_proba)

        pw_ndcg = _compute_pairwise_ndcg(pw_df, lr_model, cfg.PAIRWISE_SCORE_COLS, cfg.TOP_K)

        pw_metrics = {
            "accuracy": lr_acc, "precision": lr_prec, "recall": lr_rec,
            "f1": lr_f1, "auc_roc": lr_auc, "ndcg@10": pw_ndcg,
        }
        logger.info("Pairwise LR -> Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %.4f | NDCG@%d: %.4f",
                    lr_acc, lr_prec, lr_rec, lr_f1, lr_auc, cfg.TOP_K, pw_ndcg)

        plot_feature_importance(
            np.abs(lr_model.coef_[0]),
            [f"diff_{c}" for c in cfg.PAIRWISE_SCORE_COLS],
            "Pairwise_LR_Coefficients",
        )
    else:
        logger.warning("Skipping pairwise model — no data")

    # =================================================================
    # Cross-Validation
    # =================================================================
    logger.info("=" * 60)
    logger.info("CROSS-VALIDATION")
    logger.info("=" * 60)

    kfold_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.RANDOM_STATE)

    cv_results = {}
    for name, model_cls, params in [
        ("catboost_clf", CatBoostClassifier,
         dict(iterations=cfg.CAT_ITER, depth=cfg.CAT_DEPTH, learning_rate=cfg.CAT_LR,
              random_seed=cfg.RANDOM_STATE, verbose=False)),
        ("xgboost_clf", XGBClassifier,
         dict(n_estimators=cfg.XGB_N_EST, max_depth=cfg.XGB_DEPTH, learning_rate=cfg.XGB_LR,
              random_state=cfg.RANDOM_STATE, n_jobs=-1, verbosity=0, eval_metric="logloss")),
        ("lightgbm_clf", LGBMClassifier,
         dict(n_estimators=cfg.LGB_N_EST, max_depth=cfg.LGB_DEPTH, learning_rate=cfg.LGB_LR,
              random_state=cfg.RANDOM_STATE, n_jobs=-1, verbose=-1)),
    ]:
        from sklearn.pipeline import Pipeline as SkPipeline
        pipe = SkPipeline([("scaler", StandardScaler()), ("model", model_cls(**params))])
        scores = cross_val_score(pipe, X, y_clf, cv=kfold_clf, scoring="f1", n_jobs=-1)
        cv_results[name] = {"mean": scores.mean(), "std": scores.std()}
        logger.info("%s CV F1: %.4f +/- %.4f", name.upper(), scores.mean(), scores.std())

    # =================================================================
    # Compare models
    # =================================================================
    logger.info("=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)

    all_results = {
        "neural_ranking_mlp": mlp_metrics,
        "catboost_classification": cat_metrics,
    }
    if pw_metrics:
        all_results["pairwise_ranking_lr"] = pw_metrics

    rows = []
    for name, metrics in all_results.items():
        row = {"model": name}
        row.update(metrics)
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    logger.info("\n%s", comparison_df.to_string(index=False))

    best_model_name = max(all_results, key=lambda k: all_results[k].get("ndcg@10", 0))
    logger.info("Best model by NDCG@10: %s (%.4f)", best_model_name,
                all_results[best_model_name]["ndcg@10"])

    # =================================================================
    # Generate Top-30 ranking
    # =================================================================
    logger.info("=" * 60)
    logger.info("TOP-%d CANDIDATE RANKING", cfg.TOP_K)
    logger.info("=" * 60)

    top30 = _generate_top30(raw_df, X, scaler, mlp_model, cat_model, lr_model, cfg, best_model_name)
    logger.info("\n%s", top30.to_string(index=False))

    # =================================================================
    # Save everything
    # =================================================================
    _save_artifacts(
        mlp_model, cat_model, lr_model,
        scaler, label_encoders, feature_names,
        all_results, cv_results, comparison_df, top30,
    )

    logger.info("Pipeline completed. Artifacts saved to %s", MODELS_DIR)
    return comparison_df, top30


# ===================================================================
# Helper: pairwise NDCG
# ===================================================================

def _compute_pairwise_ndcg(pw_df, model, score_cols, top_k):
    ndcg_vals = []
    for position in pw_df["position"].unique():
        pos_pw = pw_df[pw_df["position"] == position]
        if len(pos_pw) < 3:
            continue

        cand_scores = {}
        cand_pred_sums = {}
        cand_pred_counts = {}

        a_cols = [f"a_{c}" for c in score_cols]
        b_cols = [f"b_{c}" for c in score_cols]
        diffs = pos_pw[a_cols].values - pos_pw[b_cols].values
        probas = model.predict_proba(diffs)[:, 1]

        for idx, (_, row) in enumerate(pos_pw.iterrows()):
            a, b = row["candidate_a"], row["candidate_b"]
            diff = row["score_diff"]
            p = probas[idx]

            cand_scores.setdefault(a, []).append(diff if row["label"] > 0 else 0)
            cand_scores.setdefault(b, []).append(diff if row["label"] < 0 else 0)

            cand_pred_sums[a] = cand_pred_sums.get(a, 0.0) + p
            cand_pred_counts[a] = cand_pred_counts.get(a, 0) + 1
            cand_pred_sums[b] = cand_pred_sums.get(b, 0.0) + (1 - p)
            cand_pred_counts[b] = cand_pred_counts.get(b, 0) + 1

        candidates = list(cand_scores.keys())
        if len(candidates) < 3:
            continue

        relevance = np.array([np.mean(cand_scores[c]) for c in candidates])
        predictions = np.array([cand_pred_sums[c] / cand_pred_counts[c] for c in candidates])

        if relevance.sum() > 0:
            ndcg_vals.append(ndcg_score([relevance], [predictions], k=min(top_k, len(candidates))))

    return float(np.mean(ndcg_vals)) if ndcg_vals else 0.0


# ===================================================================
# Helper: Top-30 generation
# ===================================================================

def _generate_top30(raw_df, X, scaler, mlp_model, cat_model, lr_model, cfg, best_model_name):
    df = raw_df.copy()
    X_s = scaler.transform(X)

    if best_model_name == "neural_ranking_mlp":
        X_t = torch.tensor(X_s, dtype=torch.float32)
        mlp_model.eval()
        with torch.no_grad():
            logits = mlp_model(X_t)
            df["model_score"] = torch.sigmoid(logits).cpu().numpy()

    elif best_model_name == "catboost_classification":
        df["model_score"] = cat_model.predict_proba(X_s)[:, 1]

    else:  # pairwise
        df["model_score"] = cat_model.predict_proba(X_s)[:, 1]

    df = df.sort_values("model_score", ascending=False).head(cfg.TOP_K).reset_index(drop=True)
    df["rank"] = range(1, cfg.TOP_K + 1)

    cols_out = ["rank", "candidate_id", "position_applied", "experience_level",
                "final_score", "model_score", "is_hired"]
    cols_out = [c for c in cols_out if c in df.columns]
    return df[cols_out]


# ===================================================================
# Helper: Save artifacts
# ===================================================================

def _save_artifacts(mlp_model, cat_model, lr_model,
                    scaler, label_encoders, feature_names,
                    all_results, cv_results, comparison_df, top30):
    with open(MODELS_DIR / "mlp_model.pt", "wb") as f:
        torch.save(mlp_model.state_dict(), f)

    with open(MODELS_DIR / "catboost_model.pkl", "wb") as f:
        pickle.dump(cat_model, f)

    if lr_model is not None:
        with open(MODELS_DIR / "pairwise_lr_model.pkl", "wb") as f:
            pickle.dump(lr_model, f)

    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "feature_names": feature_names,
        "label_encoders": {k: list(v.classes_) for k, v in label_encoders.items()},
        "model_results": {k: {mk: mv for mk, mv in v.items()
                               if not isinstance(mv, (list, np.ndarray))}
                          for k, v in all_results.items()},
        "cv_results": {k: {mk: mv for mk, mv in v.items()
                           if not isinstance(mv, (list, np.ndarray))}
                       for k, v in cv_results.items()},
        "top30": top30.to_dict(orient="records"),
        "saved_at": datetime.now().isoformat(),
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    top30.to_csv(MODELS_DIR / "top30_candidates.csv", index=False)
    comparison_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    logger.info("Saved all artifacts to %s", MODELS_DIR)


if __name__ == "__main__":
    comparison, top30 = run_pipeline()
