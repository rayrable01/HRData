"""
ML Pipeline: Ранжирование кандидатов для вакансии Frontend Developer.

Полный пайплайн:
  1. NLP модель оценивает тексты кандидатов → 7 оценок
  2. ML модели (MLP, CatBoost, Pairwise) ранжируют по NLP-оценкам
  3. Сравнение с ground truth (final_score, is_hired)
  4. Top-30 лучших кандидатов
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
    f1_score, ndcg_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.resolve()
REAL_DATA_DIR = SCRIPT_DIR / "realistic_data"
MODELS_DIR = SCRIPT_DIR / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)


# ===================================================================
# Config
# ===================================================================

class Config:
    NLP_FEATURES = [
        "nlp_exp_score", "nlp_hard_score", "nlp_quest_score",
        "nlp_auth_score", "nlp_red_flags_score", "nlp_avail_score",
        "nlp_add_value_score", "nlp_final_score",
    ]
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    TOP_K = 30

    MLP_HIDDEN = 64
    MLP_EPOCHS = 100
    MLP_LR = 1e-3
    MLP_BATCH = 32
    MLP_DROPOUT = 0.2

    CAT_ITER = 200
    CAT_DEPTH = 5
    CAT_LR = 0.05

    XGB_N_EST = 150
    XGB_DEPTH = 5
    XGB_LR = 0.05

    LGB_N_EST = 100
    LGB_DEPTH = 4
    LGB_LR = 0.05


# ===================================================================
# MLP Model
# ===================================================================

class CandidateMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train, y_train, X_test, y_test, cfg):
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
# Data Loading
# ===================================================================

def load_data():
    df = pd.read_csv(REAL_DATA_DIR / "candidates_with_nlp_scores.csv")
    logger.info("Loaded %d candidates with NLP scores", len(df))

    with open(REAL_DATA_DIR / "vacancy.json", encoding="utf-8") as f:
        vacancy = json.load(f)

    X = df[Config.NLP_FEATURES].values
    y_clf = df["is_hired"].values
    final_scores = df["final_score"].values

    return df, X, y_clf, final_scores, vacancy


# ===================================================================
# Main Pipeline
# ===================================================================

def run_pipeline():
    cfg = Config()
    logger.info("=" * 60)
    logger.info("ML PIPELINE: Frontend Developer Ranking")
    logger.info("Input: NLP scores (model never sees ground truth)")
    logger.info("=" * 60)

    df, X, y_clf, final_scores, vacancy = load_data()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_clf, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE, stratify=y_clf,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    logger.info("Train: %d | Test: %d", len(X_train_s), len(X_test_s))

    # ---- MODEL 1: MLP ----
    logger.info("=" * 60)
    logger.info("MODEL 1: NEURAL RANKING (PyTorch MLP)")
    logger.info("=" * 60)

    mlp_model, mlp_metrics = train_mlp(X_train_s, y_train, X_test_s, y_test, cfg)

    # ---- MODEL 2: CatBoost ----
    logger.info("=" * 60)
    logger.info("MODEL 2: GRADIENT BOOSTING (CatBoost)")
    logger.info("=" * 60)

    cat_model = CatBoostClassifier(
        iterations=cfg.CAT_ITER, depth=cfg.CAT_DEPTH, learning_rate=cfg.CAT_LR,
        random_seed=cfg.RANDOM_STATE, verbose=False,
    )
    cat_model.fit(X_train_s, y_train)

    cat_proba = cat_model.predict_proba(X_test_s)[:, 1]
    cat_pred = (cat_proba >= 0.5).astype(int)

    cat_metrics = {
        "precision": precision_score(y_test, cat_pred, zero_division=0),
        "recall": recall_score(y_test, cat_pred, zero_division=0),
        "f1": f1_score(y_test, cat_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, cat_proba),
        "ndcg@10": ndcg_score([y_test], [cat_proba], k=cfg.TOP_K),
    }
    logger.info("CatBoost -> Precision: %.4f | Recall: %.4f | F1: %.4f | AUC: %.4f | NDCG@%d: %.4f",
                cat_metrics["precision"], cat_metrics["recall"], cat_metrics["f1"],
                cat_metrics["auc_roc"], cfg.TOP_K, cat_metrics["ndcg@10"])

    # ---- MODEL 3: Pairwise ----
    logger.info("=" * 60)
    logger.info("MODEL 3: PAIRWISE RANKING (Logistic Regression)")
    logger.info("=" * 60)

    # Создаём pairwise данные из NLP-оценок
    pw_data = _create_pairwise_from_nlp(df)
    pw_metrics = {}
    lr_model = None

    if pw_data is not None:
        X_pw, y_pw = pw_data
        X_train_pw, X_test_pw, y_train_pw, y_test_pw = train_test_split(
            X_pw, y_pw, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE,
        )

        lr_model = LogisticRegression(max_iter=1000, random_state=cfg.RANDOM_STATE)
        lr_model.fit(X_train_pw, y_train_pw)

        lr_proba = lr_model.predict_proba(X_test_pw)[:, 1]
        lr_pred = (lr_proba >= 0.5).astype(int)

        pw_metrics = {
            "accuracy": (lr_pred == y_test_pw).mean(),
            "precision": precision_score(y_test_pw, lr_pred, zero_division=0),
            "recall": recall_score(y_test_pw, lr_pred, zero_division=0),
            "f1": f1_score(y_test_pw, lr_pred, zero_division=0),
            "auc_roc": roc_auc_score(y_test_pw, lr_proba),
        }
        logger.info("Pairwise LR -> Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %.4f",
                    pw_metrics["accuracy"], pw_metrics["precision"], pw_metrics["recall"],
                    pw_metrics["f1"], pw_metrics["auc_roc"])
    else:
        logger.warning("No pairwise data available")

    # ---- Cross-Validation ----
    logger.info("=" * 60)
    logger.info("CROSS-VALIDATION")
    logger.info("=" * 60)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.RANDOM_STATE)
    cv_results = {}

    for name, model_cls, params in [
        ("catboost", CatBoostClassifier, dict(iterations=cfg.CAT_ITER, depth=cfg.CAT_DEPTH,
                                               learning_rate=cfg.CAT_LR, random_seed=cfg.RANDOM_STATE, verbose=False)),
        ("xgboost", XGBClassifier, dict(n_estimators=cfg.XGB_N_EST, max_depth=cfg.XGB_DEPTH,
                                          learning_rate=cfg.XGB_LR, random_state=cfg.RANDOM_STATE,
                                          n_jobs=-1, verbosity=0, eval_metric="logloss")),
        ("lightgbm", LGBMClassifier, dict(n_estimators=cfg.LGB_N_EST, max_depth=cfg.LGB_DEPTH,
                                            learning_rate=cfg.LGB_LR, random_state=cfg.RANDOM_STATE,
                                            n_jobs=-1, verbose=-1)),
    ]:
        from sklearn.pipeline import Pipeline as SkPipeline
        pipe = SkPipeline([("scaler", StandardScaler()), ("model", model_cls(**params))])
        scores = cross_val_score(pipe, X, y_clf, cv=kfold, scoring="f1")
        cv_results[name] = {"mean": scores.mean(), "std": scores.std()}
        logger.info("%s CV F1: %.4f +/- %.4f", name.upper(), scores.mean(), scores.std())

    # ---- Compare ----
    logger.info("=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)

    all_results = {"neural_ranking_mlp": mlp_metrics, "catboost": cat_metrics}
    if pw_metrics:
        all_results["pairwise_lr"] = pw_metrics

    rows = [{"model": name, **metrics} for name, metrics in all_results.items()]
    comparison_df = pd.DataFrame(rows)
    logger.info("\n%s", comparison_df.to_string(index=False))

    best_model = max(all_results, key=lambda k: all_results[k].get("ndcg@10", all_results[k].get("f1", 0)))
    logger.info("Best model: %s", best_model)

    # ---- Top-30 ----
    logger.info("=" * 60)
    logger.info("TOP-%d CANDIDATES", cfg.TOP_K)
    logger.info("=" * 60)

    X_all_s = scaler.transform(X)

    if best_model == "neural_ranking_mlp":
        X_t = torch.tensor(X_all_s, dtype=torch.float32)
        mlp_model.eval()
        with torch.no_grad():
            df["model_score"] = torch.sigmoid(mlp_model(X_t)).cpu().numpy()
    else:
        df["model_score"] = cat_model.predict_proba(X_all_s)[:, 1]

    top30 = df.sort_values("model_score", ascending=False).head(cfg.TOP_K).reset_index(drop=True)
    top30["rank"] = range(1, cfg.TOP_K + 1)

    display_cols = ["rank", "candidate_id", "name", "is_relevant", "quality_tier",
                     "final_score", "nlp_final_score", "model_score", "is_hired"]
    display_cols = [c for c in display_cols if c in top30.columns]
    logger.info("\n%s", top30[display_cols].to_string(index=False))

    # ---- Save ----
    _save_artifacts(mlp_model, cat_model, lr_model, scaler, all_results, cv_results,
                    comparison_df, top30, vacancy)

    logger.info("Pipeline completed. Artifacts saved to %s", MODELS_DIR)
    return comparison_df, top30


def _create_pairwise_from_nlp(df):
    """Создаёт pairwise данные из NLP-оценок."""
    nlp_features = ["nlp_exp_score", "nlp_hard_score", "nlp_quest_score",
                     "nlp_auth_score", "nlp_red_flags_score", "nlp_avail_score",
                     "nlp_add_value_score"]

    pairs = []
    relevant = df[df["is_relevant"] == True]

    for _ in range(500):
        i, j = np.random.choice(len(relevant), size=2, replace=False)
        a, b = relevant.iloc[i], relevant.iloc[j]

        diff = (a[nlp_features].values - b[nlp_features].values)
        label = 1 if a["final_score"] > b["final_score"] else 0

        pairs.append((*diff, label))

    pairs = np.array(pairs)
    return pairs[:, :-1], pairs[:, -1].astype(int)


def _save_artifacts(mlp_model, cat_model, lr_model, scaler, all_results,
                    cv_results, comparison_df, top30, vacancy):
    with open(MODELS_DIR / "mlp_model.pt", "wb") as f:
        torch.save(mlp_model.state_dict(), f)
    with open(MODELS_DIR / "catboost_model.pkl", "wb") as f:
        pickle.dump(cat_model, f)
    if lr_model:
        with open(MODELS_DIR / "pairwise_lr_model.pkl", "wb") as f:
            pickle.dump(lr_model, f)
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "vacancy": vacancy["title"],
        "model_results": {k: {mk: mv for mk, mv in v.items() if not isinstance(mv, (list, np.ndarray))}
                          for k, v in all_results.items()},
        "cv_results": {k: {mk: mv for mk, mv in v.items() if not isinstance(mv, (list, np.ndarray))}
                       for k, v in cv_results.items()},
        "top30": top30.to_dict(orient="records"),
        "saved_at": datetime.now().isoformat(),
    }
    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str, ensure_ascii=False)

    top30.to_csv(MODELS_DIR / "top30_candidates.csv", index=False, encoding="utf-8")
    comparison_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    logger.info("Saved all artifacts to %s", MODELS_DIR)


if __name__ == "__main__":
    comparison, top30 = run_pipeline()
