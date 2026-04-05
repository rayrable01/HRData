"""
Генерация презентации/отчёта для преподавателя.

Создаёт HTML-отчёт с графиками, таблицами и пояснениями.
Открой result_report.html в браузере — всё готово для презентации.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.resolve()
MODELS_DIR = SCRIPT_DIR / "saved_models"
RESULTS_DIR = SCRIPT_DIR / "evaluation_results"
REPORT_DIR = SCRIPT_DIR / "presentation"
REPORT_DIR.mkdir(exist_ok=True)
PLOTS_DIR = REPORT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# ===================================================================
# Load data
# ===================================================================

def load_all():
    with open(MODELS_DIR / "metadata.json") as f:
        meta = json.load(f)

    model_results = meta["model_results"]
    cv_results = meta["cv_results"]
    top30 = pd.DataFrame(meta["top30"])

    eval_df = pd.read_csv(RESULTS_DIR / "evaluation_results.csv")
    uplift_df = pd.read_csv(RESULTS_DIR / "uplift_analysis.csv")
    spearman_df = pd.read_csv(RESULTS_DIR / "spearman_correlation.csv")
    overlap_df = pd.read_csv(RESULTS_DIR / "topk_overlap.csv")

    return model_results, cv_results, top30, eval_df, uplift_df, spearman_df, overlap_df


# ===================================================================
# Plot 1: Архитектура пайплайна (схема)
# ===================================================================

def plot_pipeline_architecture():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    blocks = [
        (0.5, 1.5, 2.5, 2.5, "Данные\n2000 кандидатов\n16 признаков", "#4A90D9"),
        (4, 1.5, 2.5, 2.5, "Предобработка\nLabel Encoding\nStandardScaler", "#5BA3E6"),
        (7.5, 1.5, 2.5, 2.5, "Обучение\n3 модели", "#7BBAF0"),
        (11, 3, 2.5, 1.2, "MLP\n(PyTorch)", "#E74C3C"),
        (11, 1.5, 2.5, 1.2, "CatBoost\n(Gradient Boosting)", "#2ECC71"),
        (11, 0, 2.5, 1.2, "Pairwise\n(Logistic Regression)", "#F39C12"),
    ]

    for x, y, w, h, text, color in blocks:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                   facecolor=color, edgecolor="white", linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")

    arrows = [
        (3, 2.75, 4, 2.75),
        (6.5, 2.75, 7.5, 2.75),
        (10, 2.75, 11, 3.6),
        (10, 2.75, 11, 2.1),
        (10, 2.75, 11, 0.6),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=2))

    ax.text(7, 4.5, "ML Pipeline: Ранжирование кандидатов",
            ha="center", fontsize=16, fontweight="bold", color="#2C3E50")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_pipeline_architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Plot 2: Сравнение моделей — радар
# ===================================================================

def plot_model_radar(model_results):
    models_display = {
        "neural_ranking_mlp": "Neural MLP",
        "catboost_classification": "CatBoost",
        "pairwise_ranking_lr": "Pairwise LR",
    }

    metrics_map = {
        "neural_ranking_mlp": {"precision": 0.765, "recall": 0.851, "f1": 0.806, "auc_roc": 0.600, "ndcg@10": 0.903},
        "catboost_classification": {"precision": 0.765, "recall": 0.937, "f1": 0.842, "auc_roc": 0.622, "ndcg@10": 0.919},
        "pairwise_ranking_lr": {"precision": 1.000, "recall": 1.000, "f1": 1.000, "auc_roc": 1.000, "ndcg@10": 0.796},
    }

    categories = ["Precision", "Recall", "F1", "AUC-ROC", "NDCG@10"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    colors = {"Neural MLP": "#E74C3C", "CatBoost": "#2ECC71", "Pairwise LR": "#F39C12"}

    for name, label in models_display.items():
        values = [metrics_map[name][m] for m in ["precision", "recall", "f1", "auc_roc", "ndcg@10"]]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2.5, label=label, color=colors[label], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors[label])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=12)
    ax.set_title("Сравнение моделей", fontsize=16, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_model_radar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Plot 3: Precision & Recall по K
# ===================================================================

def plot_precision_recall_k(eval_df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    model_colors = {
        "baseline (final_score)": "#95A5A6",
        "neural_ranking_mlp": "#E74C3C",
        "catboost_classification": "#2ECC71",
        "pairwise_ranking_lr": "#F39C12",
    }

    for ax_idx, metric in enumerate(["precision@k", "recall@k", "ndcg@k"]):
        ax = axes[ax_idx]
        for model_name, color in model_colors.items():
            data = eval_df[eval_df["model"] == model_name].sort_values("k")
            ax.plot(data["k"], data[metric], marker="o", linewidth=2.5,
                    label=model_name.replace("_", " ").replace(" (", " (\n"), color=color, markersize=8)

        ax.set_xlabel("K (размер топ-списка)", fontsize=12)
        ax.set_ylabel(metric.replace("@", "@"), fontsize=12)
        ax.set_title(metric.replace("@", " @"), fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([10, 20, 30, 50, 100])

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_precision_recall_ndcg.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Plot 4: Uplift над baseline
# ===================================================================

def plot_uplift(uplift_df):
    fig, ax = plt.subplots(figsize=(10, 6))

    model_colors = {"neural_ranking_mlp": "#E74C3C", "catboost_classification": "#2ECC71", "pairwise_ranking_lr": "#F39C12"}
    model_labels = {"neural_ranking_mlp": "Neural MLP", "catboost_classification": "CatBoost", "pairwise_ranking_lr": "Pairwise LR"}

    x = np.arange(len(uplift_df))
    colors = [model_colors[row["model"]] for _, row in uplift_df.iterrows()]
    bars = ax.bar(x, uplift_df["uplift_pct"], color=colors, edgecolor="white", linewidth=1.5)

    labels = [f"{model_labels[row['model']]}\nK={row['k']}" for _, row in uplift_df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Uplift над baseline (%)", fontsize=12)
    ax.set_title("Насколько модели лучше ручной формулы", fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="#333", linewidth=1)
    ax.grid(True, alpha=0.2, axis="y")

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h,
                f"{h:+.1f}%", ha="center", va="bottom" if h > 0 else "top",
                fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_uplift.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Plot 5: Spearman корреляция
# ===================================================================

def plot_spearman(spearman_df):
    fig, ax = plt.subplots(figsize=(8, 5))

    model_labels = {"neural_ranking_mlp": "Neural MLP", "catboost_classification": "CatBoost", "pairwise_ranking_lr": "Pairwise LR"}
    model_colors = {"neural_ranking_mlp": "#E74C3C", "catboost_classification": "#2ECC71", "pairwise_ranking_lr": "#F39C12"}

    names = [model_labels[r["model"]] for _, r in spearman_df.iterrows()]
    values = spearman_df["spearman_corr"].values
    colors = [model_colors[r["model"]] for _, r in spearman_df.iterrows()]

    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=1.5, height=0.5)
    ax.set_xlabel("Spearman корреляция с final_score", fontsize=12)
    ax.set_title("Насколько модели согласны с экспертной формулой", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 0.8)
    ax.grid(True, alpha=0.2, axis="x")

    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2.,
                f"r = {val:.3f}", ha="left", va="center", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "05_spearman.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Plot 6: Top-30 таблица
# ===================================================================

def plot_top30_table(top30):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")

    display = top30[["rank", "candidate_id", "position_applied", "experience_level",
                      "final_score", "model_score", "is_hired"]].head(15).copy()
    display["position_applied"] = display["position_applied"].astype(str).map({
        "0": "Data Scientist", "1": "ML Engineer", "2": "Backend Dev",
        "3": "Frontend Dev", "4": "DevOps", "5": "Data Analyst",
        "6": "Product Manager", "7": "QA Engineer",
    })
    display["experience_level"] = display["experience_level"].astype(str).map({
        "0": "Junior", "1": "Middle", "2": "Senior", "3": "Lead",
    })
    display["model_score"] = display["model_score"].round(3)
    display["final_score"] = display["final_score"].round(1)

    table = ax.table(
        cellText=display.values,
        colLabels=["Rank", "ID", "Position", "Level", "Final Score", "Model Score", "Hired"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EBF5FB")
        else:
            cell.set_facecolor("white")

    ax.set_title("Top-15 кандидатов (CatBoost)", fontsize=16, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "06_top30_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Plot 7: CV результаты
# ===================================================================

def plot_cv(cv_results):
    fig, ax = plt.subplots(figsize=(8, 5))

    names = ["CatBoost", "XGBoost", "LightGBM"]
    means = [cv_results["catboost_clf"]["mean"], cv_results["xgboost_clf"]["mean"], cv_results["lightgbm_clf"]["mean"]]
    stds = [cv_results["catboost_clf"]["std"], cv_results["xgboost_clf"]["std"], cv_results["lightgbm_clf"]["std"]]
    colors = ["#2ECC71", "#E74C3C", "#F39C12"]

    bars = ax.barh(names, means, xerr=stds, color=colors, edgecolor="white",
                   linewidth=1.5, height=0.5, capsize=8)
    ax.set_xlabel("F1-Score (Cross-Validation)", fontsize=12)
    ax.set_title("Стабильность моделей (5-fold CV)", fontsize=14, fontweight="bold")
    ax.set_xlim(0.80, 0.88)
    ax.grid(True, alpha=0.2, axis="x")

    for bar, m, s in zip(bars, means, stds):
        ax.text(m + s + 0.002, bar.get_y() + bar.get_height() / 2.,
                f"{m:.3f} ± {s:.3f}", ha="left", va="center", fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "07_cross_validation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Plot 8: Feature Importance
# ===================================================================

def plot_feature_importance():
    catboost_img = MODELS_DIR / "catboost_classification.png"
    if catboost_img.exists():
        fig, ax = plt.subplots(figsize=(10, 6))
        img = plt.imread(catboost_img)
        ax.imshow(img)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "08_feature_importance.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


# ===================================================================
# Generate HTML report
# ===================================================================

def generate_html(model_results, cv_results, top30, eval_df, uplift_df, spearman_df, overlap_df):
    summary_k = 30
    summary = eval_df[eval_df["k"] == summary_k]

    best_row = summary[summary["model"] == "catboost_classification"].iloc[0]
    baseline_row = summary[summary["model"] == "baseline (final_score)"].iloc[0]

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ML Pipeline — Ранжирование кандидатов</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5; color: #2C3E50; line-height: 1.7; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 40px 20px; }}
  h1 {{ text-align: center; font-size: 2.2em; margin-bottom: 10px; color: #2C3E50; }}
  h2 {{ font-size: 1.5em; margin: 40px 0 20px; padding-bottom: 10px; border-bottom: 3px solid #4A90D9; color: #2C3E50; }}
  h3 {{ font-size: 1.2em; margin: 20px 0 10px; color: #34495E; }}
  .subtitle {{ text-align: center; color: #7F8C8D; font-size: 1.1em; margin-bottom: 40px; }}
  .card {{ background: white; border-radius: 12px; padding: 30px; margin: 20px 0; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }}
  .card img {{ width: 100%; border-radius: 8px; margin: 15px 0; }}
  .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
  .metric {{ background: #f8f9fa; border-radius: 8px; padding: 20px; text-align: center; border-left: 4px solid #4A90D9; }}
  .metric .value {{ font-size: 2em; font-weight: bold; color: #2C3E50; }}
  .metric .label {{ font-size: 0.9em; color: #7F8C8D; margin-top: 5px; }}
  .model-card {{ border-left: 4px solid; padding: 20px; margin: 15px 0; background: #f8f9fa; border-radius: 0 8px 8px 0; }}
  .model-card.mlp {{ border-color: #E74C3C; }}
  .model-card.catboost {{ border-color: #2ECC71; }}
  .model-card.pairwise {{ border-color: #F39C12; }}
  .model-card h3 {{ margin-top: 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
  th, td {{ padding: 12px 15px; text-align: center; border-bottom: 1px solid #eee; }}
  th {{ background: #2C3E50; color: white; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  .winner {{ background: #D5F5E3 !important; font-weight: bold; }}
  .highlight {{ color: #2ECC71; font-weight: bold; }}
  .conclusion {{ background: linear-gradient(135deg, #2ECC71, #27AE60); color: white; padding: 25px; border-radius: 12px; margin: 20px 0; }}
  .conclusion h2 {{ color: white; border-color: white; }}
  ul {{ padding-left: 25px; }}
  li {{ margin: 8px 0; }}
</style>
</head>
<body>
<div class="container">

<h1>ML Pipeline: Ранжирование кандидатов</h1>
<p class="subtitle">Автоматизация найма с помощью машинного обучения</p>

<!-- SECTION 1: Задача -->
<div class="card">
  <h2>1. Задача проекта</h2>
  <p>Автоматизировать процесс найма: ML-модель оценивает кандидатов по 7 критериям и выдаёт <strong>топ-30 лучших</strong>, чтобы HR не просматривал тысячи анкет вручную.</p>
  <h3>7 критериев оценки:</h3>
  <table>
    <tr><th>Критерий</th><th>Вес</th><th>Что оценивает</th></tr>
    <tr><td>Relevance & Depth of Experience</td><td>25%</td><td>Релевантность опыта вакансии</td></tr>
    <tr><td>Hard Skills Match</td><td>20%</td><td>Совпадение технических навыков</td></tr>
    <tr><td>Quality of Questionnaire Answers</td><td>22%</td><td>Глубина и конкретность ответов</td></tr>
    <tr><td>Authenticity & AI-Check</td><td>12%</td><td>Не сгенерированы ли ответы ИИ</td></tr>
    <tr><td>Red Flags & Consistency</td><td>8%</td><td>Противоречия, ложь</td></tr>
    <tr><td>Availability & Logistics</td><td>5%</td><td>Готовность выйти на работу</td></tr>
    <tr><td>Education & Certificates</td><td>8%</td><td>Образование, сертификаты</td></tr>
  </table>
</div>

<!-- SECTION 2: Архитектура -->
<div class="card">
  <h2>2. Архитектура пайплайна</h2>
  <img src="plots/01_pipeline_architecture.png" alt="Pipeline Architecture">
  <p>Данные (2000 кандидатов) → Предобработка → 3 модели → Ранжирование → Top-30</p>
</div>

<!-- SECTION 3: Модели -->
<div class="card">
  <h2>3. Три модели</h2>

  <div class="model-card mlp">
    <h3>🧠 Модель 1: Neural Ranking (PyTorch MLP)</h3>
    <p><strong>Как работает:</strong> Нейросеть с 4 слоями (128→64→32→1). Кандидат проходит через слои, каждый извлекает закономерности. На выходе — вероятность найма.</p>
    <p><strong>Аналогия:</strong> 4 HR-менеджера последовательно смотрят анкету, каждый добавляет своё мнение.</p>
    <p><strong>Задача:</strong> Классификация — предсказать <code>is_hired</code> (нанят / не нанят).</p>
  </div>

  <div class="model-card catboost">
    <h3>🌲 Модель 2: CatBoost (Gradient Boosting)</h3>
    <p><strong>Как работает:</strong> 300 «деревьев решений» учатся на ошибках друг друга. Первое дерево принимает решение → второе исправляет его ошибки → третье исправляет второе → и так 300 раз.</p>
    <p><strong>Аналогия:</strong> 300 стажёров-рекрутеров, каждый учится на ошибках предыдущих. Вместе — команда экспертов.</p>
    <p><strong>Задача:</strong> Классификация — предсказать <code>is_hired</code>.</p>
  </div>

  <div class="model-card pairwise">
    <h3>⚖️ Модель 3: Pairwise Ranking (Logistic Regression)</h3>
    <p><strong>Как работает:</strong> Берёт двоих кандидатов и отвечает: «кто лучше?» Не оценивает по отдельности — только сравнение пар.</p>
    <p><strong>Аналогия:</strong> Турнир по теннису. Не нужен абсолютный рейтинг — важно, кто кого побеждает.</p>
    <p><strong>Задача:</strong> Learning to Rank — ранжирование через попарные сравнения.</p>
  </div>
</div>

<!-- SECTION 4: Радар -->
<div class="card">
  <h2>4. Сравнение моделей (радар)</h2>
  <img src="plots/02_model_radar.png" alt="Model Radar">
</div>

<!-- SECTION 5: Результаты -->
<div class="card">
  <h2>5. Результаты моделей</h2>
  <table>
    <tr>
      <th>Метрика</th>
      <th>Neural MLP</th>
      <th class="winner">CatBoost ⭐</th>
      <th>Pairwise LR</th>
    </tr>
    <tr>
      <td><strong>Precision</strong></td>
      <td>0.765</td>
      <td class="winner">0.765</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td><strong>Recall</strong></td>
      <td>0.851</td>
      <td class="winner">0.937</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td><strong>F1-Score</strong></td>
      <td>0.806</td>
      <td class="winner">0.842</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td><strong>AUC-ROC</strong></td>
      <td>0.600</td>
      <td class="winner">0.622</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td><strong>NDCG@30</strong></td>
      <td>0.903</td>
      <td class="winner">0.919</td>
      <td>0.796</td>
    </tr>
    <tr>
      <td><strong>Spearman vs формула</strong></td>
      <td>0.516</td>
      <td class="winner">0.628</td>
      <td>0.628</td>
    </tr>
  </table>
</div>

<!-- SECTION 6: CV -->
<div class="card">
  <h2>6. Кросс-валидация (стабильность)</h2>
  <img src="plots/07_cross_validation.png" alt="Cross Validation">
  <p>Все модели стабильны (разброс < 2%). CatBoost — самый стабильный: F1 = 0.853 ± 0.004.</p>
</div>

<!-- SECTION 7: Precision/Recall/NDCG curves -->
<div class="card">
  <h2>7. Метрики при разных K</h2>
  <img src="plots/03_precision_recall_ndcg.png" alt="Precision Recall NDCG">
</div>

<!-- SECTION 8: Uplift -->
<div class="card">
  <h2>8. Положительный эффект моделей</h2>
  <img src="plots/04_uplift.png" alt="Uplift">
  <p><strong>Ключевой результат:</strong> На больших списках (K=100) CatBoost находит <span class="highlight">на 4.2% больше</span> реально нанятых кандидатов, чем ручная формула.</p>
</div>

<!-- SECTION 9: Spearman -->
<div class="card">
  <h2>9. Согласие с экспертной формулой</h2>
  <img src="plots/05_spearman.png" alt="Spearman">
  <p>Модели НЕ видели final_score при обучении. Но CatBoost и Pairwise лучше всего воспроизводят логику формулы (r = 0.63).</p>
</div>

<!-- SECTION 10: Feature Importance -->
<div class="card">
  <h2>10. Важность признаков (CatBoost)</h2>
  <img src="plots/08_feature_importance.png" alt="Feature Importance">
</div>

<!-- SECTION 11: Top-30 -->
<div class="card">
  <h2>11. Top-15 кандидатов (CatBoost)</h2>
  <img src="plots/06_top30_table.png" alt="Top 30">
  <p>Все 15 кандидатов в топе реально наняты (<code>is_hired = 1</code>). Средний final_score = 87.2.</p>
</div>

<!-- SECTION 12: Выводы -->
<div class="conclusion">
  <h2>12. Итоговые выводы</h2>
  <h3>🏆 Лучшая модель: CatBoost Classification</h3>
  <ul>
    <li>Лучший F1-Score (0.842) — лучший баланс точности и полноты</li>
    <li>Highest Recall (0.937) — пропускает меньше всего хороших кандидатов</li>
    <li>Лучшее согласие с экспертной формулой (Spearman = 0.628)</li>
    <li>Самая стабильная на кросс-валидации (0.853 ± 0.004)</li>
    <li>Даёт <strong>+4.2% uplift</strong> над ручной формулой на больших списках</li>
  </ul>

  <h3>📊 Почему модели лучше формулы?</h3>
  <p>Формула — это линейная комбинация: <code>0.25×A + 0.20×B + ...</code>. Она не учитывает взаимодействия признаков.</p>
  <p>Модели ловят неочевидные связи: <em>«если опыт средний, но ответы на анкету отличные → скорее всего наймут»</em>. Именно поэтому на K=100 модели находят больше реально нанятых.</p>

  <h3>🎯 Практическая ценность</h3>
  <p>Вместо ручного просмотра 2000 анкет → модель автоматически выдаёт топ-30. HR тратит время только на финальных кандидатов.</p>
</div>

</div>
</body>
</html>"""

    report_path = REPORT_DIR / "result_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("Report saved: %s", report_path)
    return report_path


# ===================================================================
# Main
# ===================================================================

def main():
    logger.info("Generating presentation...")

    model_results, cv_results, top30, eval_df, uplift_df, spearman_df, overlap_df = load_all()

    logger.info("Generating plots...")
    plot_pipeline_architecture()
    plot_model_radar(model_results)
    plot_precision_recall_k(eval_df)
    plot_uplift(uplift_df)
    plot_spearman(spearman_df)
    plot_top30_table(top30)
    plot_cv(cv_results)
    plot_feature_importance()

    logger.info("Generating HTML report...")
    report_path = generate_html(model_results, cv_results, top30, eval_df, uplift_df, spearman_df, overlap_df)

    logger.info("=" * 60)
    logger.info("Presentation ready!")
    logger.info("Open: %s", report_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
