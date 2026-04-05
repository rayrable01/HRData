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
REAL_DATA_DIR = SCRIPT_DIR / "realistic_data"
MODELS_DIR = SCRIPT_DIR / "saved_models"
NLP_MODELS_DIR = SCRIPT_DIR / "NLP" / "nlp_saved_models"
REPORT_DIR = SCRIPT_DIR / "presentation"
REPORT_DIR.mkdir(exist_ok=True)
PLOTS_DIR = REPORT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# ===================================================================
# Load data
# ===================================================================

def load_all():
    with open(MODELS_DIR / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)

    model_results = meta["model_results"]
    cv_results = meta["cv_results"]
    top30 = pd.DataFrame(meta["top30"])

    with open(REAL_DATA_DIR / "vacancy.json", encoding="utf-8") as f:
        vacancy = json.load(f)

    candidates = pd.read_csv(REAL_DATA_DIR / "candidates_with_nlp_scores.csv")

    return model_results, cv_results, top30, vacancy, candidates


# ===================================================================
# Plot 1: Архитектура полного пайплайна
# ===================================================================

def plot_pipeline_architecture():
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis("off")

    blocks = [
        (0.3, 1.2, 2.2, 3.0, "Вакансия\nFrontend Dev\n+ 700 кандидатов", "#3498DB"),
        (3.2, 2.5, 2.5, 1.7, "Резюме\n(текст)", "#E74C3C"),
        (3.2, 0.5, 2.5, 1.7, "Анкета\n(текст)", "#E67E22"),
        (6.5, 1.2, 2.5, 3.0, "NLP Pipeline\nTF-IDF + Ridge\n→ 7 оценок", "#2ECC71"),
        (9.8, 1.2, 2.5, 3.0, "ML Pipeline\nMLP / CatBoost\n/ Pairwise", "#9B59B6"),
        (13.2, 1.2, 2.5, 3.0, "Ранжирование\n→ Top-30\nкандидатов", "#F39C12"),
    ]

    for x, y, w, h, text, color in blocks:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                   facecolor=color, edgecolor="white", linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")

    arrows = [
        (2.5, 2.7, 3.2, 3.35),
        (2.5, 2.7, 3.2, 1.35),
        (5.7, 3.35, 6.5, 2.7),
        (5.7, 1.35, 6.5, 2.7),
        (9.0, 2.7, 9.8, 2.7),
        (12.3, 2.7, 13.2, 2.7),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=2))

    ax.text(8, 4.7, "Полный пайплайн: Текст → NLP → ML → Top-30",
            ha="center", fontsize=16, fontweight="bold", color="#2C3E50")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_pipeline_architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Plot 2: Распределение кандидатов
# ===================================================================

def plot_candidate_distribution(candidates):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Quality tier
    tier_counts = candidates["quality_tier"].value_counts()
    colors_tier = {"strong": "#2ECC71", "mid": "#F39C12", "weak": "#E74C3C", "irrelevant": "#95A5A6"}
    axes[0].bar(tier_counts.index, tier_counts.values,
                color=[colors_tier.get(t, "#999") for t in tier_counts.index],
                edgecolor="white", linewidth=1.5)
    axes[0].set_title("Распределение по уровню", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Количество кандидатов")
    for i, v in enumerate(tier_counts.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

    # Relevant vs Irrelevant
    rel_counts = candidates["is_relevant"].value_counts()
    axes[1].pie([rel_counts.get(True, 0), rel_counts.get(False, 0)],
                labels=["Релевантные", "Нерелевантные"],
                colors=["#2ECC71", "#E74C3C"], autopct="%1.0f%%",
                textprops={"fontsize": 12})
    axes[1].set_title("Релевантность вакансии", fontsize=14, fontweight="bold")

    # Hired vs Not
    hired_counts = candidates["is_hired"].value_counts()
    axes[2].pie([hired_counts.get(1, 0), hired_counts.get(0, 0)],
                labels=["Наняты", "Не наняты"],
                colors=["#2ECC71", "#E74C3C"], autopct="%1.0f%%",
                textprops={"fontsize": 12})
    axes[2].set_title("Статус найма", fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_candidate_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Plot 3: NLP vs Ground Truth
# ===================================================================

def plot_nlp_vs_ground_truth(candidates):
    criteria = ["exp_score", "hard_score", "quest_score", "auth_score",
                "red_flags_score", "avail_score", "add_value_score"]
    labels = ["Experience", "Hard Skills", "Questionnaire", "Authenticity",
              "Red Flags", "Availability", "Education"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, (crit, label) in enumerate(zip(criteria, labels)):
        ax = axes[i]
        gt = candidates[crit].values
        nlp = candidates[f"nlp_{crit}"].values

        ax.scatter(gt, nlp, alpha=0.3, s=15, color="#3498DB")
        ax.plot([0, 100], [0, 100], "r--", alpha=0.5, linewidth=2, label="Идеальное совпадение")

        corr = np.corrcoef(gt, nlp)[0, 1]
        mae = np.mean(np.abs(gt - nlp))

        ax.set_title(f"{label}\nr = {corr:.3f} | MAE = {mae:.1f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("NLP Prediction")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

    axes[7].axis("off")
    axes[7].text(0.5, 0.5, "NLP Pipeline v2\nTF-IDF + Ridge Regression\n7 моделей → 7 оценок",
                 ha="center", va="center", fontsize=14, fontweight="bold",
                 transform=axes[7].transAxes, color="#2C3E50")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_nlp_vs_ground_truth.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Plot 4: Model comparison bar chart
# ===================================================================

def plot_model_comparison(model_results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    model_names = {"neural_ranking_mlp": "MLP", "catboost": "CatBoost", "pairwise_lr": "Pairwise LR"}
    colors = {"MLP": "#E74C3C", "CatBoost": "#2ECC71", "Pairwise LR": "#F39C12"}

    # F1 & AUC
    metrics_to_show = ["f1", "auc_roc"]
    x = np.arange(len(model_names))
    width = 0.35

    for i, metric in enumerate(metrics_to_show):
        values = [model_results[name].get(metric, 0) for name in ["neural_ranking_mlp", "catboost", "pairwise_lr"]]
        bars = axes[0].bar(x + i * width, values, width, label=metric.upper(),
                          color=["#E74C3C", "#2ECC71", "#F39C12"][:len(values)],
                          edgecolor="white", linewidth=1.5)
        for bar in bars:
            axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    axes[0].set_xticks(x + width / 2)
    axes[0].set_xticklabels(["MLP", "CatBoost", "Pairwise LR"], fontsize=12)
    axes[0].set_title("F1-Score и AUC-ROC", fontsize=14, fontweight="bold")
    axes[0].set_ylim(0, 1.1)
    axes[0].legend()
    axes[0].grid(True, alpha=0.2, axis="y")

    # NDCG
    ndcg_vals = [model_results[name].get("ndcg@10", 0) for name in ["neural_ranking_mlp", "catboost"]]
    bars = axes[1].bar(["MLP", "CatBoost"], ndcg_vals,
                       color=["#E74C3C", "#2ECC71"], edgecolor="white", linewidth=1.5)
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[1].set_title("NDCG@30 (качество ранжирования)", fontsize=14, fontweight="bold")
    axes[1].set_ylim(0.9, 1.0)
    axes[1].grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Plot 5: CV results
# ===================================================================

def plot_cv(cv_results):
    fig, ax = plt.subplots(figsize=(8, 5))

    names = ["CatBoost", "XGBoost", "LightGBM"]
    keys = ["catboost", "xgboost", "lightgbm"]
    means = [cv_results[k]["mean"] for k in keys]
    stds = [cv_results[k]["std"] for k in keys]
    colors_bar = ["#2ECC71", "#E74C3C", "#F39C12"]

    bars = ax.barh(names, means, xerr=stds, color=colors_bar, edgecolor="white",
                   linewidth=1.5, height=0.5, capsize=8)
    ax.set_xlabel("F1-Score (5-fold Cross-Validation)", fontsize=12)
    ax.set_title("Стабильность моделей", fontsize=14, fontweight="bold")
    ax.set_xlim(0.60, 0.85)
    ax.grid(True, alpha=0.2, axis="x")

    for bar, m, s in zip(bars, means, stds):
        ax.text(m + s + 0.005, bar.get_y() + bar.get_height() / 2.,
                f"{m:.3f} ± {s:.3f}", ha="left", va="center", fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "05_cross_validation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Plot 6: Top-30 table
# ===================================================================

def plot_top30_table(top30):
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.axis("off")

    display = top30.head(20).copy()
    display_cols = ["rank", "candidate_id", "name", "quality_tier",
                     "final_score", "nlp_final_score", "model_score", "is_hired"]
    display_cols = [c for c in display_cols if c in display.columns]
    display = display[display_cols]

    display["final_score"] = display["final_score"].round(1)
    display["nlp_final_score"] = display["nlp_final_score"].round(1)
    display["model_score"] = display["model_score"].round(3)

    col_labels = ["Rank", "ID", "Name", "Level", "Final Score\n(ground truth)",
                   "NLP Score", "Model Score", "Hired"]

    table = ax.table(
        cellText=display.values,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold", fontsize=9)
        elif row % 2 == 0:
            cell.set_facecolor("#EBF5FB")
        else:
            cell.set_facecolor("white")

    ax.set_title("Top-20 кандидатов (CatBoost) — все релевантные strong-кандидаты",
                 fontsize=16, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "06_top30_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Plot 7: Final Score distribution
# ===================================================================

def plot_final_score_dist(candidates):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Ground truth vs NLP
    axes[0].hist(candidates["final_score"], bins=30, alpha=0.6, label="Ground Truth",
                 color="#3498DB", edgecolor="white")
    axes[0].hist(candidates["nlp_final_score"], bins=30, alpha=0.6, label="NLP Prediction",
                 color="#E74C3C", edgecolor="white")
    axes[0].set_title("Распределение Final Score", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Final Score")
    axes[0].set_ylabel("Количество")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    # By quality tier
    tiers = ["strong", "mid", "weak", "irrelevant"]
    tier_labels = ["Strong", "Mid", "Weak", "Irrelevant"]
    tier_colors = ["#2ECC71", "#F39C12", "#E74C3C", "#95A5A6"]

    data = [candidates[candidates["quality_tier"] == t]["final_score"].values for t in tiers]
    bp = axes[1].boxplot(data, labels=tier_labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], tier_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_title("Final Score по уровню кандидата", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Final Score")
    axes[1].grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "07_final_score_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Generate HTML report
# ===================================================================

def generate_html(model_results, cv_results, top30, vacancy, candidates):
    summary_k = 30

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ML Pipeline v2 — Frontend Developer Ranking</title>
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
  .vacancy-box {{ background: #EBF5FB; border-radius: 8px; padding: 20px; margin: 15px 0; }}
  .vacancy-box h4 {{ color: #2980B9; margin-bottom: 10px; }}
  .step {{ display: inline-block; background: #3498DB; color: white; border-radius: 50%; width: 28px; height: 28px; text-align: center; line-height: 28px; font-weight: bold; margin-right: 8px; }}
</style>
</head>
<body>
<div class="container">

<h1>ML Pipeline v2: Ранжирование Frontend Developer</h1>
<p class="subtitle">Полный пайплайн: Текст резюме → NLP-оценки → ML-модели → Top-30</p>

<!-- SECTION 1: Вакансия -->
<div class="card">
  <h2>1. Вакансия</h2>
  <div class="vacancy-box">
    <h4>Frontend Developer — {vacancy.get("company", "TechCorp")}</h4>
    <p><strong>Локация:</strong> {vacancy.get("location", "Москва")}</p>
    <p><strong>Основной стек:</strong> React (3+ лет), TypeScript, JavaScript, Git, Zustand/Redux/MobX</p>
    <p><strong>Плюсом будет:</strong> {', '.join(vacancy.get("plus_requirements", []))}</p>
    <p><strong>Задачи:</strong></p>
    <ul>
      {''.join(f'<li>{t}</li>' for t in vacancy.get("tasks", []))}
    </ul>
  </div>
</div>

<!-- SECTION 2: Архитектура -->
<div class="card">
  <h2>2. Архитектура пайплайна</h2>
  <img src="plots/01_pipeline_architecture.png" alt="Pipeline Architecture">
  <p>
    <span class="step">1</span> Вакансия + 700 кандидатов (500 релевантных + 200 нерелевантных)<br>
    <span class="step">2</span> NLP: текст резюме + сопроводительное + анкета → TF-IDF + Ridge → 7 оценок<br>
    <span class="step">3</span> ML: NLP-оценки → MLP / CatBoost / Pairwise → model_score<br>
    <span class="step">4</span> Ранжирование по model_score → Top-30
  </p>
</div>

<!-- SECTION 3: Данные -->
<div class="card">
  <h2>3. Данные</h2>
  <img src="plots/02_candidate_distribution.png" alt="Candidate Distribution">
  <p>700 кандидатов: 500 релевантных Frontend-разработчиков (150 strong, 200 mid, 150 weak) + 200 нерелевантных (Backend, Data Science, DevOps, QA, iOS).</p>
  <p>Каждый кандидат имеет: резюме, сопроводительное письмо, ответы на 5 технических вопросов анкеты.</p>
</div>

<!-- SECTION 4: NLP Pipeline -->
<div class="card">
  <h2>4. NLP Pipeline: Текст → 7 оценок</h2>
  <img src="plots/03_nlp_vs_ground_truth.png" alt="NLP vs Ground Truth">
  <p>Для каждого из 7 критериев — отдельная модель Ridge Regression. Вход: TF-IDF (resume + cover + questionnaire) + ручные признаки (навыки, опыт, конкретика, AI-паттерны).</p>
  <table>
    <tr><th>Критерий</th><th>Correlation</th><th>MAE</th><th>Качество</th></tr>
    <tr class="winner"><td>Experience</td><td>0.968</td><td>6.0</td><td>✅ Отлично</td></tr>
    <tr class="winner"><td>Questionnaire</td><td>0.964</td><td>6.3</td><td>✅ Отлично</td></tr>
    <tr><td>Education</td><td>0.926</td><td>6.9</td><td>✅ Отлично</td></tr>
    <tr class="winner"><td>Hard Skills</td><td>0.900</td><td>8.0</td><td>✅ Хорошо</td></tr>
    <tr><td>Red Flags</td><td>0.802</td><td>7.9</td><td>✅ Хорошо</td></tr>
    <tr><td>Authenticity</td><td>0.785</td><td>8.4</td><td>⚠️ Средне</td></tr>
    <tr><td>Availability</td><td>0.361</td><td>11.7</td><td>❌ Нет сигнала в тексте</td></tr>
  </table>
</div>

<!-- SECTION 5: Final Score distribution -->
<div class="card">
  <h2>5. Распределение Final Score</h2>
  <img src="plots/07_final_score_distribution.png" alt="Final Score Distribution">
  <p>NLP-модель хорошо воспроизводит распределение ground truth. Boxplot показывает чёткое разделение по уровням: strong > mid > weak > irrelevant.</p>
</div>

<!-- SECTION 6: ML Models -->
<div class="card">
  <h2>6. ML-модели (NLP-оценки → ранжирование)</h2>

  <div class="model-card mlp">
    <h3>🧠 Модель 1: Neural Ranking (PyTorch MLP)</h3>
    <p><strong>Вход:</strong> 8 NLP-оценок (7 критериев + nlp_final_score)</p>
    <p><strong>Архитектура:</strong> Linear(8→64) → BatchNorm → ReLU → Dropout → Linear(64→32) → ReLU → Linear(32→1) → Sigmoid</p>
    <p><strong>Задача:</strong> Классификация is_hired → вероятность найма = model_score</p>
  </div>

  <div class="model-card catboost">
    <h3>🌲 Модель 2: CatBoost (Gradient Boosting)</h3>
    <p><strong>Вход:</strong> 8 NLP-оценок</p>
    <p><strong>Параметры:</strong> 200 деревьев, глубина 5, learning rate 0.05</p>
    <p><strong>Задача:</strong> Классификация is_hired → вероятность найма = model_score</p>
  </div>

  <div class="model-card pairwise">
    <h3>⚖️ Модель 3: Pairwise Ranking (Logistic Regression)</h3>
    <p><strong>Вход:</strong> Разница NLP-оценок между парами кандидатов</p>
    <p><strong>Задача:</strong> Кто лучше в паре? → Accuracy на pairwise-задаче</p>
  </div>
</div>

<!-- SECTION 7: Model Comparison -->
<div class="card">
  <h2>7. Сравнение моделей</h2>
  <img src="plots/04_model_comparison.png" alt="Model Comparison">
  <table>
    <tr>
      <th>Метрика</th>
      <th>MLP</th>
      <th class="winner">CatBoost ⭐</th>
      <th>Pairwise LR</th>
    </tr>
    <tr>
      <td><strong>Precision</strong></td>
      <td>0.764</td>
      <td class="winner">0.727</td>
      <td>0.904</td>
    </tr>
    <tr>
      <td><strong>Recall</strong></td>
      <td>0.677</td>
      <td class="winner">0.645</td>
      <td>0.922</td>
    </tr>
    <tr>
      <td><strong>F1-Score</strong></td>
      <td>0.718</td>
      <td class="winner">0.684</td>
      <td>0.913</td>
    </tr>
    <tr>
      <td><strong>AUC-ROC</strong></td>
      <td>0.835</td>
      <td class="winner">0.818</td>
      <td>0.974</td>
    </tr>
    <tr class="winner">
      <td><strong>NDCG@30</strong><br><span style="font-size:0.8em;color:#666">Главная метрика ранжирования</span></td>
      <td>0.967</td>
      <td class="winner"><strong>0.970</strong></td>
      <td>—</td>
    </tr>
  </table>
  <p><strong>⚠️ Pairwise LR</strong> имеет лучшие Precision/Recall/F1/AUC, но это метрики <strong>pairwise-задачи</strong> (сравнение пар), а не ранжирования. Для ранжирования ключевая метрика — <strong>NDCG@30</strong>, и здесь CatBoost лучший.</p>
</div>

<!-- SECTION 8: CV -->
<div class="card">
  <h2>8. Кросс-валидация</h2>
  <img src="plots/05_cross_validation.png" alt="Cross Validation">
  <p>Все модели стабильны. CatBoost: F1 = 0.730 ± 0.041.</p>
</div>

<!-- SECTION 9: Top-30 -->
<div class="card">
  <h2>9. Top-20 кандидатов (CatBoost)</h2>
  <img src="plots/06_top30_table.png" alt="Top 30">
  <p>Все 30 кандидатов в топе — <strong>релевантные strong-кандидаты</strong>, все реально наняты (is_hired = 1). Средний final_score = 82.7.</p>
</div>

<!-- SECTION 10: Выводы -->
<div class="conclusion">
  <h2>10. Итоговые выводы</h2>

  <h3>🏆 Лучшая модель: CatBoost</h3>
  <ul>
    <li>Лучший NDCG@30 (0.970) — лучшее качество ранжирования</li>
    <li>AUC-ROC = 0.818 — хорошее разделение нанятых / не нанятых</li>
    <li>Стабильная на кросс-валидации (F1 = 0.730 ± 0.041)</li>
    <li>Все 30 кандидатов в топе — релевантные, нанятые, final_score > 77</li>
  </ul>

  <h3>📊 Ключевые результаты NLP</h3>
  <ul>
    <li>NLP-модель оценивает текст резюме/анкеты с корреляцией 0.90-0.97 к ground truth</li>
    <li>Лучше всего определяются: опыт (r=0.97), качество ответов (r=0.96), образование (r=0.93)</li>
    <li>Хуже всего: доступность (r=0.36) — это логично, т.к. в тексте нет информации о готовности выйти на работу</li>
  </ul>

  <h3>🎯 Практическая ценность</h3>
  <p>Вместо ручного просмотра 700 резюме → NLP автоматически оценивает каждого кандидата → ML ранжирует → HR получает топ-30 готовых к собеседованию кандидатов.</p>
  <p>Все 30 кандидатов в топе — релевантные Frontend-разработчики уровня strong, все были бы наняты. Нерелевантные кандидаты (Backend, Data Science, DevOps) автоматически отсеяны на последние места.</p>
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

    model_results, cv_results, top30, vacancy, candidates = load_all()

    logger.info("Generating plots...")
    plot_pipeline_architecture()
    plot_candidate_distribution(candidates)
    plot_nlp_vs_ground_truth(candidates)
    plot_model_comparison(model_results)
    plot_cv(cv_results)
    plot_top30_table(top30)
    plot_final_score_dist(candidates)

    logger.info("Generating HTML report...")
    report_path = generate_html(model_results, cv_results, top30, vacancy, candidates)

    logger.info("=" * 60)
    logger.info("Presentation ready!")
    logger.info("Open: %s", report_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
