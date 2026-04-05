"""
NLP Pipeline: Текст резюме + сопроводительное + анкета → 7 оценок.

Работает с реалистичными данными для вакансии Frontend Developer.
Для каждого критерия — отдельная модель (Ridge Regression).
"""
from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.resolve()
NLP_DIR = SCRIPT_DIR
REAL_DATA_DIR = SCRIPT_DIR.parent / "realistic_data"
MODELS_DIR = NLP_DIR / "nlp_saved_models"
MODELS_DIR.mkdir(exist_ok=True)

CRITERIA = [
    "exp_score", "hard_score", "quest_score", "auth_score",
    "red_flags_score", "avail_score", "add_value_score",
]

CRITERIA_DESC = {
    "exp_score": "Relevance & Depth of Experience (25%)",
    "hard_score": "Hard Skills Match (20%)",
    "quest_score": "Quality & Depth of Questionnaire Answers (22%)",
    "auth_score": "Authenticity & AI-Generation Check (12%)",
    "red_flags_score": "Red Flags & Consistency (8%)",
    "avail_score": "Availability & Logistics (5%)",
    "add_value_score": "Education, Certificates & Additional Value (8%)",
}

# Стек вакансии Frontend
VACANCY_STACK = {
    "required": ["React", "TypeScript", "JavaScript", "Git", "Zustand", "Redux", "MobX"],
    "plus": ["Vite", "Docker", "Gitlab CI"],
}

# Глобальные векторизаторы
resume_tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2), stop_words=None)
cover_tfidf = TfidfVectorizer(max_features=200, ngram_range=(1, 2), stop_words=None)
quest_tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2), stop_words=None)
manual_scaler = StandardScaler()
criterion_models = {}


# ===================================================================
# Ручные признаки из текста
# ===================================================================

def extract_manual_features(resume: str, cover: str, questionnaire: str) -> list:
    """Извлекает числовые признаки из текстов кандидата."""
    resume_lower = resume.lower()
    cover_lower = cover.lower()
    quest_lower = questionnaire.lower()
    all_text = resume_lower + " " + cover_lower + " " + quest_lower

    # --- Навыки ---
    required_skills = VACANCY_STACK["required"]
    plus_skills = VACANCY_STACK["plus"]

    req_matched = sum(1 for s in required_skills if s.lower() in all_text)
    plus_matched = sum(1 for s in plus_skills if s.lower() in all_text)

    # --- Опыт ---
    years_match = re.findall(r'(\d+)\s*(лет|год|года|year)', resume_lower)
    years_exp = max([int(y[0]) for y in years_match], default=0)

    seniority_words = ["senior", "lead", "руководил", "архитектур", "спроектировал", "оптимизировал"]
    seniority_count = sum(1 for w in seniority_words if w in all_text)

    # --- Качество ответов ---
    word_count = len(questionnaire.split())
    sentence_count = len(re.split(r'[.!?]+', questionnaire))

    # Конкретика: числа, проценты, метрики
    numbers_count = len(re.findall(r'\d+', questionnaire))
    percent_count = len(re.findall(r'\d+%', questionnaire))
    metrics_words = ["ndcg", "lcp", "мс", "сек", "млн", "тысяч", "k запросов",
                     "покрытие", "test", "coverage", "%"]
    metrics_count = sum(1 for w in metrics_words if w in quest_lower)

    # --- AI-детекция ---
    ai_patterns = [
        "динамично развивающ", "высококвалифицирован", "стремление к совершенству",
        "непрерывному обучению", "комплексным пониманием", "передовых технологий",
        "разносторонним опытом", "оптимальные решения", "в современном",
        "являюсь.*специалистом", "обширным опытом",
    ]
    ai_count = sum(1 for pat in ai_patterns if re.search(pat, all_text))

    # --- Red flags ---
    red_words = ["не смог", "не соответствует", "расхожден", "противореч",
                 "не совпадают", "негативн", "ложь", "обман", "фриланс",
                 "небольшая компания", "стажёр"]
    red_count = sum(1 for w in red_words if w in all_text)

    # --- Образование ---
    edu_score = 0
    if any(w in all_text for w in ["магистр", "магистратур"]):
        edu_score = 3
    elif any(w in all_text for w in ["специалист"]):
        edu_score = 2
    elif any(w in all_text for w in ["бакалавр", "бакалавриат"]):
        edu_score = 1

    # --- CI/CD, Docker ---
    devops_words = ["docker", "ci/cd", "gitlab ci", "deploy", "деплой", "pipeline", "nginx"]
    devops_count = sum(1 for w in devops_words if w in all_text)

    # --- Тестирование ---
    test_words = ["jest", "testing library", "unit-тест", "e2e", "playwright",
                  "cypress", "coverage", "покрытие тестами", "mock"]
    test_count = sum(1 for w in test_words if w in all_text)

    return [
        req_matched,
        plus_matched,
        years_exp,
        seniority_count,
        word_count,
        sentence_count,
        numbers_count,
        percent_count,
        metrics_count,
        ai_count,
        red_count,
        edu_score,
        devops_count,
        test_count,
    ]


def build_features(resume: str, cover: str, questionnaire: str):
    """Строит полный вектор признаков для одного кандидата."""
    tfidf_r = resume_tfidf.transform([resume])
    tfidf_c = cover_tfidf.transform([cover])
    tfidf_q = quest_tfidf.transform([questionnaire])

    manual = np.array([extract_manual_features(resume, cover, questionnaire)])
    manual_scaled = manual_scaler.transform(manual)

    combined = hstack([tfidf_r, tfidf_c, tfidf_q, csr_matrix(manual_scaled)])
    return combined


# ===================================================================
# Обучение
# ===================================================================

def train():
    logger.info("=" * 60)
    logger.info("NLP PIPELINE v2: Обучение на реалистичных данных")
    logger.info("=" * 60)

    df = pd.read_csv(REAL_DATA_DIR / "candidates_realistic.csv")
    logger.info("Loaded %d candidates", len(df))

    # Fit TF-IDF
    resume_tfidf.fit(df["resume_text"])
    cover_tfidf.fit(df["cover_letter"])
    quest_tfidf.fit(df["questionnaire_text"])

    # Fit scaler
    manual_feats = []
    for _, row in df.iterrows():
        manual_feats.append(extract_manual_features(
            row["resume_text"], row["cover_letter"], row["questionnaire_text"]
        ))
    manual_scaler.fit(np.array(manual_feats))

    # Build full feature matrix
    all_features = []
    for _, row in df.iterrows():
        feat = build_features(row["resume_text"], row["cover_letter"], row["questionnaire_text"])
        all_features.append(feat)

    X = vstack(all_features)
    logger.info("Feature matrix shape: %s", X.shape)

    # Train models
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}

    for criterion in CRITERIA:
        logger.info("\n--- %s ---", CRITERIA_DESC[criterion])
        y = df[criterion].values

        model = Ridge(alpha=1.0)
        scores = cross_val_score(model, X, y, cv=kfold, scoring="r2")
        cv_results[criterion] = {"mean": scores.mean(), "std": scores.std()}
        logger.info("CV R²: %.4f ± %.4f", scores.mean(), scores.std())

        model.fit(X, y)
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        logger.info("Train MAE: %.2f | R²: %.4f", mae, r2)

        criterion_models[criterion] = model

    _save_models()

    logger.info("\n" + "=" * 60)
    logger.info("CV Results Summary:")
    for c in CRITERIA:
        logger.info("  %-35s R² = %.4f ± %.4f", c, cv_results[c]["mean"], cv_results[c]["std"])
    logger.info("=" * 60)

    return cv_results


def _save_models():
    with open(MODELS_DIR / "resume_tfidf.pkl", "wb") as f:
        pickle.dump(resume_tfidf, f)
    with open(MODELS_DIR / "cover_tfidf.pkl", "wb") as f:
        pickle.dump(cover_tfidf, f)
    with open(MODELS_DIR / "quest_tfidf.pkl", "wb") as f:
        pickle.dump(quest_tfidf, f)
    with open(MODELS_DIR / "manual_scaler.pkl", "wb") as f:
        pickle.dump(manual_scaler, f)

    for name, model in criterion_models.items():
        with open(MODELS_DIR / f"model_{name}.pkl", "wb") as f:
            pickle.dump(model, f)

    logger.info("Saved all NLP models to %s", MODELS_DIR)


def load_models():
    global resume_tfidf, cover_tfidf, quest_tfidf, manual_scaler, criterion_models

    with open(MODELS_DIR / "resume_tfidf.pkl", "rb") as f:
        resume_tfidf = pickle.load(f)
    with open(MODELS_DIR / "cover_tfidf.pkl", "rb") as f:
        cover_tfidf = pickle.load(f)
    with open(MODELS_DIR / "quest_tfidf.pkl", "rb") as f:
        quest_tfidf = pickle.load(f)
    with open(MODELS_DIR / "manual_scaler.pkl", "rb") as f:
        manual_scaler = pickle.load(f)

    for criterion in CRITERIA:
        with open(MODELS_DIR / f"model_{criterion}.pkl", "rb") as f:
            criterion_models[criterion] = pickle.load(f)

    logger.info("Loaded all NLP models")


# ===================================================================
# Inference
# ===================================================================

def predict_candidate(resume: str, cover: str, questionnaire: str) -> dict:
    if not criterion_models:
        load_models()

    features = build_features(resume, cover, questionnaire)

    scores = {}
    for criterion in CRITERIA:
        pred = criterion_models[criterion].predict(features)[0]
        pred = float(np.clip(pred, 0, 100))
        scores[criterion] = round(pred, 1)

    scores["final_score"] = round(
        0.25 * scores["exp_score"] +
        0.20 * scores["hard_score"] +
        0.22 * scores["quest_score"] +
        0.12 * scores["auth_score"] +
        0.08 * scores["red_flags_score"] +
        0.05 * scores["avail_score"] +
        0.08 * scores["add_value_score"],
        1
    )

    return scores


# ===================================================================
# Generate NLP scores for all candidates
# ===================================================================

def generate_all_scores():
    """Предсказывает NLP-оценки для всех кандидатов и сохраняет."""
    logger.info("Generating NLP scores for all candidates...")

    if not criterion_models:
        load_models()

    df = pd.read_csv(REAL_DATA_DIR / "candidates_realistic.csv")

    nlp_scores = []
    for _, row in df.iterrows():
        scores = predict_candidate(
            row["resume_text"], row["cover_letter"], row["questionnaire_text"]
        )
        nlp_scores.append(scores)

    scores_df = pd.DataFrame(nlp_scores)

    # Добавляем NLP-оценки к оригинальным данным
    for col in scores_df.columns:
        df[f"nlp_{col}"] = scores_df[col]

    df.to_csv(REAL_DATA_DIR / "candidates_with_nlp_scores.csv", index=False, encoding="utf-8")
    logger.info("Saved NLP scores to %s", REAL_DATA_DIR / "candidates_with_nlp_scores.csv")

    # Сравнение NLP vs ground truth
    logger.info("\nNLP vs Ground Truth comparison:")
    for criterion in CRITERIA:
        gt = df[criterion].values
        nlp = df[f"nlp_{criterion}"].values
        mae = mean_absolute_error(gt, nlp)
        corr = np.corrcoef(gt, nlp)[0, 1]
        logger.info("  %-25s MAE=%.2f | corr=%.3f", criterion, mae, corr)

    return df


# ===================================================================
# Demo
# ===================================================================

def demo():
    logger.info("=" * 60)
    logger.info("NLP DEMO v2: Текст → 7 оценок (Frontend Developer)")
    logger.info("=" * 60)

    if not criterion_models:
        load_models()

    # Сильный кандидат
    logger.info("\n--- Сильный Frontend Developer ---")
    s1 = predict_candidate(
        resume="=== РЕЗЮМЕ ===\nИмя: Александр Иванов\nОпыт: 6 лет\n\nSenior Frontend Developer, Яндекс, 2020–н.в.\n• Разрабатывал архитектуру фронтенд-приложений на React + TypeScript\n• Оптимизировал производительность: снизил время загрузки на 40%\n• Стек: React, TypeScript, Redux, Zustand, Vite, Docker, Gitlab CI\n\nМагистр, МГУ, компьютерные науки",
        cover="Здравствуйте! 6 лет опыта с React и TypeScript. Разрабатывал архитектуру, оптимизировал производительность, менторил джуниоров. Работал с Zustand, Vite, Docker, Gitlab CI. Буду рад обсудить как мой опыт полезен вашей команде.",
        questionnaire="Вопрос 1: Использую React.memo, useMemo, useCallback. Виртуализация через react-window — на проекте с 10K строк снизило рендер с 3с до 200мс. Code splitting через React.lazy — бандл с 1.2MB до 400KB.\n\nВопрос 2: Zustand для большинства проектов, Redux Toolkit для enterprise. Context API только для темы и локали.\n\nВопрос 3: Jest + RTL, покрытие 80%. Playwright для E2E. MSW для моков API.\n\nВопрос 4: React Query для серверного состояния. Автоматический re-fetch, invalidation, optimistic updates.\n\nВопрос 5: Gitlab CI: lint → test → build → deploy. Docker + nginx. Деплой на Kubernetes.",
    )
    _print_scores(s1)

    # Слабый кандидат
    logger.info("\n--- Junior Frontend Developer ---")
    s2 = predict_candidate(
        resume="=== РЕЗЮМЕ ===\nИмя: Дмитрий Петров\nОпыт: 1 год\n\nJunior Frontend Developer, небольшая компания, 2023–н.в.\n• Верстал страницы, писал компоненты на React\n• Стек: React, JavaScript, Git\n\nНеоконченное высшее",
        cover="Здравствуйте! Хочу у вас работать. Знаю React, учусь дальше.",
        questionnaire="Вопрос 1: Слышал про React.memo, но не использовал.\n\nВопрос 2: Использую useState и Context. С Redux не работал.\n\nВопрос 3: Тесты не писал.\n\nВопрос 4: Делал простые fetch-запросы.\n\nВопрос 5: Не деплоил сам. Запускал локально через npm start.",
    )
    _print_scores(s2)

    # Нерелевантный (Backend)
    logger.info("\n--- Backend Developer (нерелевантный) ---")
    s3 = predict_candidate(
        resume="=== РЕЗЮМЕ ===\nИмя: Сергей Козлов\nОпыт: 5 лет\n\nBackend Developer, Яндекс, 2020–н.в.\n• Разрабатывал микросервисы на Python (FastAPI)\n• Работал с PostgreSQL, Redis, RabbitMQ\n• Стек: Python, FastAPI, PostgreSQL, Docker, Kubernetes\n\nМагистр, МГТУ Баумана",
        cover="Здравствуйте! Я Backend Developer с опытом на Python. Хочу попробовать фронтенд. Знаю основы React, но коммерческого опыта нет.",
        questionnaire="Вопрос 1: С фронтендом не работал глубоко. Знаю что React что-то рендерит.\n\nВопрос 2: С React-стейтом не работал.\n\nВопрос 3: Тестировал бэкенд через pytest.\n\nВопрос 4: На бэкенде я сам пишу API.\n\nВопрос 5: Деплоил бэкенд через Docker + Kubernetes.",
    )
    _print_scores(s3)


def _print_scores(scores):
    for criterion, desc in CRITERIA_DESC.items():
        val = scores.get(criterion, 0)
        bar = "█" * int(val // 5) + "░" * (20 - int(val // 5))
        logger.info("  %-35s %5.1f |%s|", desc[:35], val, bar)
    logger.info("  %-35s %5.1f", "FINAL SCORE", scores["final_score"])


if __name__ == "__main__":
    train()
    generate_all_scores()
    demo()
