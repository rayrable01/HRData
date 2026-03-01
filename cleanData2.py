import pandas as pd
import numpy as np

# Читаем датасет
df = pd.read_csv("./data/IT_vacancies.csv")

print("Исходный размер:", df.shape)

# =========================
# Очистка зарплаты
# =========================
# Оставляем строки, где хотя бы одна зарплата есть
df = df.dropna(subset=["salary_from", "salary_to"], how="all")

# Пропуски заменяем
df["salary_from"] = df["salary_from"].fillna(0)
df["salary_to"] = df["salary_to"].fillna(df["salary_from"])

# Мягкий порог: оставляем вакансии с зарплатой больше 0
df = df[(df["salary_from"] > 0) | (df["salary_to"] > 0)]

# Средняя зарплата
df["avg_salary"] = (df["salary_from"] + df["salary_to"]) / 2

# =========================
# Валюта
# =========================
df["salary_currency"] = df["salary_currency"].fillna("Unknown")

# =========================
# Пропуски по важным колонкам
# =========================
# Для анализа ключевых навыков, опыта и тестов можно заполнить 'Unknown' или False
df["experience"] = df["experience"].fillna("Unknown")
df["key_skills"] = df["key_skills"].fillna("")
df["test_required"] = df["test_required"].fillna(False)

# =========================
# Даты
# =========================
# Преобразуем даты в datetime
for col in ["created_at", "published_at", "initial_created_at"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# =========================
# Проверка пропусков после очистки
# =========================
print("После очистки:", df.shape)
print(df.isnull().sum())

# =========================
# Сохраняем чистый датасет
# =========================
df.to_csv("./data/IT_vacancies_clean.csv", index=False)
print("Сохранено: vacancies_clean.csv")