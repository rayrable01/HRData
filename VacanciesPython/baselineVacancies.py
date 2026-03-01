import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import ast
from collections import Counter

sns.set_theme(style="whitegrid")

# =========================
# Чтение датасета
# =========================
df = pd.read_csv(r'D:\Work\HRData\data\vacancies_clean.csv')

# =========================
# Распределение зарплат
# =========================
plt.figure(figsize=(10,5))
sns.histplot(df['avg_salary'], kde=True, bins=30)
plt.title("Распределение средней зарплаты")
plt.xlabel("Средняя зарплата")
plt.ylabel("Количество вакансий")
plt.tight_layout()
plt.show()

# =========================
# Вакансии по опыту
# =========================
plt.figure(figsize=(12,6))
experience_counts = df['experience'].value_counts()
sns.barplot(x=experience_counts.index, y=experience_counts.values)
plt.title("Количество вакансий по опыту")
plt.xlabel("Опыт")
plt.ylabel("Количество вакансий")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# =========================
# Топ вакансий
# =========================
plt.figure(figsize=(12,5))
top_jobs = df['name'].value_counts().head(10)
sns.barplot(x=top_jobs.values, y=top_jobs.index)
plt.title("Топ 10 вакансий")
plt.xlabel("Количество вакансий")
plt.ylabel("Вакансия")
plt.tight_layout()
plt.show()

# =========================
# Топ навыков
# =========================
tags_series = df['tags'].apply(lambda x: ast.literal_eval(x) if x != '[]' else [])
all_tags = [tag for sublist in tags_series for tag in sublist]
top_tags = Counter(all_tags).most_common(10)

plt.figure(figsize=(12,5))
sns.barplot(x=[t[1] for t in top_tags], y=[t[0] for t in top_tags])
plt.title("Топ 10 навыков")
plt.xlabel("Количество упоминаний")
plt.ylabel("Навык")
plt.tight_layout()
plt.show()

# =========================
# Базовые метрики (proxy baseline)
# =========================
baseline = {
    "avg_salary_mean": df['avg_salary'].mean(),
    "avg_salary_median": df['avg_salary'].median(),
    "vacancies_total": df.shape[0],
    "unique_positions": df['name'].nunique(),
    "unique_skills": len(all_tags),
    "experience_distribution": df['experience'].value_counts(normalize=True).to_dict()
}

print("Baseline показатели:")
for k,v in baseline.items():
    print(f"{k}: {v}")