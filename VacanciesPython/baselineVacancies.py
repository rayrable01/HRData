import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import ast
from collections import Counter
import numpy as np
from scipy import stats

sns.set_theme(style="whitegrid")

df = pd.read_csv(r'D:\Work\HRData\data\vacancies_clean.csv')

# =========================
# Распределение зарплат
# =========================
plt.figure(figsize=(10,5))
sns.histplot(df['avg_salary'], kde=True, bins=30, color='steelblue')
median_salary = df['avg_salary'].median()
plt.axvline(median_salary, color='red', linestyle='--', linewidth=2, label=f'Медиана: {median_salary:,.0f} ₽')
plt.title("Распределение уровня зарплат")
plt.xlabel("Уровень зарплат (₽)")
plt.ylabel("Количество вакансий")
plt.legend()
plt.tight_layout()
plt.show()

print(f"Медиана зарплаты: {median_salary:,.0f} ₽")
print(f"Среднее зарплаты: {df['avg_salary'].mean():,.0f} ₽")

# =========================
# Bootstrap анализ для зарплат
# =========================
np.random.seed(42)
n_bootstrap = 10000
bootstrap_medians = []
for _ in range(n_bootstrap):
    sample = np.random.choice(df['avg_salary'].dropna(), size=len(df), replace=True)
    bootstrap_medians.append(np.median(sample))

ci_lower = np.percentile(bootstrap_medians, 2.5)
ci_upper = np.percentile(bootstrap_medians, 97.5)

print(f"\n=== Bootstrap анализ медианы зарплаты (n={n_bootstrap}) ===")
print(f"95% доверительный интервал: [{ci_lower:,.0f}, {ci_upper:,.0f}] ₽")
print(f"Стандартная ошибка: {np.std(bootstrap_medians):,.0f} ₽")

plt.figure(figsize=(8, 5))
sns.histplot(bootstrap_medians, bins=50, kde=True, color='steelblue')
plt.axvline(ci_lower, color='red', linestyle='--', label=f'2.5%: {ci_lower:,.0f}')
plt.axvline(ci_upper, color='red', linestyle='--', label=f'97.5%: {ci_upper:,.0f}')
plt.axvline(median_salary, color='green', linestyle='-', linewidth=2, label=f'Медиана: {median_salary:,.0f}')
plt.title("Bootstrap распределение медианы зарплат")
plt.xlabel("Медиана зарплаты (₽)")
plt.ylabel("Частота")
plt.legend()
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