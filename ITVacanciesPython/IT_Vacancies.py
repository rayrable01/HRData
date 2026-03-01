import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")

# ---------------------------
# 1. Загружаем неочищенный датасет
# ---------------------------
df = pd.read_csv(r'D:\Work\HRData\data\IT_vacancies.csv')

# ---------------------------
# 2. Time-to-Hire (proxy)
# ---------------------------
# Если нет даты выхода сотрудника, используем разницу между initial_created_at и published_at
df['initial_created_at'] = pd.to_datetime(df['initial_created_at'], errors='coerce')
df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
df['time_to_hire_days'] = (df['published_at'] - df['initial_created_at']).dt.days
# Выбрасываем отрицательные или слишком большие значения (ненадежные)
df['time_to_hire_days'] = df['time_to_hire_days'].apply(lambda x: x if 0 <= x <= 365 else np.nan)

# ---------------------------
# 3. Precision Shortlist (proxy: вакансии с тестом / всего вакансий)
# ---------------------------
precision_shortlist = df['test_required'].value_counts(normalize=True).get(True, 0) * 100

# ---------------------------
# 4. Recall (proxy: вакансии с ≥1 ключевым навыком / все вакансии)
# ---------------------------
recall = (df['count_key_skills'] > 0).sum() / len(df) * 100

# ---------------------------
# 5. Время обработки (нагрузка на HR, proxy через среднее количество ключевых навыков)
# ---------------------------
avg_processing_time = df['count_key_skills'].mean()  # proxy: HR тратит время на навыки

# ---------------------------
# 6. Cost-per-Review (proxy через среднюю зарплату)
# ---------------------------
# Если есть зарплата_from и salary_to, берем среднее
df['salary_mean'] = df[['salary_from','salary_to']].mean(axis=1)
cost_per_review = df['salary_mean'].mean()  # proxy: средняя зарплата как “стоимость обработки”

# ---------------------------
# 7. Вывод baseline
# ---------------------------
baseline = {
    'Time_to_Hire_mean_days': df['time_to_hire_days'].mean(),
    'Time_to_Hire_median_days': df['time_to_hire_days'].median(),
    'Precision_Shortlist_pct': precision_shortlist,
    'Recall_pct': recall,
    'Avg_processing_time_per_vacancy': avg_processing_time,
    'Cost_per_Review_proxy': cost_per_review
}

print("=== Baseline KPI (с proxy) ===")
for k, v in baseline.items():
    print(f"{k}: {v}")

# ---------------------------
# 8. Визуализация
# ---------------------------

# Time-to-Hire 
plt.figure(figsize=(10,5))
sns.histplot(df['time_to_hire_days'].dropna(), bins=50, kde=True)
plt.xlim(0, 50)  # ограничиваем ось X 0-50 дней
plt.title("Распределение Time-to-Hire (дней, ограничено 0-50)")
plt.xlabel("Дни")
plt.ylabel("Количество вакансий")
plt.show()

# Precision Shortlist
plt.figure(figsize=(6,4))
sns.countplot(x='test_required', data=df)
plt.title("Наличие теста (proxy Precision Shortlist)")
plt.ylabel("Количество вакансий")
plt.show()

# Recall по ключевым навыкам
plt.figure(figsize=(6,4))
sns.histplot(df['count_key_skills'], bins=30, kde=True)
plt.title("Распределение количества ключевых навыков (proxy Recall)")
plt.xlabel("Ключевые навыки")
plt.ylabel("Количество вакансий")
plt.show()
