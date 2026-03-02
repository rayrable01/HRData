import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'

df = pd.read_csv(r'D:\Work\HRData\data\IT_vacancies.csv')

df['initial_created_at'] = pd.to_datetime(df['initial_created_at'], errors='coerce')
df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
df['time_to_hire'] = (df['published_at'] - df['initial_created_at']).dt.days
df['time_to_hire'] = df['time_to_hire'].apply(lambda x: x if 0 <= x <= 365 else np.nan)

df['skills_range'] = df['count_key_skills'].apply(
    lambda x: '0' if x == 0 else ('1-3' if x <= 3 else ('4-6' if x <= 6 else ('7-10' if x <= 10 else '11+')))
)

skills_order = ['0', '1-3', '4-6', '7-10', '11+']
skills_palette = {'0': '#95a5a6', '1-3': '#2ecc71', '4-6': '#3498db', '7-10': '#f39c12', '11+': '#e74c3c'}

df_clean = df.dropna(subset=['time_to_hire'])

fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(x='skills_range', y='time_to_hire', data=df_clean, ax=ax, 
            order=skills_order, palette=skills_palette, width=0.6)
ax.set_title('Зависимость времени найма от количества навыков', fontsize=14, fontweight='bold')
ax.set_xlabel('Количество навыков', fontsize=12)
ax.set_ylabel('Время найма (дней)', fontsize=12)
ax.set_ylim(0, 40)

plt.tight_layout()
plt.savefig('time_to_hire_by_skills.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ: ВРЕМЯ НАЙМА ОТ КОЛИЧЕСТВА НАВЫКОВ")
print("="*60)

summary = df_clean.groupby('skills_range')['time_to_hire'].agg(['count', 'mean', 'median', 'std']).reindex(skills_order)
summary.columns = ['n', 'среднее', 'медиана', 'std']
print("\n📊 Статистика по группам:")
print(summary.round(2).to_string())

groups = [df_clean[df_clean['skills_range'] == s]['time_to_hire'].dropna().values for s in skills_order]
stat, p = stats.kruskal(*groups)
print(f"\n📈 Тест Kruskal-Wallis: H={stat:.2f}, p={p:.4f}")
