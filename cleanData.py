import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

# ------------------------
# Путь к исходному файлу
# ------------------------
path = "./data/vacancies.csv"

# ------------------------
# Чтение данных
# ------------------------
df = pd.read_csv(path)

print("До очистки:", df.shape)

# ------------------------
# Убираем строки с "мусорной" зарплатой
# ------------------------
df = df.dropna(subset=["lower_salary", "upper_salary"])

df = df[
    (df["lower_salary"] >= 10000) &
    (df["upper_salary"] >= 10000)
]

# ------------------------
# Средняя зарплата
# ------------------------
df["avg_salary"] = (df["lower_salary"] + df["upper_salary"]) / 2

# ------------------------
# Валидация схемы через pandera
# ------------------------
schema = DataFrameSchema({
    "name": Column(str, nullable=False),
    "tags": Column(str, nullable=True),
    "link": Column(str, nullable=False),
    "experience": Column(str, nullable=True),
    "lower_salary": Column(float, Check.ge(10000), nullable=False),
    "upper_salary": Column(float, Check.ge(10000), nullable=False),
    "currency": Column(str, Check.isin(["₽","$","€"]), nullable=True),
    "avg_salary": Column(float, Check.ge(10000), nullable=False)
})

validated_df = schema.validate(df, lazy=True)

print("После очистки и валидации:", validated_df.shape)

# ------------------------
# Сохраняем готовый CSV
# ------------------------
validated_df.to_csv("./data/vacancies_clean.csv", index=False)
print("Сохранено: vacancies_clean.csv")