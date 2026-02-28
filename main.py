import pandas as pd

df = pd.read_csv('./data/vacancies.csv');

print("Данные загружены успешно!")
print(f"Размер: {df.shape[0]} строк, {df.shape[1]} колонок")
print("\nПервые 3 строки:")
print(df.head(3))

print("\nКолонки:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")