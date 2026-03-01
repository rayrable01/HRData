import pandas as pd
import great_expectations as ge

# =========================
# 1. Загружаем датасет
# =========================
df = pd.read_csv(r'D:\Work\HRData\data\vacancies_clean.csv')

# Преобразуем в объект GE
ge_df = ge.from_pandas(df)

# =========================
# 2. Создание набора ожиданий
# =========================
# Простейшие проверки, которые можно расширять
ge_df.expect_table_row_count_to_be_between(min_value=1)
ge_df.expect_column_values_to_not_be_null("id")
ge_df.expect_column_values_to_be_of_type("avg_salary", "float64")
ge_df.expect_column_values_to_be_between("avg_salary", min_value=0)
ge_df.expect_column_values_to_be_in_set("experience", ["Нет опыта", "От 1 года до 3 лет", "От 3 до 6 лет", "Более 6 лет", None])

# =========================
# 3. Генерация Data Docs (HTML отчет)
# =========================
# Инициализация local data docs
context = ge.get_context()

# Создание Expectation Suite
suite_name = "it_vacancies_suite"
suite = context.create_expectation_suite(suite_name, overwrite_existing=True)
ge_df.save_expectation_suite(discard_failed_expectations=False, suite_name=suite_name)

# Проверка данных
results = ge_df.validate(expectation_suite=suite_name)

# Сохранение HTML отчета
context.build_data_docs()
docs_url = context.get_docs_sites_urls()
print("Отчет сгенерирован, открой локально:", docs_url)