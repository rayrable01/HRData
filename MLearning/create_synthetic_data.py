import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Настройка случайности для воспроизводимости
np.random.seed(42)
random.seed(42)

def generate_synthetic_candidates(n_candidates=1000):
    """
    Генерация синтетического датасета кандидатов с оценками по 7 критериям
    """
    
    # Базовые данные кандидатов
    candidates = []
    
    # Списки для генерации
    positions = ['Data Scientist', 'ML Engineer', 'Backend Developer', 'Frontend Developer', 
                 'DevOps Engineer', 'Data Analyst', 'Product Manager', 'QA Engineer']
    
    experience_levels = ['Junior', 'Middle', 'Senior', 'Lead']
    
    skills_pool = ['Python', 'SQL', 'Machine Learning', 'Deep Learning', 'Docker', 'Kubernetes',
                   'AWS', 'Azure', 'React', 'Vue.js', 'Node.js', 'Java', 'C++', 'TensorFlow',
                   'PyTorch', 'Spark', 'Hadoop', 'Git', 'CI/CD', 'REST API', 'GraphQL']
    
    education_levels = ['Bachelor', 'Master', 'PhD', 'No degree']
    
    for i in range(n_candidates):
        # Базовые характеристики
        position = np.random.choice(positions)
        experience = np.random.choice(experience_levels, p=[0.3, 0.4, 0.2, 0.1])
        
        # Генерация оценок по критериям (0-100 баллов)
        
        # 1. Relevance & Depth of Experience (25%)
        # Зависит от опыта и релевантности позиции
        base_exp_score = {'Junior': 40, 'Middle': 65, 'Senior': 85, 'Lead': 95}[experience]
        exp_score = np.random.normal(base_exp_score, 10)
        exp_score = np.clip(exp_score, 0, 100)
        
        # 2. Hard Skills Match (20%)
        # Зависит от количества и релевантности навыков
        n_skills = np.random.poisson(8) + 3  # 3-15 навыков
        hard_score = min(20 + n_skills * 5 + np.random.normal(0, 10), 100)
        hard_score = np.clip(hard_score, 0, 100)
        
        # 3. Quality & Depth of Questionnaire Answers (22%)
        # Самый важный анкетный критерий
        quest_score = np.random.beta(2, 1) * 100  # Скошенное распределение вправо
        quest_score = np.clip(quest_score, 0, 100)
        
        # 4. Authenticity & AI-Generation Check (12%)
        # Большинство кандидатов честные, но некоторые используют ИИ
        if np.random.random() < 0.15:  # 15% используют ИИ
            auth_score = np.random.normal(30, 15)
        else:
            auth_score = np.random.normal(85, 10)
        auth_score = np.clip(auth_score, 0, 100)
        
        # 5. Red Flags & Consistency (8%)
        # Обратная метрика: 100 - количество красных флагов
        n_red_flags = np.random.poisson(0.5)  # В среднем 0.5 красных флагов
        red_flags_score = 100 - min(n_red_flags * 20, 100)
        
        # 6. Availability & Logistics (5%)
        avail_score = np.random.choice([100, 80, 60, 40], p=[0.6, 0.2, 0.15, 0.05])
        
        # 7. Education, Certificates & Additional Value (8%)
        education_weight = {'No degree': 40, 'Bachelor': 70, 'Master': 85, 'PhD': 95}
        education = np.random.choice(education_levels, p=[0.1, 0.5, 0.3, 0.1])
        base_edu_score = education_weight[education]
        add_value_score = base_edu_score + np.random.normal(0, 15)
        add_value_score = np.clip(add_value_score, 0, 100)
        
        # Расчет FinalScore по формуле
        final_score = (
            0.25 * exp_score +
            0.20 * hard_score +
            0.22 * quest_score +
            0.12 * auth_score +
            0.08 * red_flags_score +
            0.05 * avail_score +
            0.08 * add_value_score
        )
        
        # Генерация навыков (строка с разделителями)
        candidate_skills = random.sample(skills_pool, min(n_skills, len(skills_pool)))
        skills_str = ', '.join(candidate_skills)
        
        # Дата подачи заявки (последние 90 дней)
        application_date = datetime.now() - timedelta(days=np.random.randint(0, 90))
        
        # Статус найма (целевая переменная)
        # Вероятность найма зависит от final_score
        hire_probability = 1 / (1 + np.exp(-(final_score - 60) / 10))
        is_hired = np.random.random() < hire_probability
        
        candidate = {
            'candidate_id': f'CAND_{i:04d}',
            'position_applied': position,
            'experience_level': experience,
            'education': education,
            'skills': skills_str,
            'n_skills': n_skills,
            'application_date': application_date.strftime('%Y-%m-%d'),
            
            # Оценки по критериям
            'exp_score': round(exp_score, 2),
            'hard_score': round(hard_score, 2),
            'quest_score': round(quest_score, 2),
            'auth_score': round(auth_score, 2),
            'red_flags_score': round(red_flags_score, 2),
            'avail_score': round(avail_score, 2),
            'add_value_score': round(add_value_score, 2),
            
            # Итоговые метрики
            'final_score': round(final_score, 2),
            'is_hired': int(is_hired),
            'hire_probability': round(hire_probability, 3),
            
            # Дополнительные синтетические признаки
            'years_experience': {'Junior': 2, 'Middle': 5, 'Senior': 8, 'Lead': 12}[experience] + np.random.randint(-1, 2),
            'has_portfolio': int(np.random.random() < 0.4),
            'has_certifications': int(np.random.random() < 0.3),
            'expected_salary': np.random.normal(150000, 50000) * (1 + {'Junior': 0, 'Middle': 0.3, 'Senior': 0.6, 'Lead': 1.0}[experience]),
            'notice_period_days': np.random.choice([0, 14, 30, 60], p=[0.1, 0.4, 0.4, 0.1]),
        }
        
        candidates.append(candidate)
    
    return pd.DataFrame(candidates)

def generate_vacancy_data(n_vacancies=100):
    """
    Генерация синтетических данных о вакансиях
    """
    vacancies = []
    
    positions = ['Data Scientist', 'ML Engineer', 'Backend Developer', 'Frontend Developer', 
                 'DevOps Engineer', 'Data Analyst', 'Product Manager', 'QA Engineer']
    
    for i in range(n_vacancies):
        position = np.random.choice(positions)
        experience_required = np.random.choice(['Junior', 'Middle', 'Senior', 'Lead'], p=[0.2, 0.5, 0.2, 0.1])
        
        # Зарплатный диапазон зависит от позиции и опыта
        base_salary = {
            'Junior': 80000, 'Middle': 150000, 'Senior': 250000, 'Lead': 350000
        }[experience_required]
        
        salary_from = base_salary * (1 + np.random.normal(0, 0.1))
        salary_to = salary_from * (1.2 + np.random.normal(0, 0.1))
        
        vacancy = {
            'vacancy_id': f'VAC_{i:04d}',
            'position': position,
            'experience_required': experience_required,
            'salary_from': round(salary_from),
            'salary_to': round(salary_to),
            'avg_salary': round((salary_from + salary_to) / 2),
            'currency': 'RUB',
            'is_active': int(np.random.random() < 0.8),
            'n_applications': np.random.poisson(15),
            'n_hired': np.random.binomial(np.random.poisson(15), 0.1),  # 10% успешных наймов
            'created_date': (datetime.now() - timedelta(days=np.random.randint(0, 180))).strftime('%Y-%m-%d'),
            'required_skills': ', '.join(random.sample(['Python', 'SQL', 'Machine Learning', 'Docker', 
                                                       'AWS', 'React', 'Node.js', 'Java'], k=np.random.randint(3, 8))),
            'location': np.random.choice(['Moscow', 'Saint Petersburg', 'Remote', 'Hybrid'], 
                                        p=[0.4, 0.2, 0.3, 0.1]),
            'work_type': np.random.choice(['Full-time', 'Part-time', 'Contract'], p=[0.8, 0.1, 0.1]),
        }
        
        vacancies.append(vacancy)
    
    return pd.DataFrame(vacancies)

def main():
    print("Генерация синтетических данных для ML проекта...")
    
    # Генерация данных кандидатов
    print("Генерация данных кандидатов...")
    candidates_df = generate_synthetic_candidates(2000)  # 2000 кандидатов
    candidates_df.to_csv('MLearning/candidates_data.csv', index=False, encoding='utf-8')
    print(f"Создано {len(candidates_df)} записей кандидатов")
    
    # Генерация данных вакансий
    print("Генерация данных вакансий...")
    vacancies_df = generate_vacancy_data(200)  # 200 вакансий
    vacancies_df.to_csv('MLearning/vacancies_data.csv', index=False, encoding='utf-8')
    print(f"Создано {len(vacancies_df)} записей вакансий")
    
    # Создание датасета для парного ранжирования
    print("Создание датасета для парного ранжирования...")
    create_pairwise_dataset(candidates_df)
    
    # Базовая статистика
    print("\n=== Статистика датасета кандидатов ===")
    print(f"Всего кандидатов: {len(candidates_df)}")
    print(f"Нанято: {candidates_df['is_hired'].sum()} ({candidates_df['is_hired'].mean()*100:.1f}%)")
    print(f"Средний final_score: {candidates_df['final_score'].mean():.2f}")
    print(f"Распределение по опыту:\n{candidates_df['experience_level'].value_counts()}")
    
    print("\n=== Статистика датасета вакансий ===")
    print(f"Всего вакансий: {len(vacancies_df)}")
    print(f"Активных вакансий: {vacancies_df['is_active'].sum()}")
    print(f"Среднее количество заявок: {vacancies_df['n_applications'].mean():.1f}")
    
    print("\nДанные сохранены в:")
    print("- MLearning/candidates_data.csv")
    print("- MLearning/vacancies_data.csv")
    print("- MLearning/pairwise_ranking_data.csv")

def create_pairwise_dataset(candidates_df):
    """
    Создание датасета для парного ранжирования
    Для каждой пары кандидатов определяем, кто лучше
    """
    pairwise_data = []
    
    # Группируем по позициям
    for position in candidates_df['position_applied'].unique():
        position_candidates = candidates_df[candidates_df['position_applied'] == position]
        
        if len(position_candidates) < 2:
            continue
            
        # Создаем пары кандидатов
        candidate_ids = position_candidates['candidate_id'].values
        n_candidates = len(candidate_ids)
        
        # Создаем ограниченное количество пар для каждой позиции
        max_pairs_per_position = min(100, n_candidates * (n_candidates - 1) // 2)
        
        for _ in range(max_pairs_per_position):
            i, j = np.random.choice(range(n_candidates), size=2, replace=False)
            cand1 = position_candidates.iloc[i]
            cand2 = position_candidates.iloc[j]
            
            # Определяем, кто лучше на основе final_score
            if cand1['final_score'] > cand2['final_score']:
                better_candidate = cand1['candidate_id']
                worse_candidate = cand2['candidate_id']
                label = 1  # cand1 лучше cand2
            elif cand1['final_score'] < cand2['final_score']:
                better_candidate = cand2['candidate_id']
                worse_candidate = cand1['candidate_id']
                label = -1  # cand2 лучше cand1
            else:
                continue  # Пропускаем равные пары
            
            pairwise_data.append({
                'position': position,
                'candidate_a': cand1['candidate_id'],
                'candidate_b': cand2['candidate_id'],
                'better_candidate': better_candidate,
                'label': label,
                'score_diff': abs(cand1['final_score'] - cand2['final_score']),
                # Признаки кандидата A
                'a_exp_score': cand1['exp_score'],
                'a_hard_score': cand1['hard_score'],
                'a_quest_score': cand1['quest_score'],
                'a_auth_score': cand1['auth_score'],
                'a_red_flags_score': cand1['red_flags_score'],
                'a_avail_score': cand1['avail_score'],
                'a_add_value_score': cand1['add_value_score'],
                # Признаки кандидата B
                'b_exp_score': cand2['exp_score'],
                'b_hard_score': cand2['hard_score'],
                'b_quest_score': cand2['quest_score'],
                'b_auth_score': cand2['auth_score'],
                'b_red_flags_score': cand2['red_flags_score'],
                'b_avail_score': cand2['avail_score'],
                'b_add_value_score': cand2['add_value_score'],
            })
    
    pairwise_df = pd.DataFrame(pairwise_data)
    pairwise_df.to_csv('MLearning/pairwise_ranking_data.csv', index=False, encoding='utf-8')
    print(f"Создано {len(pairwise_df)} пар для ранжирования")

if __name__ == "__main__":
    main()