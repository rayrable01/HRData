"""
Генерация реалистичных данных для вакансии Frontend Developer.

Создаёт:
  - vacancy.json — описание вакансии
  - candidates_realistic.csv — 700 кандидатов (500 релевантных + 200 нерелевантных)
    С полями: resume_text, cover_letter, questionnaire_answers, is_hired, final_score
    + 7 критериев (скрыты от модели, используются для оценки)
"""
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
REAL_DATA_DIR = SCRIPT_DIR / "realistic_data"
REAL_DATA_DIR.mkdir(exist_ok=True)

# ===================================================================
# Вакансия
# ===================================================================

VACANCY = {
    "title": "Frontend Developer",
    "company": "TechCorp",
    "location": "Москва, м. Технопарк (гибрид)",
    "stack": {
        "required": ["React", "TypeScript", "JavaScript", "Git", "Zustand", "Redux", "MobX"],
        "plus": ["Vite", "Docker", "Gitlab CI"],
    },
    "tasks": [
        "Построение и аудит архитектуры фронт-части проектов",
        "Развитие Best Practices",
        "Оптимизация разработки фронтенда",
        "Анализ задач и проектирование под них UI",
        "Разработка компонентов фронтенд системы по заданным спецификациям",
        "Анализ собственного и чужого кода",
    ],
    "requirements": [
        "React от 3х лет",
        "Zustand или знание любого другого стейт менеджера (Redux, MobX)",
        "TypeScript, JavaScript",
        "Git",
        "Ответственность за результат",
        "Деловое общение: уважение к собеседнику, ясность формулировок, соблюдение субординации",
    ],
    "plus_requirements": ["Vite", "Docker", "Gitlab CI"],
    "offers": [
        "Работа в развивающемся продукте, который решает реальные бизнес-задачи",
        "Возможность профессионального роста и влияния на архитектуру решения",
        "Офис в 3 минутах от м. Технопарк (Москва) гибрид",
        "Работа по SCRUM, прозрачные процессы, минимум бюрократии",
        "Регулярные и уютные корпоративные мероприятия",
        "Заработная плата по результатам собеседования",
    ],
    "process": [
        "Неоплачиваемое тестовое задание на этапе отбора",
        "Техническое интервью с разработчиками",
    ],
    "questions": [
        "Как вы оптимизируете рендеринг в React-приложениях? Приведите конкретные примеры из вашего опыта.",
        "Как вы управляете состоянием в крупных приложениях? Когда выбираете Zustand, Redux, а когда Context API?",
        "Как вы тестируете React-компоненты? Какие инструменты используете и что покрываете тестами?",
        "Как вы работаете с API? Как обрабатываете ошибки, загрузку, кэширование данных?",
        "Как вы деплоите фронтенд-приложение? Опишите ваш CI/CD пайплайн.",
    ],
}


# ===================================================================
# Текстовые шаблоны
# ===================================================================

NAMES_FIRST = ["Александр", "Дмитрий", "Максим", "Иван", "Артём", "Кирилл", "Михаил",
               "Даниил", "Егор", "Андрей", "Никита", "Илья", "Алексей", "Сергей",
               "Павел", "Роман", "Владислав", "Тимур", "Марк", "Лев",
               "Анна", "Мария", "Екатерина", "Ольга", "Наталья", "Елена",
               "Дарья", "Алина", "Юлия", "Виктория", "Полина", "Софья", "Ксения"]

NAMES_LAST = ["Иванов(а)", "Смирнов(а)", "Кузнецов(а)", "Попов(а)", "Васильев(а)",
              "Петров(а)", "Соколов(а)", "Михайлов(а)", "Новиков(а)", "Фёдоров(а)",
              "Морозов(а)", "Волков(а)", "Алексеев(а)", "Лебедев(а)", "Семёнов(а)",
              "Егоров(а)", "Павлов(а)", "Козлов(а)", "Степанов(а)", "Николаев(а)"]

UNIS = ["МГУ", "МГТУ им. Баумана", "НИУ ВШЭ", "ИТМО", "МИФИ", "СПбГУ", "УрФУ", "НГУ", "МГТУ Станкин", "РЭУ Плеханова"]

EDUCATION_LEVELS = ["Бакалавр", "Магистр", "Специалист", "Неоконченное высшее", "Без высшего"]

# --- Резюме: опыт ---

EXPERIENCE_STRONG = [
    "Senior Frontend Developer, Яндекс, 2021–н.в.\n"
    "• Разрабатывал архитектуру фронтенд-приложений на React + TypeScript\n"
    "• Оптимизировал производительность: снизил время загрузки на 40%\n"
    "• Внедрил code review и best practices в команде из 8 человек\n"
    "• Стек: React, TypeScript, Redux, Zustand, Vite, Docker, Gitlab CI",

    "Lead Frontend Developer, Сбер, 2020–н.в.\n"
    "• Руководил командой из 5 фронтенд-разработчиков\n"
    "• Спроектировал микрофронтенд-архитектуру для 3 продуктов\n"
    "• Мигрировал проект с JavaScript на TypeScript\n"
    "• Стек: React, TypeScript, MobX, Webpack, Docker, Jenkins",

    "Middle+/Senior Frontend Developer, Тинькофф, 2019–н.в.\n"
    "• Разрабатывал компоненты дизайн-системы на React\n"
    "• Оптимизировал рендеринг: убрал лишние ре-рендеры, внедрил useMemo/useCallback\n"
    "• Настроил CI/CD пайплайн с Gitlab CI\n"
    "• Стек: React, TypeScript, Zustand, Vite, Jest, React Testing Library",
]

EXPERIENCE_MID = [
    "Middle Frontend Developer, Ozon, 2022–н.в.\n"
    "• Разрабатывал UI-компоненты на React + TypeScript\n"
    "• Участвовал в код-ревью, писал unit-тесты\n"
    "• Стек: React, TypeScript, Redux, Git",

    "Frontend Developer, VK, 2021–н.в.\n"
    "• Поддерживал и развивал фронтенд-часть продукта\n"
    "• Работал с REST API, обрабатывал ошибки и состояния загрузки\n"
    "• Стек: React, JavaScript, Redux, Git",

    "Frontend Developer, стартап, 2022–н.в.\n"
    "• Разрабатывал SPA на React с нуля\n"
    "• Настраивал роутинг, работу с API, управление состоянием\n"
    "• Стек: React, JavaScript, Redux Toolkit, Git",
]

EXPERIENCE_WEAK = [
    "Junior Frontend Developer, небольшая компания, 2023–н.в.\n"
    "• Верстал страницы, писал компоненты на React\n"
    "• Базовая работа с API\n"
    "• Стек: React, JavaScript, Git",

    "Стажёр-разработчик, 2023–2024\n"
    "• Помогал с фронтенд-задачами\n"
    "• Изучал React и TypeScript\n"
    "• Стек: React, JavaScript",

    "Фриланс, 2023–н.в.\n"
    "• Делал лендинги и небольшие сайты\n"
    "• React, HTML, CSS, JavaScript",
]

EXPERIENCE_IRRELEVANT = [
    "Backend Developer, Яндекс, 2020–н.в.\n"
    "• Разрабатывал микросервисы на Python (FastAPI)\n"
    "• Работал с PostgreSQL, Redis, RabbitMQ\n"
    "• Стек: Python, FastAPI, PostgreSQL, Docker, Kubernetes",

    "Data Scientist, Сбер, 2021–н.в.\n"
    "• Строил ML-модели для прогнозирования\n"
    "• Работал с большими данными, Spark\n"
    "• Стек: Python, PyTorch, SQL, Spark, Airflow",

    "DevOps Engineer, Тинькофф, 2020–н.в.\n"
    "• Настраивал CI/CD пайплайны, управлял Kubernetes\n"
    "• Автоматизировал инфраструктуру\n"
    "• Стек: Kubernetes, Terraform, Ansible, Docker, Linux",

    "QA Engineer, VK, 2022–н.в.\n"
    "• Писал автотесты на Python + Selenium\n"
    "• Тестировал API, проводил регрессионное тестирование\n"
    "• Стек: Python, Selenium, Pytest, Postman",

    "Mobile Developer (iOS), 2021–н.в.\n"
    "• Разрабатывал iOS-приложения на Swift\n"
    "• Работал с UIKit, SwiftUI\n"
    "• Стек: Swift, SwiftUI, Xcode, Git",
]


# --- Навыки ---

SKILLS_STRONG = ["React", "TypeScript", "JavaScript", "Redux", "Zustand", "HTML", "CSS",
                  "Git", "Vite", "Webpack", "Jest", "React Testing Library", "Docker",
                  "Gitlab CI", "REST API", "GraphQL", "Node.js", "Next.js"]

SKILLS_MID = ["React", "JavaScript", "TypeScript", "Redux", "HTML", "CSS", "Git",
               "REST API", "Webpack", "Jest"]

SKILLS_WEAK = ["React", "JavaScript", "HTML", "CSS", "Git"]

SKILLS_IRRELEVANT_PYTHON = ["Python", "Django", "FastAPI", "PostgreSQL", "SQL", "Docker", "Linux", "Git", "Redis"]
SKILLS_IRRELEVANT_DS = ["Python", "PyTorch", "TensorFlow", "SQL", "Pandas", "NumPy", "Scikit-learn", "Git", "Jupyter"]
SKILLS_IRRELEVANT_DEVOPS = ["Kubernetes", "Docker", "Terraform", "Ansible", "Linux", "CI/CD", "AWS", "Git", "Bash"]
SKILLS_IRRELEVANT_QA = ["Selenium", "Pytest", "Postman", "Python", "SQL", "Git", "Jira", "TestRail"]
SKILLS_IRRELEVANT_IOS = ["Swift", "SwiftUI", "UIKit", "Xcode", "Git", "CocoaPods", "CoreData"]


# --- Сопроводительное письмо ---

COVER_STRONG = [
    "Здравствуйте! Меня заинтересовала вакансия Frontend Developer в вашей компании.\n\n"
    "У меня {years} лет коммерческого опыта с React и TypeScript. На последнем месте занимался "
    "архитектурой фронтенд-приложений, оптимизацией производительности и менторством джуниоров.\n\n"
    "Работал с Zustand и Redux, настраивал CI/CD через Gitlab CI, использовал Vite для сборки. "
    "Знаком с Docker — контейнеризовал фронтенд-приложения для локальной разработки.\n\n"
    "Буду рад обсудить, как мой опыт может быть полезен вашей команде.",

    "Добрый день! Увидел вашу вакансию и понял, что это то, что я ищу.\n\n"
    "За {years} лет работы с React я прошёл путь от Junior до Senior. Разрабатывал компоненты "
    "дизайн-системы, оптимизировал рендеринг (React.memo, useMemo, виртуализация списков), "
    "настраивал микрофронтенд-архитектуру.\n\n"
    "Из вашего стека: React, TypeScript, Zustand, Vite, Docker, Gitlab CI — всё использовал в продакшене. "
    "Особенно интересна возможность влиять на архитектуру — это то, чем я занимаюсь сейчас.\n\n"
    "Готов выполнить тестовое задание и пройти техническое интервью.",
]

COVER_MID = [
    "Здравствуйте! Хочу откликнуться на вакансию Frontend Developer.\n\n"
    "У меня {years} года опыта с React. На текущем месте разрабатываю UI-компоненты, "
    "работаю с Redux и REST API. Знаю TypeScript, использую Git.\n\n"
    "С Docker и CI/CD знаком на базовом уровне, но готов быстро разобраться.\n\n"
    "Буду рад возможности поучаствовать в отборе.",

    "Добрый день! Меня зовут {name}, я Frontend Developer с {years}-летним опытом.\n\n"
    "Работаю с React и JavaScript/TypeScript. Есть опыт с Redux, Git, REST API. "
    "Изучаю Zustand и Vite.\n\n"
    "Интересует возможность роста и работа над продуктом, а не аутсорс.\n\n"
    "Спасибо за рассмотрение!",
]

COVER_WEAK = [
    "Здравствуйте! Хочу у вас работать. Знаю React, учусь дальше.",
    "Привет! Я junior-разработчик, ищу первую работу. Знаю React и JavaScript.",
    "Добрый день. Откликаюсь на вакансию. Опыт небольшой, но быстро учусь.",
]

COVER_IRRELEVANT = [
    "Здравствуйте! Я Backend Developer с опытом на Python. Хочу попробовать фронтенд.\n"
    "Знаю основы React, но коммерческого опыта нет. Готов учиться.",

    "Добрый день! Я Data Scientist, но мне интересен фронтенд. Делал пет-проекты на React.\n"
    "Коммерческого опыта нет, но быстро учусь.",

    "Привет! Я DevOps-инженер. Иногда помогаю фронтендерам с деплоем. "
    "Хочу перейти во фронтенд-разработку.",
]


# --- Ответы на вопросы анкеты ---

ANSWERS_Q1_RENDERING = {
    "strong": [
        "Для оптимизации рендеринга использую несколько подходов:\n\n"
        "1. React.memo для предотвращения лишних ре-рендеров компонентов, которые не изменились.\n"
        "2. useMemo и useCallback для мемоизации вычислений и функций.\n"
        "3. Виртуализация длинных списков через react-window — на проекте с таблицей в 10K строк "
        "это снизило время рендера с 3 секунд до 200мс.\n"
        "4. Code splitting через React.lazy и Suspense — разбил бандл на чанки, начальная загрузка "
        "уменьшилась с 1.2MB до 400KB.\n"
        "5. Использую React DevTools Profiler для поиска узких мест.",

        "Основные техники оптимизации, которые применял:\n\n"
        "• Избегал inline-объектов и функций в JSX (создают новые ссылки → ре-рендеры)\n"
        "• Использовал useMemo для тяжёлых вычислений (фильтрация списка из 5000 элементов)\n"
        "• Внедрил виртуализацию для infinite scroll — react-virtuoso\n"
        "• Разделил бандл по роутам через lazy loading\n"
        "• Оптимизировал re-renders через Redux shallow equality checks\n"
        "• На проекте в Яндексе это дало 40% улучшение LCP (Largest Contentful Paint)",
    ],
    "mid": [
        "Использую React.memo для компонентов, которые часто перерендериваются. "
        "Также стараюсь не создавать новые объекты в пропсах. "
        "Для больших списков использую пагинацию вместо виртуализации.",

        "Применяю useMemo и useCallback когда замечаю тормоза. "
        "Разбиваю компоненты на меньшие, чтобы меньше перерендерилось. "
        "Пока не использовал виртуализацию, но знаю про react-window.",
    ],
    "weak": [
        "Слышал про React.memo, но не использовал. Стараюсь писать компоненты поменьше.",
        "Пока не сталкивался с проблемами производительности. Знаю, что есть useMemo.",
    ],
    "irrelevant": [
        "С фронтендом не работал глубоко. Знаю, что React что-то рендерит, но деталей не знаю.",
        "Я бэкендер, с оптимизацией рендеринга не сталкивался.",
    ],
}

ANSWERS_Q2_STATE = {
    "strong": [
        "Выбор стейт-менеджера зависит от задачи:\n\n"
        "• Zustand — для большинства проектов. Минимальный бойлерплейт, хороший DX. "
        "Использую его как default выбор.\n"
        "• Redux Toolkit — для крупных приложений с complex state, где нужен devtools, middleware, "
        "time-travel debugging. Использовал в Сбере.\n"
        "• Context API — только для низкоуровневых вещей (тема, локаль), не для бизнес-логики. "
        "При частых обновлениях Context вызывает ре-рендеры всех потребителей.\n\n"
        "Для серверного состояния использую React Query / SWR — это отдельная история, "
        "не стоит смешивать с клиентским стейтом.",

        "На последнем проекте мигрировали с Redux на Zustand —减少了 60% бойлерплейта. "
        "Zustand отлично подходит для среднего размера приложений. Redux оставляю для enterprise.\n\n"
        "Context API использую только для dependency injection (тема, auth). "
        "Для глобального состояния — Zustand или Redux, в зависимости от сложности.",
    ],
    "mid": [
        "В основном использую Redux Toolkit. Знаю про Zustand, изучаю. "
        "Context API использую для темы и авторизации. "
        "Для серверных данных пробовал React Query — понравилось.",

        "Работаю с Redux. Понимаю принципы: store, actions, reducers. "
        "Знаю про middleware (thunk, saga). Zustand слышал, но не пробовал.",
    ],
    "weak": [
        "Использую useState и Context. С Redux не работал.",
        "Пока только useState для локального состояния. Глобальное состояние не нужно было.",
    ],
    "irrelevant": [
        "С React-стейтом не работал. На бэкенде другие подходы к управлению состоянием.",
        "Знаю теоретически про Redux, но не использовал.",
    ],
}

ANSWERS_Q3_TESTING = {
    "strong": [
        "Мой подход к тестированию:\n\n"
        "1. Unit-тесты утилит и хуков — Jest + React Testing Library. "
        "Покрытие: 80%+ для критичной бизнес-логики.\n"
        "2. Компонентные тесты — RTL, тестирую поведение, а не реализацию. "
        "«Найди кнопку, кликни, проверь что появился текст».\n"
        "3. Интеграционные тесты — Playwright для критичных user flow (логин, оплата).\n"
        "4. Snapshot-тесты — только для UI-компонентов дизайн-системы.\n\n"
        "Не тестирую: стили, third-party библиотеки, простые presentational компоненты.\n"
        "CI запускает тесты на каждый PR, без прохождения — мерж невозможен.",

        "Использую Jest + RTL. Тестирую:\n"
        "• Хуки (useForm, useAuth) — unit\n"
        "• Компоненты — поведение через userEvent\n"
        "• API-слой — моки через MSW (Mock Service Worker)\n\n"
        "Покрытие ~75%, но стремлюсь к качеству тестов, а не к цифре. "
        "Лучше 50 хороших тестов, чем 200 бессмысленных.",
    ],
    "mid": [
        "Пишу unit-тесты на Jest. Для компонентов использую React Testing Library. "
        "Стараюсь покрывать основную бизнес-логику. Покрытие около 60%. "
        "С E2E-тестами не работал, но слышал про Cypress и Playwright.",

        "Тестирую хуки и утилиты через Jest. Компоненты тестирую редко — "
        "сложно настраивать. Знаю про RTL, но на текущем проекте не используем.",
    ],
    "weak": [
        "Тесты не писал. Знаю, что это важно, но не было времени разобраться.",
        "Слышал про Jest, но не использовал. Хочу научиться.",
    ],
    "irrelevant": [
        "Тестировал бэкенд через pytest. С фронтенд-тестами не знаком.",
        "Написал пару тестов для своего pet-проекта на React, но это всё.",
    ],
}

ANSWERS_Q4_API = {
    "strong": [
        "Мой подход к работе с API:\n\n"
        "1. Абстракция: отдельный слой API-клиента (axios/fetch) с типизацией через TypeScript.\n"
        "2. Обработка ошибок: единый error handler, маппинг HTTP-кодов в user-friendly сообщения.\n"
        "3. Загрузка: состояния loading/error/data через кастомный хук useApi или React Query.\n"
        "4. Кэширование: React Query для серверного состояния — автоматический re-fetch, "
        "invalidation, optimistic updates.\n"
        "5. Retry-логика: экспоненциальный backoff для временных ошибок.\n\n"
        "На проекте в Тинькофф внедрил React Query —减少了 70% бойлерплейта с состоянием загрузки.",

        "Использую React Query (TanStack Query) для всего серверного состояния. "
        "Он решает: кэширование, re-fetch, optimistic updates, error handling.\n\n"
        "Для мутаций: invalidate queries после успешного запроса. "
        "Для ошибок: глобальный error boundary + toast-уведомления.\n\n"
        "Типизирую все запросы/ответы через TypeScript — автодополнение и проверка на compile-time.",
    ],
    "mid": [
        "Работаю с API через fetch или axios. Делаю отдельный файл с функциями API. "
        "Обработка ошибок через try/catch. Состояние загрузки через useState. "
        "С React Query знаком, но не использовал в продакшене.",

        "Вызываю API через axios. Показываю спиннер во время загрузки. "
        "Если ошибка — показываю сообщение. Для кэширования пока ничего не использую.",
    ],
    "weak": [
        "Делал простые fetch-запросы. Обработку ошибок пока не настраивал.",
        "Знаю как сделать GET-запрос. С POST пока не работал.",
    ],
    "irrelevant": [
        "На бэкенде я сам пишу API. Как фронтенд его потребляет — не знаю.",
        "Делал API на FastAPI. Фронтенд-часть не трогал.",
    ],
}

ANSWERS_Q5_DEPLOY = {
    "strong": [
        "Мой типичный CI/CD пайплайн для фронтенда:\n\n"
        "1. Git push → Gitlab CI запускает пайплайн\n"
        "2. Stage: lint (ESLint) + type check (tsc) + test (Jest)\n"
        "3. Stage: build (Vite/Webpack) → оптимизация, минификация\n"
        "4. Stage: deploy на staging → автоматические smoke-тесты\n"
        "5. Stage: deploy на production (manual approval)\n\n"
        "Инфраструктура: Docker-контейнер с nginx для статики. "
        "Деплой на Kubernetes или просто на сервер через rsync.\n\n"
        "Настроил это на последнем месте — время деплоя сократилось с 30 минут до 3.",

        "Использую Gitlab CI:\n"
        "• .gitlab-ci.yml с stages: test → build → deploy\n"
        "• Docker-образ: node:lts для build, nginx:alpine для продакшена\n"
        "• Кэширование node_modules между пайплайнами\n"
        "• Preview-окружение на каждый MR (через Vercel/Netlify)\n\n"
        "Для мониторинга: Sentry для ошибок, Lighthouse CI для производительности.",
    ],
    "mid": [
        "Деплоил через простой скрипт: npm run build → scp на сервер. "
        "С CI/CD знаком теоретически, настраивал простой пайплайн в Gitlab. "
        "Docker использую для локальной разработки.",

        "Билдил проект через npm run build и заливал на хостинг. "
        "С Docker и CI/CD не работал, но хочу разобраться.",
    ],
    "weak": [
        "Не деплоил сам. Проект запускал локально через npm start.",
        "Выкладывал на GitHub Pages. Больше ничего не делал.",
    ],
    "irrelevant": [
        "Деплоил бэкенд-сервисы через Docker + Kubernetes. "
        "Фронтенд не деплоил.",

        "Настраивал CI/CD для бэкенда. Фронтенд-пайплайн не настраивал.",
    ],
}


# ===================================================================
# Генерация кандидатов
# ===================================================================

def generate_candidate(idx, is_relevant=True, quality="mid"):
    """Генерирует одного кандидата."""
    name = f"{random.choice(NAMES_FIRST)} {random.choice(NAMES_LAST)}"
    email = f"candidate{idx:04d}@email.com"
    years_exp = {"strong": random.randint(4, 10), "mid": random.randint(2, 5),
                 "weak": random.randint(0, 2), "irrelevant": random.randint(2, 8)}[quality]

    # Резюме
    # Резюме
    if quality == "irrelevant":
        exp_text = random.choice(EXPERIENCE_IRRELEVANT)
        skills = random.choice([SKILLS_IRRELEVANT_PYTHON, SKILLS_IRRELEVANT_DS,
                                SKILLS_IRRELEVANT_DEVOPS, SKILLS_IRRELEVANT_QA, SKILLS_IRRELEVANT_IOS])
    else:
        exp_text = {"strong": random.choice(EXPERIENCE_STRONG),
                     "mid": random.choice(EXPERIENCE_MID),
                     "weak": random.choice(EXPERIENCE_WEAK)}[quality]
        skills = {"strong": random.sample(SKILLS_STRONG, random.randint(8, 14)),
                   "mid": random.sample(SKILLS_MID, random.randint(5, 8)),
                   "weak": random.sample(SKILLS_WEAK, random.randint(2, 4))}[quality]

    edu_level = random.choice(EDUCATION_LEVELS) if quality != "strong" else random.choice(["Магистр", "Специалист", "Бакалавр"])
    uni = random.choice(UNIS)

    resume = f"=== РЕЗЮМЕ ===\n"
    resume += f"Имя: {name}\n"
    resume += f"Email: {email}\n"
    resume += f"Опыт работы: {years_exp} лет\n\n"
    resume += f"--- Опыт ---\n{exp_text}\n\n"
    resume += f"--- Навыки ---\n{', '.join(skills)}\n\n"
    resume += f"--- Образование ---\n{edu_level}, {uni}\n"

    # Сопроводительное письмо
    cover = {"strong": random.choice(COVER_STRONG),
             "mid": random.choice(COVER_MID),
             "weak": random.choice(COVER_WEAK),
             "irrelevant": random.choice(COVER_IRRELEVANT)}[quality]
    cover = cover.format(years=years_exp, name=name.split()[0])

    # Ответы на вопросы
    answers_pool = {
        "strong": (ANSWERS_Q1_RENDERING["strong"], ANSWERS_Q2_STATE["strong"],
                    ANSWERS_Q3_TESTING["strong"], ANSWERS_Q4_API["strong"],
                    ANSWERS_Q5_DEPLOY["strong"]),
        "mid": (ANSWERS_Q1_RENDERING["mid"], ANSWERS_Q2_STATE["mid"],
                ANSWERS_Q3_TESTING["mid"], ANSWERS_Q4_API["mid"],
                ANSWERS_Q5_DEPLOY["mid"]),
        "weak": (ANSWERS_Q1_RENDERING["weak"], ANSWERS_Q2_STATE["weak"],
                 ANSWERS_Q3_TESTING["weak"], ANSWERS_Q4_API["weak"],
                 ANSWERS_Q5_DEPLOY["weak"]),
        "irrelevant": (ANSWERS_Q1_RENDERING["irrelevant"], ANSWERS_Q2_STATE["irrelevant"],
                        ANSWERS_Q3_TESTING["irrelevant"], ANSWERS_Q4_API["irrelevant"],
                        ANSWERS_Q5_DEPLOY["irrelevant"]),
    }
    a1, a2, a3, a4, a5 = [random.choice(pool) for pool in answers_pool[quality]]
    questionnaire = f"Вопрос 1 (оптимизация рендеринга):\n{a1}\n\n"
    questionnaire += f"Вопрос 2 (управление состоянием):\n{a2}\n\n"
    questionnaire += f"Вопрос 3 (тестирование):\n{a3}\n\n"
    questionnaire += f"Вопрос 4 (работа с API):\n{a4}\n\n"
    questionnaire += f"Вопрос 5 (деплой):\n{a5}"

    # Рассчитываем 7 критериев (ground truth, модель их НЕ видит)
    scores = calculate_ground_truth(quality, years_exp, skills, edu_level, resume, cover, questionnaire)

    return {
        "candidate_id": f"CAND_{idx:04d}",
        "name": name,
        "email": email,
        "is_relevant": is_relevant,
        "quality_tier": quality,
        "years_experience": years_exp,
        "skills": ", ".join(skills),
        "education": edu_level,
        "university": uni,
        "resume_text": resume,
        "cover_letter": cover,
        "questionnaire_text": questionnaire,
        "exp_score": scores["exp_score"],
        "hard_score": scores["hard_score"],
        "quest_score": scores["quest_score"],
        "auth_score": scores["auth_score"],
        "red_flags_score": scores["red_flags_score"],
        "avail_score": scores["avail_score"],
        "add_value_score": scores["add_value_score"],
        "final_score": scores["final_score"],
        "is_hired": scores["is_hired"],
    }


def calculate_ground_truth(quality, years_exp, skills, edu_level, resume, cover, questionnaire):
    """Рассчитывает ground truth оценки на основе качества кандидата."""
    # 1. Experience (25%)
    exp_map = {"strong": 85 + random.uniform(-10, 15), "mid": 55 + random.uniform(-10, 20),
               "weak": 25 + random.uniform(-10, 20), "irrelevant": 10 + random.uniform(-5, 15)}
    exp_score = max(0, min(100, exp_map[quality]))

    # 2. Hard Skills (20%)
    required = {"React", "TypeScript", "JavaScript", "Git"}
    plus = {"Vite", "Docker", "Gitlab CI"}
    state_mgmt = {"Zustand", "Redux", "MobX"}
    skill_set = set(skills)

    req_match = len(required & skill_set) / len(required)
    plus_match = len(plus & skill_set) / len(plus)
    state_match = 1.0 if state_mgmt & skill_set else 0.0

    if quality == "irrelevant":
        hard_score = 15 + random.uniform(-5, 15)
    else:
        hard_score = (req_match * 60 + plus_match * 20 + state_match * 20) + random.uniform(-10, 10)
    hard_score = max(0, min(100, hard_score))

    # 3. Questionnaire Quality (22%)
    quest_map = {"strong": 85 + random.uniform(-10, 15), "mid": 55 + random.uniform(-10, 20),
                 "weak": 25 + random.uniform(-10, 20), "irrelevant": 15 + random.uniform(-10, 15)}
    quest_score = max(0, min(100, quest_map[quality]))

    # 4. Authenticity (12%) — сильные кандидаты пишут конкретно, слабые — общо
    auth_map = {"strong": 85 + random.uniform(-10, 15), "mid": 70 + random.uniform(-15, 20),
                "weak": 50 + random.uniform(-20, 25), "irrelevant": 60 + random.uniform(-20, 20)}
    auth_score = max(0, min(100, auth_map[quality]))

    # 5. Red Flags (8%)
    red_map = {"strong": 90 + random.uniform(-10, 10), "mid": 75 + random.uniform(-15, 15),
               "weak": 55 + random.uniform(-20, 25), "irrelevant": 60 + random.uniform(-20, 20)}
    red_flags_score = max(0, min(100, red_map[quality]))

    # 6. Availability (5%)
    avail_score = max(0, min(100, 70 + random.uniform(-20, 30)))

    # 7. Education & Additional (8%)
    edu_map = {"Магистр": 85, "Специалист": 80, "Бакалавр": 70, "Неоконченное высшее": 45, "Без высшего": 30}
    base_edu = edu_map.get(edu_level, 50)
    add_value_score = max(0, min(100, base_edu + random.uniform(-15, 15)))

    # Final Score
    final_score = (
        0.25 * exp_score +
        0.20 * hard_score +
        0.22 * quest_score +
        0.12 * auth_score +
        0.08 * red_flags_score +
        0.05 * avail_score +
        0.08 * add_value_score
    )

    # is_hired — вероятность на основе final_score
    hire_prob = 1 / (1 + np.exp(-(final_score - 55) / 12))
    is_hired = int(random.random() < hire_prob)

    return {
        "exp_score": round(exp_score, 1),
        "hard_score": round(hard_score, 1),
        "quest_score": round(quest_score, 1),
        "auth_score": round(auth_score, 1),
        "red_flags_score": round(red_flags_score, 1),
        "avail_score": round(avail_score, 1),
        "add_value_score": round(add_value_score, 1),
        "final_score": round(final_score, 1),
        "is_hired": is_hired,
    }


def main():
    print("Generating realistic candidate data...")

    candidates = []

    # 500 релевантных кандидатов
    # Распределение качества: 15% strong, 40% mid, 30% weak, 15% strong (senior)
    for i in range(500):
        if i < 75:
            quality = "strong"
        elif i < 275:
            quality = "mid"
        elif i < 425:
            quality = "weak"
        else:
            quality = "strong"
        candidates.append(generate_candidate(i, is_relevant=True, quality=quality))

    # 200 нерелевантных кандидатов
    for i in range(500, 700):
        candidates.append(generate_candidate(i, is_relevant=False, quality="irrelevant"))

    random.shuffle(candidates)

    # Сохраняем
    import pandas as pd
    df = pd.DataFrame(candidates)

    # CSV с текстами
    df.to_csv(REAL_DATA_DIR / "candidates_realistic.csv", index=False, encoding="utf-8")

    # Вакансия
    with open(REAL_DATA_DIR / "vacancy.json", "w", encoding="utf-8") as f:
        json.dump(VACANCY, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(df)} candidates:")
    print(f"  Relevant: {df['is_relevant'].sum()}")
    print(f"  Irrelevant: {(~df['is_relevant']).sum()}")
    print(f"  Hired: {df['is_hired'].sum()} ({df['is_hired'].mean()*100:.1f}%)")
    print(f"  Avg final_score: {df['final_score'].mean():.1f}")
    print(f"\nQuality distribution:")
    print(df['quality_tier'].value_counts())
    print(f"\nSaved to: {REAL_DATA_DIR}")

    # Примеры
    print(f"\n{'='*60}")
    print("EXAMPLE: Strong candidate")
    strong = df[df['quality_tier'] == 'strong'].iloc[0]
    print(f"  Name: {strong['name']}")
    print(f"  Final Score: {strong['final_score']}")
    print(f"  Hired: {strong['is_hired']}")
    print(f"  Resume preview: {strong['resume_text'][:150]}...")

    print(f"\n{'='*60}")
    print("EXAMPLE: Irrelevant candidate")
    irr = df[df['is_relevant'] == False].iloc[0]
    print(f"  Name: {irr['name']}")
    print(f"  Final Score: {irr['final_score']}")
    print(f"  Hired: {irr['is_hired']}")
    print(f"  Resume preview: {irr['resume_text'][:150]}...")


if __name__ == "__main__":
    main()
