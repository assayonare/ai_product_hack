# ai-product-hack

# MindTrace — мультиагентная система с трассировкой и LLM-аналитикой

Интерактивная среда для запуска простой мультиагентной системы, сбора трасс (events), визуализации графов вызовов и анализа логов через LLM. Веб-интерфейс на Streamlit автоматически поднимает все сервисы, позволяет отправлять промпты в оркестратор и анализировать загруженные логи.

##  Структура проекта

```
.
├─ agent_with_orchestrator.py   # FastAPI-приложения: оркестратор + агенты (search, analysis, report)
├─ custom_trace.py              # Примеры/утилиты трассировки (обертки для tracer)
├─ llm_analytics.py             # Аналитика трасс: чанкинг, вызов LLM, офлайн-фолбэк
├─ streamlit_app.py             # Веб-UI: подъем агентов, отправка промптов, визуализации, LLM-анализ
├─ app.py                       # Утилита запуска Streamlit на фиксированном порту
└─ multiagent_tracer/
   └─ multiagent_tracer.py      # Класс MultiAgentTracer: сбор событий, rebuild графа, граф/sequence/timeline
```

*Streamlit-приложение поднимает сам оркестратор и агентов в фоновом режиме через `uvicorn`.*

---

## Требования

Необходимые для запуска библиотеки находятся в requirements.txt

---

## Переменные окружения

Создайте `.env` в корне (или выставьте через системное окружение):

```env
# LLM аналитика (опционально)
OPENAI_API_KEY=...                 # или
OPENROUTER_API_KEY=...

# Для OpenRouter укажите базовый URL
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=gpt-4o-mini              # либо, для OpenRouter: openai/gpt-4o-mini
```

`llm_analytics.py` автоматически использует переданный `api_key` или берет ключи из окружения (`OPENROUTER_API_KEY` при работе через OpenRouter, иначе `OPENAI_API_KEY`) и нормализует базовый URL. При отсутствии ключа будет использован офлайн-фолбэк (вердикт `approve` по флагу успеха).

---

## 

```bash
# 1) создать окружение и установить зависимости (пример)
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -U fastapi uvicorn requests aiohttp streamlit pandas plotly networkx matplotlib python-dotenv
pip install -U langchain-openai langchain-core jsonschema

# 2) запустить UI (он сам поднимет агентов на 8000..8003)
python -m streamlit run streamlit_app.py

```

Скрипт `streamlit_app.py` стартует Streamlit на порту `8501`. Логи пишутся в `streamlit.log`.

При старте страницы Streamlit-приложение автоматически пытается поднять четыре процесса `uvicorn`:
- `agent_with_orchestrator:orchestrator_app` → `:8000`
- `agent_with_orchestrator:search_app`       → `:8001`
- `agent_with_orchestrator:analysis_app`     → `:8002`
- `agent_with_orchestrator:report_app`       → `:8003`

Есть кнопки «Проверить здоровье агентов» и «Остановить агенты» в сайдбаре.

---

## Архитектура

### Оркестратор и агенты

Файл `agent_with_orchestrator.py` содержит четыре FastAPI-приложения (оркестратор + 3 агента) и эндпоинты `/health` для каждого сервиса. В одной из конфигураций файл также умеет сам поднимать все сервисы в потоках (free-port, `uvicorn.run`), если его запускать как модуль.

Здоровье сервисов проверяется UI по `http://localhost:8000/health` … `:8003/health`.

### Веб-UI (Streamlit)

`streamlit_app.py`:
- Автоматический запуск агентов в фоне (через `subprocess.Popen` + `uvicorn ... --reload`).
- Проверка здоровья сервисов и управление их жизненным циклом.
- Поле для ввода промпта и отправка в оркестратор на `POST {orchestrator_url}` (по умолчанию `http://localhost:8000/orchestrate`).
- Загрузка логов (JSON), генерация визуализаций и вызов LLM-аналитики.

### Трассировка и графы

`multiagent_tracer/multiagent_tracer.py` — ядро трассировки:
- События `AGENT_START/END`, `TOOL_START/END`, `MESSAGE_SENT`, `ERROR`.
- Восстановление **invocations-графа** по событиям с «очисткой»:
  — удаление HTTP-обёрток (`http_call::*`, `::HTTP POST`) и мгновенных «служебных» шагов;  
  — пересвязка родитель→дети через скрытые узлы;  
  — канонизация имён узлов (без префиксов/суффиксов).
- Визуализации:
  — **Call Graph** (режимы *invocations* и *aggregated*), с цветами по типам и стрелками статуса;  
  — **Sequence Diagram** (линии жизни, activation-box по парам START→END, стрелки parent/message);  
  — **Timeline** (полоски длительности по агентам).

### LLM-аналитика логов

`llm_analytics.py` выполняет разбор JSON-трассы:
- `_session_view_from_trace` сжимает события до полезных для LLM полей.
- `classify_session_with_llm` бьёт события на чанки, вызывает LLM, собирает вердикты (`approve/reject/uncertain`) и суммарную секцию. При любой ошибке провайдера/лимитах — **офлайн-фолбэк**: `approve` по признаку успеха из `_success_map`. Возвращает также токен-статистику, собранную из `response_metadata/headers` провайдера, если она доступна.
- `analyze_trace_bytes` — удобная обёртка для Streamlit: читает файл, вызывает классификатор, возвращает новый JSON с полем `"assessments"` и `"token_stats"`, а также множество `approved_event_id`.

---

## ▶️ Как пользоваться UI

1. Запустите `python streamlit_app.py` или выполните команду ` python -m streamlit run streamlit_app.py` и откройте страницу по ссылке из консоли (обычно `http://localhost:8501`) если страница не откроется автоматически.  
2. Дождитесь, пока сайдбар покажет успешный статус всех агентов (или нажмите «🩺 Проверить здоровье агентов»).  
3. Введите промпт в форму и отправьте — запрос уйдёт в оркестратор (`POST /orchestrate`). Адрес хранится в `orchestrator_url` в состоянии приложения и по умолчанию равен `http://localhost:8000/orchestrate`.
4. Загрузите JSON-трассу, нажмите кнопку анализа — будет выполнен анализ, а в результирующий JSON добавятся `assessments` и `token_stats` (в UI есть предпросмотр графов и диаграмм).

---

## 🧪 Ручной запуск без UI (опционально)

Запуск каждого сервиса отдельно:

```bash
uvicorn agent_with_orchestrator:orchestrator_app --port 8000 --reload
uvicorn agent_with_orchestrator:search_app       --port 8001 --reload
uvicorn agent_with_orchestrator:analysis_app     --port 8002 --reload
uvicorn agent_with_orchestrator:report_app       --port 8003 --reload
```

Проверка здоровья:

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
```

Эти же эндпоинты использует Streamlit-приложение для статуса.

---

## 🧩 Формат трассы

JSON трассы — это список событий с полями: `event_id`, `timestamp`, `event_type`, `agent_name`, `agent_type`, `data`, `parent_event_id`, плюс, для *_END, поля `duration` и `success`. UI добавляет в результат `"assessments"` и `"token_stats"`. См. функции подготовки сессии и сбор статистики в `llm_analytics.py`.

---

## 🐞 Траблшутинг

- **Агенты не поднимаются** — проверьте, что порты `8000–8003` свободны; используйте кнопки «Проверить здоровье»/«Остановить» в сайдбаре; при необходимости перезапустите UI.
- **Нет API-ключа** — аналитика перейдет в офлайн-режим, в консоли будет сообщение «API key not provided. Using offline fallback». Это ожидаемо.
- **429/лимиты провайдера** — `classify_session_with_llm` имеет backoff и при повторных сбоях уходит в офлайн-фолбэк, чтобы интерфейс не падал.
- **Sequence диаграмма без связей** — убедитесь, что логи сформированы с корректными `parent_event_id`. Внутри `rebuild` HTTP-обёртки и мгновенные шаги удаляются, а ребра «прокидываются» к видимым узлам, чтобы пайплайн был непрерывным.

