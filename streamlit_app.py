import json, time, os
from typing import Dict, Any
import pandas as pd
import streamlit as st
import tempfile
import requests
import aiohttp
import asyncio
from multiagent_tracer.multiagent_tracer import MultiAgentTracer
from llm_analytics import analyze_trace_bytes
import subprocess

from dotenv import load_dotenv
load_dotenv()

def start_agents():
    """Запускает агентские сервисы в фоновом режиме"""
    
    if "agents_started" not in st.session_state:
        st.session_state.agents_started = False
        st.session_state.agent_processes = []
    
    if not st.session_state.agents_started:
        try:
            try:
                response = requests.get("http://localhost:8000/health", timeout=2)
                if response.status_code == 200:
                    st.session_state.agents_started = True
                    st.sidebar.success("✅ Агенты уже запущены")
                    return True
            except:
                pass
            
            st.sidebar.info("🚀 Запускаем агентскую систему...")

            commands = [
                ["uvicorn", "agent_with_orchestrator:orchestrator_app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
                ["uvicorn", "agent_with_orchestrator:search_app", "--host", "0.0.0.0", "--port", "8001", "--reload"],
                ["uvicorn", "agent_with_orchestrator:analysis_app", "--host", "0.0.0.0", "--port", "8002", "--reload"],
                ["uvicorn", "agent_with_orchestrator:report_app", "--host", "0.0.0.0", "--port", "8003", "--reload"]
            ]
            
            processes = []
            for cmd in commands:
                try:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                    )
                    processes.append(process)
                    time.sleep(2)  
                except Exception as e:
                    st.sidebar.error(f"Ошибка запуска агента: {e}")
                    continue
            
            st.session_state.agent_processes = processes
            st.session_state.agents_started = True
            time.sleep(5)

            healthy = check_agents_health()
            if healthy:
                st.sidebar.success("✅ Агенты успешно запущены!")
                return True
            else:
                st.sidebar.warning("⚠️ Некоторые агенты не ответили")
                return False
                
        except Exception as e:
            st.sidebar.error(f"❌ Ошибка запуска агентов: {e}")
            return False
    return True

def check_agents_health():
    """Проверяет здоровье всех агентов"""
    services = {
        "Оркестратор": "http://localhost:8000/health",
        "Поисковый агент": "http://localhost:8001/health", 
        "Аналитический агент": "http://localhost:8002/health",
        "Агент отчетов": "http://localhost:8003/health"
    }
    
    all_healthy = True
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                st.sidebar.success(f"✅ {name}: работает")
            else:
                st.sidebar.error(f"❌ {name}: ошибка {response.status_code}")
                all_healthy = False
        except Exception as e:
            st.sidebar.error(f"❌ {name}: недоступен ({e})")
            all_healthy = False
    
    return all_healthy

def stop_agents():
    """Останавливает запущенные агенты"""
    if hasattr(st.session_state, 'agent_processes'):
        for process in st.session_state.agent_processes:
            try:
                process.terminate()
            except:
                pass
        st.session_state.agents_started = False
        st.sidebar.info("🛑 Агенты остановлены")

agents_ready = start_agents()

if not agents_ready:
    st.sidebar.warning("""
    ⚠️ Агенты не запущены или недоступны.
    
    Возможные решения:
    1. Запустите агенты вручную: `python app.py`
    2. Перезагрузите страницу
    3. Проверьте, что порты 8000-8003 свободны
    """)

st.sidebar.markdown("---")
if st.sidebar.button("🩺 Проверить здоровье агентов"):
    check_agents_health()

if st.sidebar.button("🛑 Остановить агенты"):
    stop_agents()
    st.sidebar.info("Агенты остановлены. Для перезапуска обновите страницу.")

# ------------------------------
# Страница + базовый стиль
# ------------------------------
st.set_page_config(
    page_title="MindTrace by мл",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
div.block-container { padding-top: 1.2rem; }
h1, h2, h3, .stMetric, .stMarkdown, .stCaption, .stSelectbox, .stDownloadButton {
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, 'Noto Sans';
  letter-spacing: .2px;
}
.card { background: rgba(255,255,255,.55); border: 1px solid rgba(0,0,0,.06);
  border-radius: 16px; padding: 16px 18px; box-shadow: 0 6px 18px rgba(0,0,0,.06); }
[data-theme="dark"] .card { background: rgba(17,24,39,.55); border-color: rgba(255,255,255,.08); }
.stButton > button, .stDownloadButton > button { border-radius: 12px; padding: .55rem 1rem; border: 1px solid rgba(0,0,0,.08); }
.badge { display:inline-flex; align-items:center; gap:.4rem; padding:.15rem .55rem; border-radius:999px; font-size:.82rem;
  background:rgba(20,184,166,.12); color:#059669; border:1px solid rgba(5,150,105,.35); }
.prompt-box { background: rgba(240, 253, 250, 0.7); border: 1px solid rgba(5, 150, 105, 0.3); border-radius: 12px; padding: 16px; margin: 10px 0; }
.response-box { background: rgba(239, 246, 255, 0.7); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 12px; padding: 16px; margin: 10px 0; }
</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.title("🛰️ MindTrace by мл")
st.caption("Лёгкая визуализация спанов и логов для вашей мультиагентной системы")

# ------------------------------
# Инициализация состояния
# ------------------------------
ss = st.session_state
if "ma_tracer" not in ss:
    ss.ma_tracer = None
if "graph_mode" not in ss:
    ss.graph_mode = "Aggregated"
if "hide_http" not in ss:
    ss.hide_http = True
if "edge_width" not in ss:
    ss.edge_width = 2.2
if "arrow_size" not in ss:
    ss.arrow_size = 2.0
if "group_by" not in ss:
    ss.group_by = "agent_name"
if "include_types" not in ss:
    ss.include_types = ['orchestrator','search','analysis','report','clarification','tool']
if "limit_preview" not in ss:
    ss.limit_preview = 300
if "plot_height" not in ss:
    ss.plot_height = 600
if "label_font_size" not in ss:
    ss.label_font_size = 12

# загруженные файлы
if "uploaded_buffers" not in ss:
    ss.uploaded_buffers = {}   
if "selected_file" not in ss:
    ss.selected_file = None

# метки LLM
if "llm_labels" not in ss:
    ss.llm_labels = {}        
if "show_approve" not in ss:
    ss.show_approve = True

# агентские состояния
if "orchestrator_url" not in ss:
    ss.orchestrator_url = "http://localhost:8000/orchestrate"
if "prompt_input" not in ss:
    ss.prompt_input = ""
if "agent_response" not in ss:
    ss.agent_response = None
if "agent_trace_id" not in ss:
    ss.agent_trace_id = None

# ------------------------------
# Утилита: единый стиль Plotly
# ------------------------------
def polish_plotly(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=40, r=40, t=40, b=40),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter")
    )
    return fig

# ------------------------------
# Функция для отправки запроса к оркестратору
# ------------------------------
async def send_to_orchestrator_async(prompt: str) -> Dict[str, Any]:
    """Асинхронная отправка запроса к оркестратору"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                ss.orchestrator_url,
                json={"input": prompt},
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    return {"error": f"HTTP {response.status}: {error_text}"}
    except Exception as e:
        return {"error": str(e)}

def send_to_orchestrator(prompt: str) -> Dict[str, Any]:
    """Синхронная обертка для асинхронной отправки"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=3)
        if response.status_code != 200:
            return {"error": "Оркестратор недоступен. Запустите агенты сначала."}
    except:
        return {"error": "Не удалось подключиться к оркестратору. Убедитесь, что агенты запущены."}
    
    return asyncio.run(send_to_orchestrator_async(prompt))

# ------------------------------
# Блок для ввода промпта и запуска агента
# ------------------------------
st.markdown("### 🤖 Запуск агентской системы")

with st.form(key="agent_prompt_form"):
    col1, col2 = st.columns([3, 1])
    with col1:
        prompt_input = st.text_area(
            "Введите промпт для агентской системы:",
            value=ss.prompt_input,
            height=100,
            placeholder="Например: 'Найди последние новости об искусственном интеллекте и сделай анализ'"
        )
    with col2:
        orchestrator_url = st.text_input(
            "URL оркестратора:",
            value=ss.orchestrator_url,
            help="URL эндпоинта /orchestrate оркестратора"
        )
    
    submit_prompt = st.form_submit_button("🚀 Запустить агентов")

if submit_prompt:
    ss.prompt_input = prompt_input
    ss.orchestrator_url = orchestrator_url
    
    if not prompt_input.strip():
        st.warning("Пожалуйста, введите промпт")
    else:
        with st.spinner("Агенты обрабатывают запрос... Это может занять несколько минут."):
            try:
                response = send_to_orchestrator(prompt_input)
                ss.agent_response = response
                
                if "error" in response:
                    st.error(f"Ошибка: {response['error']}")
                else:
                    ss.agent_trace_id = response.get("trace_id")
                    st.success("Запрос успешно обработан!")
                    
                    with st.expander("📋 Результат выполнения", expanded=True):
                        st.markdown('<div class="response-box">', unsafe_allow_html=True)
                        st.write("**Финальный результат:**")
                        st.write(response.get("final_result", "Нет результата"))
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.write("**Цепочка выполнения:**")
                        execution_chain = response.get("execution_chain", [])
                        for step in execution_chain:
                            with st.container():
                                st.markdown(f"**Шаг {step['step']} - {step['agent']}**")
                                st.text(f"Вход: {step.get('input', '')[:200]}...")
                                st.text(f"Выход: {step.get('output', '')[:200]}...")

                    if ss.agent_trace_id:
                        st.info(f"Trace ID: {ss.agent_trace_id}")
                        
            except Exception as e:
                st.error(f"Ошибка при выполнении запроса: {str(e)}")

if ss.agent_response and "error" not in ss.agent_response:
    with st.expander("📋 Предыдущий результат", expanded=False):
        st.markdown('<div class="response-box">', unsafe_allow_html=True)
        st.write("**Финальный результат:**")
        st.write(ss.agent_response.get("final_result", "Нет результата"))
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# Загрузка файлов
# ------------------------------


def merge_traces(trace_files: Dict[str, bytes]) -> bytes:
    """
    Объединяет трассы из нескольких файлов в один
    """
    if not trace_files:
        return b'{}'
    
    merged_events = []
    merged_stats = {
        "total_events": 0,
        "agents": {},
        "session_duration": 0,
        "error_count": 0
    }
    
    session_id = None
    
    for filename, content in trace_files.items():
        try:
            data = json.loads(content.decode('utf-8'))
            
            if session_id is None and 'session_id' in data:
                session_id = data['session_id']

            if 'events' in data:
                merged_events.extend(data['events'])
                merged_stats["total_events"] += len(data['events'])

            if 'stats' in data and 'agents' in data['stats']:
                for agent_name, agent_stats in data['stats']['agents'].items():
                    if agent_name not in merged_stats["agents"]:
                        merged_stats["agents"][agent_name] = agent_stats
                    else:
                        merged_stats["agents"][agent_name]["calls"] += agent_stats.get("calls", 0)
                        merged_stats["agents"][agent_name]["total_duration"] += agent_stats.get("total_duration", 0)
                        merged_stats["agents"][agent_name]["errors"] += agent_stats.get("errors", 0)

                        if agent_stats.get("calls", 0) > 0:
                            merged_stats["agents"][agent_name]["avg_duration"] = (
                                merged_stats["agents"][agent_name]["total_duration"] / 
                                merged_stats["agents"][agent_name]["calls"]
                            )

            if 'stats' in data:
                merged_stats["session_duration"] = max(
                    merged_stats["session_duration"], 
                    data['stats'].get("session_duration", 0)
                )
                merged_stats["error_count"] += data['stats'].get("error_count", 0)
                
        except Exception as e:
            print(f"Ошибка при обработке файла {filename}: {e}")
            continue
    
    merged_events.sort(key=lambda x: x.get('timestamp', 0))

    merged_data = {
        "session_id": session_id or f"merged_session_{int(time.time())}",
        "events": merged_events,
        "stats": merged_stats
    }
    
    return json.dumps(merged_data, ensure_ascii=False, indent=2).encode('utf-8')

uploaded_files = st.file_uploader("Загрузить файлы трасс (JSON)", accept_multiple_files=True, type="json")

if uploaded_files:
    for up in uploaded_files:
        if up.name not in ss.uploaded_buffers:
            ss.uploaded_buffers[up.name] = up.getvalue()
            st.success(f"Файл загружен: {up.name}")

if ss.uploaded_buffers:
    ss.selected_file = st.selectbox("Выберите файл для просмотра", options=list(ss.uploaded_buffers.keys()))

    if st.button("🔄 Объединить все трассы в один файл"):
        with st.spinner("Объединяем трассы..."):
            merged_content = merge_traces(ss.uploaded_buffers)
            merged_filename = f"merged_traces_{int(time.time())}.json"
            ss.uploaded_buffers[merged_filename] = merged_content
            ss.selected_file = merged_filename
            st.success(f"✅ Трассы объединены в файл: {merged_filename}")

with st.sidebar:
    
    st.subheader("⚙️ Параметры визуализации")
    with st.form(key="viz_params"):
        graph_mode = st.radio("Режим графа", ["Aggregated","Invocations"], index=(0 if ss.graph_mode=="Aggregated" else 1), horizontal=True)
        hide_http = st.toggle("Скрывать HTTP/tool-узлы", value=ss.hide_http)
        group_by = st.selectbox("Группировка (для Aggregated)", ["agent_name", "agent_type"], index=(0 if ss.group_by=="agent_name" else 1))
        include_types = st.multiselect("Типы агентов (для Aggregated)", 
                                       ['orchestrator','search','analysis','report','clarification','tool','custom'],
                                       default=ss.include_types)
        edge_width = st.slider("Толщина рёбер", 1.0, 5.0, ss.edge_width, step=0.2)
        arrow_size = st.slider("Размер стрелок", 1.0, 4.0, ss.arrow_size, step=0.2)
        plot_height = st.slider("Высота графа (px)", 400, 2000, ss.plot_height, step=50)
        label_font_size = st.slider("Размер шрифта меток", 8, 20, ss.label_font_size, step=1)
        submit = st.form_submit_button("Построить граф")

        if submit:
            if not ss.selected_file:
                st.warning("Сначала выберите файл.")
            else:
                with st.spinner("Загружаем трассу…"):
                    t = MultiAgentTracer(session_id="streamlit_viewer")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                        tmp.write(ss.uploaded_buffers[ss.selected_file]); tmp.flush(); t.load_traces(tmp.name)
                    os.unlink(tmp.name)

                ss.ma_tracer = t
                ss.graph_mode = graph_mode
                ss.hide_http = hide_http
                ss.group_by = group_by
                ss.include_types = include_types
                ss.edge_width = edge_width
                ss.arrow_size = arrow_size
                ss.plot_height = plot_height
                ss.label_font_size = label_font_size
                st.success("Граф построен по новым параметрам.")

tabs = st.tabs(["🕸️ Граф вызовов", "⏱️ Диаграмма последовательности", "📈 Таймлайн", "🤖 LLM-аналитик"])

with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Граф вызовов")
    if ss.ma_tracer:
        fig_graph = ss.ma_tracer.get_call_graph_viz(
            mode=("aggregated" if ss.graph_mode == "Aggregated" else "invocations"),
            include_types=set(ss.include_types),
            hide_http_tools=ss.hide_http,
            group_by=ss.group_by,
            edge_width_scale=ss.edge_width,
            arrow_size=ss.arrow_size,
            height=ss.plot_height,
            label_font_size=ss.label_font_size,
        )
        fig_graph = polish_plotly(fig_graph)
        st.plotly_chart(
            fig_graph, use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True,
                    "modeBarButtonsToRemove": ["toggleSpikelines","autoScale2d","lasso2d","select2d"]}
        )
        st.download_button(
            "⬇️ Скачать граф (HTML)",
            data=fig_graph.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8"),
            file_name="graph.html",
            mime="text/html",
        )
    else:
        st.info("Загрузите трассы, выберите файл и нажмите «Построить граф».")
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Диаграмма последовательности")
    if ss.ma_tracer:
        fig_seq = polish_plotly(ss.ma_tracer.get_sequence_diagram())
        st.plotly_chart(fig_seq, use_container_width=True, config={"displaylogo": False})
        st.download_button(
            "⬇️ Скачать диаграмму (HTML)",
            data=fig_seq.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8"),
            file_name="sequence.html",
            mime="text/html",
        )
    else:
        st.info("Нет данных для диаграммы. Постройте граф.")
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Таймлайн")
    if ss.ma_tracer:
        fig_tl = polish_plotly(ss.ma_tracer.get_timeline_viz())
        st.plotly_chart(fig_tl, use_container_width=True, config={"displaylogo": False})
        st.download_button(
            "⬇️ Скачать таймлайн (HTML)",
            data=fig_tl.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8"),
            file_name="timeline.html",
            mime="text/html",
        )
    else:
        st.info("Нет данных для таймлайна. Постройте граф.")
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🤖 LLM-аналитик")
    st.caption(
        "Запустит LLM-анализ трассы: разобьёт на чанки, оценит каждое событие, проставит approve/reject и вернёт новый JSON-файл. "
        "Можно использовать кастом LLM (OpenRouter/Qwen). Если нет — оффлайн-эвристика (approve=success)."
    )

    use_llm = st.toggle("Использовать LLM (если нет — оффлайн фолбэк)", value=True)


    col_a, col_b = st.columns([1, 1])
    run_clicked = col_a.button("🔍 Запустить аналитика на выбранном файле")
    if run_clicked:
        if not ss.selected_file:
            st.warning("Сначала выберите файл.")
        else:
            try:
                base = ss.selected_file.rsplit(".json", 1)[0]
                new_name = f"{base}__assessed.json"

                if new_name in ss.uploaded_buffers:
                    new_name = f"{base}__assessed_{int(time.time())}.json"

                with st.spinner("Анализируем лог и формируем новый файл…"):
                    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY") \
                            or st.secrets.get("OPENAI_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")
                    base_url = os.getenv("LLM_BASE_URL") or st.secrets.get("LLM_BASE_URL")
                    new_bytes, approved_ids, token_stats = analyze_trace_bytes(
                        ss.uploaded_buffers[ss.selected_file],
                        use_llm=True,
                        model_name=os.getenv("LLM_MODEL"),
                        base_url=base_url,
                        api_key=api_key
                    )
                    # new_bytes, approved_ids, token_stats = analyze_trace_bytes(
                    #     ss.uploaded_buffers[ss.selected_file],
                    #     use_llm=True,
                    #     model_name=os.getenv("LLM_MODEL"),  # или любая другая модель OpenRouter
                    #     base_url=os.getenv("LLM_BASE_URL"),
                    #     api_key=os.getenv("OPENAI_API_KEY")
                    # )
                    ss.uploaded_buffers[new_name] = new_bytes
                    ss.selected_file = new_name

                    new_trace = json.loads(new_bytes.decode("utf-8"))
                    assessments = new_trace.get("assessments", {})
                    session_assessment = assessments.get("session_assessment", {})
                    assessed_events = assessments.get("events", [])

                    t = MultiAgentTracer(session_id="streamlit_viewer")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                        tmp.write(new_bytes); tmp.flush(); t.load_traces(tmp.name)
                    os.unlink(tmp.name)

                    G = t.call_graph
                    for nid in G.nodes():
                        if nid in approved_ids:
                            G.nodes[nid]["approved"] = True

                    ss.ma_tracer = t

                fig_graph = polish_plotly(ss.ma_tracer.get_call_graph_viz(
                    mode=("aggregated" if ss.graph_mode == "Aggregated" else "invocations"),
                    include_types=set(ss.include_types),
                    hide_http_tools=ss.hide_http,
                    group_by=ss.group_by,
                    edge_width_scale=ss.edge_width,
                    arrow_size=ss.arrow_size,
                    height=ss.plot_height,
                    label_font_size=ss.label_font_size
                ))
                graph_slot = st.empty()
                seq_slot = st.empty()
                tl_slot = st.empty()
                
                graph_slot.empty(); graph_slot.plotly_chart(
                    fig_graph, use_container_width=True,
                    config={"displaylogo": False, "scrollZoom": True})

                st.success(f"Новый файл создан и подгружен: {new_name}")
                col_b.download_button("⬇️ Скачать новый JSON", data=new_bytes, file_name=new_name, mime="application/json")


                with st.expander("🧠 Мысли LLM о сессии", expanded=True):

                    if use_llm:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Токены промпта", token_stats.get("total_prompt_tokens", 0))
                        with col2:
                            st.metric("Токены ответа", token_stats.get("total_completion_tokens", 0))
                        with col3:
                            st.metric("Всего токенов", token_stats.get("total_tokens", 0))
                        st.divider()
                    
                    st.text(session_assessment.get("summary", "Нет summary от LLM."))
                    if session_assessment.get("expert_review"):
                        st.markdown(f"**Экспертная оценка:** {session_assessment['expert_review']}")
                    if session_assessment.get("improvement_suggestions"):
                        st.markdown("**Рекомендации по улучшению:**")
                        for rec in session_assessment["improvement_suggestions"]:
                            st.markdown(f"- {rec}")
                    if session_assessment.get("major_issues"):
                        issues_df = pd.DataFrame(session_assessment["major_issues"])
                        st.dataframe(issues_df, use_container_width=True)

                with st.expander("📋 Детальный анализ событий LLM"):
                    if assessed_events:

                        events_map = {e.get("event_id"): e for e in new_trace.get("events", [])}
                        
                        events_df = pd.DataFrame([
                            {
                                "event_id": ev.get("event_id"),
                                "agent": events_map.get(ev.get("event_id"), {}).get("agent_name", ""),
                                "type": events_map.get(ev.get("event_id"), {}).get("agent_type", ""),
                                "tool": (events_map.get(ev.get("event_id"), {}).get("data", {}) or {}).get("tool_name", ""),
                                "verdict": ev.get("assessment", {}).get("verdict"),
                                "reason": (
                                    ev.get("assessment", {}).get("reason")
                                    or ev.get("assessment", {}).get("explanation")
                                )
                            }
                            for ev in assessed_events
                        ])

                        if events_df["tool"].isna().all() or (events_df["tool"] == "").all():
                            events_df = events_df.drop(columns=["tool"])
                        st.dataframe(events_df, use_container_width=True)
                    else:
                        st.info("Нет детального анализа событий от LLM.")


                with st.expander("📄 Исходный файл (raw JSON)"):
                    original_bytes = ss.uploaded_buffers.get(base + ".json") or ss.uploaded_buffers[ss.selected_file]  # Берем оригинал по имени
                    if original_bytes:
                        original_trace = json.loads(original_bytes.decode("utf-8"))
                        st.json(original_trace, expanded=False)
                    else:
                        st.warning("Исходный файл не найден.")

            except Exception as e:
                st.exception(e)

    st.markdown('</div>', unsafe_allow_html=True)

    

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("© 2025 мл. MindTrace")