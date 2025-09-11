import sys, io, json, time, os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt  # ок, пусть останется для совместимости
import streamlit as st
import tempfile
from multiagent_tracer.multiagent_tracer import MultiAgentTracer
#импорт нашего агента-аналитика
from llm_analytics import analyze_trace_bytes

# ------------------------------
# Страница + базовый стиль
# ------------------------------
st.set_page_config(
    page_title="Tracing & Log Viz (by мл)",
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
</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.title("🛰️ Tracing & Log Viz (by мл)")
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
    ss.plot_height = 950
if "label_font_size" not in ss:
    ss.label_font_size = 12

# загруженные файлы
if "uploaded_buffers" not in ss:
    ss.uploaded_buffers = {}   # filename -> bytes
if "selected_file" not in ss:
    ss.selected_file = None

# метки LLM
if "llm_labels" not in ss:
    ss.llm_labels = {}         # agent_name -> True/False (approve)
if "show_approve" not in ss:
    ss.show_approve = True

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
# Загрузка файлов
# ------------------------------
uploaded_files = st.file_uploader("Загрузить файлы трасс (JSON)", accept_multiple_files=True, type="json")

if uploaded_files:
    for up in uploaded_files:
        if up.name not in ss.uploaded_buffers:
            ss.uploaded_buffers[up.name] = up.getvalue()
            st.success(f"Файл загружен: {up.name}")

# Выбор файла
if ss.uploaded_buffers:
    ss.selected_file = st.selectbox("Выберите файл для просмотра", options=list(ss.uploaded_buffers.keys()))

# Форма параметров и построение графа
with st.sidebar:
    st.subheader("⚙️ Параметры")
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


# Tabs
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
    mdl = st.text_input("Модель LLM", value="gpt-4o-mini")
    base_url = st.text_input("Base URL (для OpenRouter)", value="https://openrouter.ai/api/v1")
    api_key = st.text_input("API Key (для OpenRouter или OpenAI)", type="password")

    col_a, col_b = st.columns([1, 1])
    run_clicked = col_a.button("🔍 Запустить аналитика на выбранном файле")
    if run_clicked:
        if not ss.selected_file:
            st.warning("Сначала выберите файл.")
        else:
            try:
                base = ss.selected_file.rsplit(".json", 1)[0]
                new_name = f"{base}__assessed.json"
                # если уже есть — добавим приписку
                if new_name in ss.uploaded_buffers:
                    new_name = f"{base}__assessed_{int(time.time())}.json"

                with st.spinner("Анализируем лог и формируем новый файл…"):
                    new_bytes, approved_ids = analyze_trace_bytes(
                        ss.uploaded_buffers[ss.selected_file],
                        model_name=mdl,
                        base_url=base_url if base_url else None,
                        api_key=api_key if api_key else None,
                        use_llm=use_llm
                    )
                    # сохраняем в «загруженные»
                    ss.uploaded_buffers[new_name] = new_bytes
                    ss.selected_file = new_name

                    # Распарсить новый JSON для отображения мыслей LLM
                    new_trace = json.loads(new_bytes.decode("utf-8"))
                    assessments = new_trace.get("assessments", {})
                    session_assessment = assessments.get("session_assessment", {})
                    assessed_events = assessments.get("events", [])

                    # Собираем tracer по новому файлу
                    t = MultiAgentTracer(session_id="streamlit_viewer")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                        tmp.write(new_bytes); tmp.flush(); t.load_traces(tmp.name)
                    os.unlink(tmp.name)

                    # проставляем approve узлам по event_id
                    G = t.call_graph
                    for nid in G.nodes():
                        if nid in approved_ids:
                            G.nodes[nid]["approved"] = True

                    ss.ma_tracer = t

                # перерисовываем все 3 графика
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

                fig_seq = polish_plotly(ss.ma_tracer.get_sequence_diagram())
                seq_slot.empty(); seq_slot.plotly_chart(fig_seq, use_container_width=True, config={"displaylogo": False})

                fig_tl = polish_plotly(ss.ma_tracer.get_timeline_viz())
                tl_slot.empty(); tl_slot.plotly_chart(fig_tl, use_container_width=True, config={"displaylogo": False})

                st.success(f"Новый файл создан и подгружен: {new_name}")
                col_b.download_button("⬇️ Скачать новый JSON", data=new_bytes, file_name=new_name, mime="application/json")

                # Отображение мыслей LLM
                with st.expander("🧠 Мысли LLM о сессии", expanded=True):
                    st.text(session_assessment.get("summary", "Нет summary от LLM."))
                    if session_assessment.get("major_issues"):
                        issues_df = pd.DataFrame(session_assessment["major_issues"])
                        st.dataframe(issues_df, use_container_width=True)

                with st.expander("📋 Детальный анализ событий LLM"):
                    if assessed_events:
                        events_df = pd.DataFrame([
                            {
                                "event_id": ev.get("event_id"),
                                "verdict": ev.get("assessment", {}).get("verdict"),
                                "reason": ev.get("assessment", {}).get("reason")
                            } for ev in assessed_events
                        ])
                        st.dataframe(events_df, use_container_width=True)
                    else:
                        st.info("Нет детального анализа событий от LLM.")

                # Просмотр исходного файла
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