import sys, io, json, time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import tempfile
from multiagent_tracer.multiagent_tracer import MultiAgentTracer

# Если mini_langsmith лежит рядом с приложением — можно не добавлять путь.
# Если вы положили его в другой каталог, добавьте sys.path.append("<путь>")
# sys.path.append(".")

st.set_page_config(page_title="Tracing & Log Viz (by мл)", layout="wide")

st.title("🛰️ Tracing & Log Viz (by мл)")
st.caption("Лёгкая визуализация спанов и логов для вашей мультиагентной системы")

st.header("🕸️ MultiAgentTracer — графы и сводки")

# --- Инициализация состояния (один раз) ---
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

# --- Загрузка файлов трасс ---
uploaded_ma_traces = st.file_uploader(
    "Загрузите один или несколько файлов трасс (JSON) от MultiAgentTracer",
    type=["json"],
    accept_multiple_files=True,
    key="ma_traces_files",
)

# Кнопка «загрузить/перестроить» создаёт/обновляет tracer и сохраняет его в session_state
if st.button("Загрузить/перестроить граф"):
    if not uploaded_ma_traces:
        st.warning("Сначала выберите хотя бы один JSON-файл трасс.")
    else:
        t = MultiAgentTracer(session_id="streamlit_viewer")
        import tempfile
        for uf in uploaded_ma_traces:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                tmp.write(uf.getvalue())
                tmp.flush()
                t.load_traces(tmp.name)   # внутри load_traces уже идёт rebuild графа
        ss.ma_tracer = t
        st.success("Трассы загружены и граф построен.")

# --- Управляющие элементы графа (всегда видны; значение хранится в session_state) ---
st.subheader("Граф вызовов")

cols = st.columns([1.2, 1, 1, 1.2])
modes = ["Aggregated", "Invocations"]

cols[0].radio(
    "Режим графа",
    modes,
    horizontal=True,
    key="graph_mode",
    index=modes.index(ss.graph_mode) if ss.graph_mode in modes else 0,
)
cols[1].toggle("Скрыть HTTP-инструменты", key="hide_http", value=ss.hide_http)
cols[2].slider("Толщина стрелок", 1.0, 6.0, key="edge_width", value=ss.edge_width, step=0.2)
cols[3].slider("Размер головки стрелки", 1.0, 3.5, key="arrow_size", value=ss.arrow_size, step=0.1)

c5, c6 = st.columns([1, 1])
c5.slider("Высота графа, px", 500, 1500, key="plot_height", value=ss.get("plot_height", 950), step=50)
c6.slider("Размер шрифта меток", 9, 20, key="label_font_size", value=ss.get("label_font_size", 12), step=1)

group_opts = ["agent_name", "agent_type"]
st.selectbox(
    "Агрегировать по (для Aggregated)",
    group_opts,
    key="group_by",
    index=group_opts.index(ss.group_by) if ss.group_by in group_opts else 0,
)

type_opts = ['orchestrator','search','analysis','report','clarification','tool','custom']
st.multiselect(
    "Типы агентов (для Aggregated)",
    options=type_opts,
    default=ss.include_types,
    key="include_types",
)

st.slider(
    "Обрезать превью текстов до, симв.",
    50, 1000,
    key="limit_preview",
    value=ss.limit_preview,
    step=50,
)

# --- Рисуем графы/диаграммы, если трассер загружен ---
if ss.ma_tracer:
    fig_graph = ss.ma_tracer.get_call_graph_viz(
        mode=("aggregated" if ss.graph_mode == "Aggregated" else "invocations"),
        include_types=set(ss.include_types),
        hide_http_tools=ss.hide_http,
        group_by=ss.group_by,
        edge_width_scale=ss.edge_width,
        arrow_size=ss.arrow_size,
        height=ss.plot_height,              # <<< новая высота
        label_font_size=ss.label_font_size  # <<< размер шрифта меток
    )
    st.plotly_chart(
        fig_graph,
        use_container_width=True,
        config={
            "displaylogo": False,
            "displayModeBar": True,   # <<< панель инструментов
            "scrollZoom": True,       # <<< zoom колесиком
        },
    )

    st.subheader("Диаграмма последовательности")
    fig_seq = ss.ma_tracer.get_sequence_diagram()
    st.plotly_chart(
        fig_seq,
        use_container_width=True,
        config={"displaylogo": False, "displayModeBar": True, "scrollZoom": True},
    )

    st.subheader("Таймлайн")
    fig_tl = ss.ma_tracer.get_timeline_viz()
    st.plotly_chart(
        fig_tl,
        use_container_width=True,
        config={"displaylogo": False, "displayModeBar": True, "scrollZoom": True},
    )
else:
    st.info("Загрузите трассы и нажмите «Загрузить/перестроить граф».")
