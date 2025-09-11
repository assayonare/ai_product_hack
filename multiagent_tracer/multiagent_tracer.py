import json
import time
import logging
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from threading import Lock
import uuid
import inspect
from functools import wraps
from typing import Optional, List, Dict, Any, Callable, Set
import pandas as pd



class AgentType(Enum):
    """Типы агентов в системе"""
    ORCHESTRATOR = "orchestrator"
    SEARCH = "search"
    ANALYSIS = "analysis"
    REPORT = "report"
    CLARIFICATION = "clarification"
    TOOL = "tool"
    CUSTOM = "custom"


class EventType(Enum):
    """Типы событий в системе трассировки"""
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    ERROR = "error"
    CUSTOM = "custom"


@dataclass
class TraceEvent:
    """Событие в системе трассировки"""
    event_id: str
    timestamp: float
    event_type: EventType
    agent_name: str
    agent_type: AgentType
    data: Dict[str, Any]
    parent_event_id: Optional[str] = None
    session_id: Optional[str] = None
    duration: Optional[float] = None
    success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для сериализации"""
        result = asdict(self)
        result['event_type'] = self.event_type.value
        result['agent_type'] = self.agent_type.value
        return result

class MultiAgentTracer:
    """
    Трассировка мультиагентной системы + интерактивные визуализации
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        log_file: Optional[str] = None,
        enable_real_time_viz: bool = False,
        llm_classifier: Optional[Any] = None,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.events: List[TraceEvent] = []
        self.active_events: Dict[str, TraceEvent] = {}  # ключ: event_id из *_START
        self.lock = None  # не обязателен в Streamlit; если нужен - импортируй threading.Lock()
        self.log_file = log_file
        self.enable_real_time_viz = enable_real_time_viz
        self.llm_classifier = llm_classifier

        # Базовый граф вызовов (invocations)
        self.call_graph = nx.DiGraph()

        # Логирование
        if log_file:
            logging.basicConfig(
                filename=log_file,
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                encoding="utf-8",
            )
        self.logger = logging.getLogger("MultiAgentTracer")

    # --------------------------
    # Базовое логирование событий
    # --------------------------

    def _append_event(self, event: TraceEvent):
        self.events.append(event)
        self._log_event(event)

    def _log_event(self, event: TraceEvent):
        self.logger.info(
            f"Event: {event.event_type.value.lower()} | Agent: {event.agent_name} | Data: {event.data}"
        )

    # --------------------------
    # Старт/завершение агентов
    # --------------------------

    def start_trace(
        self,
        agent_name: str,
        agent_type: AgentType,
        data: Dict[str, Any],
        parent_event_id: Optional[str] = None,
    ) -> str:
        """Начать трассировку агента (AGENT_START)"""
        event_id = str(uuid.uuid4())
        ev = TraceEvent(
            event_id=event_id,
            timestamp=time.time(),
            event_type=EventType.AGENT_START,
            agent_name=agent_name,
            agent_type=agent_type,
            data=data,
            parent_event_id=parent_event_id,
            session_id=self.session_id,
        )
        self.active_events[event_id] = ev
        self._append_event(ev)

        # Нода в графе
        self.call_graph.add_node(
            event_id,
            agent_name=agent_name,
            agent_type=agent_type.value if isinstance(agent_type, AgentType) else str(agent_type),
            start_time=ev.timestamp,
        )
        if parent_event_id and parent_event_id in self.call_graph:
            self.call_graph.add_edge(parent_event_id, event_id)
        return event_id

    def end_trace(self, start_event_id: str, result: Dict[str, Any], success: bool = True):
        """Завершить трассировку агента (AGENT_END)"""
        start_ev = self.active_events.pop(start_event_id, None)
        now = time.time()
        duration = (now - start_ev.timestamp) if start_ev else None

        end_ev = TraceEvent(
            event_id=f"{start_event_id}_end",
            timestamp=now,
            event_type=EventType.AGENT_END,
            agent_name=start_ev.agent_name if start_ev else "unknown",
            agent_type=start_ev.agent_type if start_ev else AgentType.CUSTOM,
            data=result,
            parent_event_id=start_event_id,
            session_id=self.session_id,
            duration=duration,
            success=success,
        )
        self._append_event(end_ev)

        # Обновим атрибуты ноды
        if start_event_id in self.call_graph:
            self.call_graph.nodes[start_event_id]["duration"] = duration
            self.call_graph.nodes[start_event_id]["success"] = success
            self.call_graph.nodes[start_event_id]["end_time"] = now

    # --------------------------
    # Логирование инструментов
    # --------------------------

    def log_tool_start(
        self,
        agent_name: str,
        tool_name: str,
        tool_input: Any,
        parent_event_id: Optional[str] = None,
    ) -> str:
        """
        Старт вызова инструмента (TOOL_START) как отдельного узла в графе.
        Возвращает event_id старта, который нужно передать в log_tool_end.
        """
        event_id = str(uuid.uuid4())
        ev = TraceEvent(
            event_id=event_id,
            timestamp=time.time(),
            event_type=EventType.TOOL_START,
            agent_name=agent_name,
            agent_type=AgentType.TOOL,
            data={"tool_name": tool_name, "input": str(tool_input)},
            parent_event_id=parent_event_id,
            session_id=self.session_id,
        )
        self.active_events[event_id] = ev
        self._append_event(ev)

        # Узел инструмента
        self.call_graph.add_node(
            event_id,
            agent_name=f"{agent_name}::{tool_name}",
            agent_type="tool",
            start_time=ev.timestamp,
        )
        if parent_event_id and parent_event_id in self.call_graph:
            self.call_graph.add_edge(parent_event_id, event_id)
        return event_id

    def log_tool_end(self, start_event_id: str, tool_output: Any, success: bool = True):
        """
        Завершение вызова инструмента (TOOL_END).
        """
        start_ev = self.active_events.pop(start_event_id, None)
        now = time.time()
        duration = (now - start_ev.timestamp) if start_ev else None

        ev = TraceEvent(
            event_id=f"{start_event_id}_end",
            timestamp=now,
            event_type=EventType.TOOL_END,
            agent_name=start_ev.agent_name if start_ev else "tool",
            agent_type=AgentType.TOOL,
            data={"output": str(tool_output)},
            parent_event_id=start_event_id,
            session_id=self.session_id,
            duration=duration,
            success=success,
        )
        self._append_event(ev)

        # Обновим атрибуты ноды
        if start_event_id in self.call_graph:
            self.call_graph.nodes[start_event_id]["duration"] = duration
            self.call_graph.nodes[start_event_id]["success"] = success
            self.call_graph.nodes[start_event_id]["end_time"] = now

    # Совместимость: старый короткий one-shot лог
    def log_tool_call(
        self,
        agent_name: str,
        tool_name: str,
        tool_input: Any,
        tool_output: Any,
        parent_event_id: Optional[str] = None,
    ):
        eid = self.log_tool_start(agent_name, tool_name, tool_input, parent_event_id)
        self.log_tool_end(eid, tool_output, success=True)

    # --------------------------
    # Прочее логирование
    # --------------------------

    def log_message(
        self,
        from_agent: str,
        to_agent: str,
        message: Any,
        parent_event_id: Optional[str] = None,
    ):
        ev = TraceEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            event_type=EventType.MESSAGE_SENT,
            agent_name=from_agent,
            agent_type=AgentType.CUSTOM,
            data={"from_agent": from_agent, "to_agent": to_agent, "message": str(message)},
            parent_event_id=parent_event_id,
            session_id=self.session_id,
        )
        self._append_event(ev)

    def log_error(self, agent_name: str, error: Exception, parent_event_id: Optional[str] = None):
        ev = TraceEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            event_type=EventType.ERROR,
            agent_name=agent_name,
            agent_type=AgentType.CUSTOM,
            data={"error_type": type(error).__name__, "error_message": str(error)},
            parent_event_id=parent_event_id,
            session_id=self.session_id,
            success=False,
        )
        self._append_event(ev)

    # --------------------------
    # Визуализации
    # --------------------------

    def _contract_hidden_nodes(self, G: nx.DiGraph, hide_pred: Callable[[dict], bool]) -> nx.DiGraph:
        """
        Возвращает новый граф, где узлы, удовлетворяющие hide_pred(node_attr),
        удалены, а рёбра между видимыми узлами «перекинуты» в обход скрытых.
        """
        if not G or G.number_of_nodes() == 0:
            return nx.DiGraph()

        visible = [n for n in G.nodes if not hide_pred(G.nodes[n])]
        if not visible:
            return nx.DiGraph()

        H = nx.DiGraph()
        for n in visible:
            H.add_node(n, **G.nodes[n])

        for src in visible:
            for v in G.successors(src):
                if v in visible:
                    H.add_edge(src, v, weight=H.get_edge_data(src, v, {}).get("weight", 0) + 1)
                else:
                    # разворачиваем цепочку скрытых узлов
                    stack = [v]
                    seen = set()
                    while stack:
                        h = stack.pop()
                        if h in seen:
                            continue
                        seen.add(h)
                        for w in G.successors(h):
                            if w in visible and w != src:
                                H.add_edge(
                                    src, w, weight=H.get_edge_data(src, w, {}).get("weight", 0) + 1
                                )
                            else:
                                if w not in seen:
                                    stack.append(w)
        return H

    def _build_aggregated_graph(
        self,
        include_types: Optional[Set[str]] = None,
        hide_http_tools: bool = False,  # оставлено для совместимости; http/tool уже очищены в rebuild
        group_by: str = "agent_name",
    ) -> nx.DiGraph:
        """
        Агрегация поверх очищенного invocations-графа.
        group_by: "agent_name" (по каноническим именам) или "agent_type".
        include_types: множество строковых типов, которые оставляем.
        """
        G = self.call_graph
        H = nx.DiGraph()
        if G.number_of_nodes() == 0:
            return H

        include_types = include_types or {t.value for t in AgentType}

        def key_of(nd: dict) -> Optional[str]:
            at = str(nd.get("agent_type", "custom"))
            if at not in include_types:
                return None
            if group_by == "agent_type":
                return at
            # имена уже канонизированы на этапе rebuild
            return str(nd.get("agent_name", "unknown"))

        # 1) агрегируем узлы
        bag: Dict[str, Dict[str, Any]] = {}
        for _, nd in G.nodes(data=True):
            k = key_of(nd)
            if not k:
                continue
            b = bag.setdefault(k, {
                "agent_name": k if group_by == "agent_name" else nd.get("agent_name", k),
                "agent_type": nd.get("agent_type", "custom") if group_by == "agent_type" else "custom",
                "count": 0, "durations": [], "successes": []
            })
            b["count"] += 1
            if nd.get("duration") is not None:
                b["durations"].append(float(nd["duration"]))
            if "success" in nd:
                b["successes"].append(bool(nd["success"]))

        for k, b in bag.items():
            avg_dur = sum(b["durations"]) / len(b["durations"]) if b["durations"] else None
            succ = sum(b["successes"]) / len(b["successes"]) if b["successes"] else None
            H.add_node(
                k,
                agent_name=b["agent_name"],
                agent_type=b["agent_type"] if group_by == "agent_type" else "custom",
                count=b["count"],
                duration=avg_dur,
                success=True if succ is None else succ
            )

        # 2) агрегируем рёбра
        for u, v in G.edges():
            nu, nv = key_of(G.nodes[u]), key_of(G.nodes[v])
            if nu and nv and nu != nv:
                H.add_edge(nu, nv, weight=H.get_edge_data(nu, nv, {}).get("weight", 0) + 1)

        return H



    def _canon_name(self, name: str) -> str:
        """Единый канонический вид имени для всех представлений."""
        if not name:
            return "unknown"
        # убираем http_call::*
        if name.startswith("http_call::"):
            name = name[len("http_call::"):]
        # убираем суффиксы после :: (например ::HTTP POST)
        name = name.split("::")[0]
        # убираем /execute и прочие суффиксы после /
        name = name.split("/")[0]
        return name

    def _is_http_wrapper_start(self, e: "TraceEvent") -> bool:
        """HTTP-обёртка удаляется из invocations-графа (будем пересвязывать родителя с её детьми)."""
        return (
            e.event_type in (EventType.AGENT_START, EventType.TOOL_START)
            and (
                e.agent_name.startswith("http_call::")
                or (e.event_type == EventType.TOOL_START and (e.data or {}).get("tool_name") == "HTTP POST")
            )
        )

    def _is_ephemeral_step(self, start_ev: "TraceEvent", end_by_parent: dict) -> bool:
        """
        Мгновенные служебные шаги (например prepare_prompt с duration=0) выкидываем из графа.
        """
        if start_ev.event_type not in (EventType.AGENT_START, EventType.TOOL_START):
            return False
        nm = start_ev.agent_name
        if self._canon_name(nm) in {"prepare_prompt"}:
            end_ev = end_by_parent.get(start_ev.event_id)
            # duration == 0 или end не найден — считаем шумом
            return not end_ev or (end_ev.duration is not None and float(end_ev.duration) == 0.0)
        return False


    def get_call_graph_viz(
        self,
        output_file: Optional[str] = None,
        mode: str = "invocations",  # "aggregated" | "invocations"
        include_types: Optional[Set[str]] = None,
        hide_http_tools: bool = False,
        group_by: str = "agent_name",
        edge_width_scale: float = 2.0,
        arrow_size: float = 2.0,
        height: int = 900,             # <<< новая высота по умолчанию
        label_font_size: int = 12       # <<< размер шрифта меток узлов
    ) -> go.Figure:
        if not self.call_graph.nodes():
            return go.Figure()

        # --- вспомогательная функция контраста текста с цветом узла
        def _label_color_for(hex_color: str) -> str:
            try:
                hex_color = hex_color.lstrip('#')
                r = int(hex_color[0:2], 16) / 255.0
                g = int(hex_color[2:4], 16) / 255.0
                b = int(hex_color[4:6], 16) / 255.0
            except Exception:
                return "black"

            def _linear(c):
                return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

            R, G, B = _linear(r), _linear(g), _linear(b)
            # относительная яркость по WCAG
            L = 0.2126 * R + 0.7152 * G + 0.0722 * B
            return "black" if L > 0.6 else "white"

        graph = self.call_graph.copy()
        fallback_note = False

        if mode == "aggregated":
            graph = self._build_aggregated_graph(
                include_types=include_types,
                hide_http_tools=hide_http_tools,
                group_by=group_by,
            )
        elif mode == "invocations" and hide_http_tools:
            def _hide(nd):
                name = str(nd.get("agent_name", ""))
                at = str(nd.get("agent_type", ""))
                return name.startswith("http_call::") or name.endswith("::HTTP POST") or at == "tool"
            G2 = self._contract_hidden_nodes(graph, _hide)
            if G2.number_of_nodes() == 0:
                fallback_note = True
            else:
                graph = G2

        if graph.number_of_nodes() == 0:
            return go.Figure()

        pos = nx.spring_layout(graph, k=4, iterations=100, seed=42)
        fig = go.Figure()

        color_map = {
            "orchestrator": "#FF6B6B",
            "search": "#4ECDC4",
            "analysis": "#45B7D1",
            "report": "#96CEB4",
            "clarification": "#FECA57",
            "tool": "#DDA0DD",
            "custom": "#98D8C8",
        }
        symbol_map = {
            "orchestrator": "diamond",
            "search": "circle",
            "analysis": "square",
            "report": "triangle-up",
            "clarification": "star",
            "tool": "hexagon",
            "custom": "circle",
        }
        arrow_annotations = []
        approve_annotations = [] 
        # 1) линии рёбер + стрелочные аннотации поверх
        for u, v in graph.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_success = graph.nodes[v].get("success", True)
            edge_color = "#2ECC71" if edge_success else "#E74C3C"

        # линия ребра (под узлами — ок)
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(width=edge_width_scale, color=edge_color),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            # стрелка поверх: аннотация с «arrowhead»
            arrow_annotations.append(
                dict(
                    x=x1, y=y1, ax=x0, ay=y0,
                    xref="x", yref="y", axref="x", ayref="y",
                    arrowhead=3, arrowsize=arrow_size, arrowwidth=2,
                    arrowcolor=edge_color, opacity=0.95,
                    standoff=12  # чтобы наконечник не утыкался в центр узла
                )
            )

        
        # 3) узлы
        for n in graph.nodes():
            x, y = pos[n]
            nd = graph.nodes[n]
            agent_name = nd.get("agent_name", "Unknown")
            agent_type = nd.get("agent_type", "custom")
            duration = nd.get("duration", 0) or nd.get("count", 0)
            success = nd.get("success", True)
            start_time = nd.get("start_time", 0)

            if "count" in nd:
                node_size = max(24, min(64, 18 + 4 * nd["count"]))
            else:
                node_size = max(20, min(60, duration * 100)) if duration else 26

            node_color = color_map.get(agent_type, "#98D8C8")
            node_symbol = symbol_map.get(agent_type, "circle")
            text_color = _label_color_for(node_color)  # <<< авто-выбор цвета текста
            line_color = "#E74C3C" if not success else "white"
            line_width = 4 if not success else 2

            if "count" in nd:
                hover_text = f"""
                <b>{agent_name}</b><br>
                Тип: {agent_type}<br>
                Вызовов: {nd.get('count', 0)}<br>
                Ср. длительность: {nd.get('duration', 0) or 0:.3f}с<br>
                Успех: {nd.get('success', True)}
                """
            else:
                hover_text = f"""
                <b>{agent_name}</b><br>
                Тип: {agent_type}<br>
                Время выполнения: {(duration or 0):.3f}с<br>
                Статус: {'✅ Успех' if success else '❌ Ошибка'}<br>
                Время запуска: {datetime.fromtimestamp(start_time).strftime('%H:%M:%S') if start_time else 'N/A'}
                """

            fig.add_trace(
                go.Scatter(
                    x=[x], y=[y],
                    mode="markers+text",
                    marker=dict(
                        size=node_size,
                        color=node_color,
                        symbol=node_symbol,
                        line=dict(width=line_width, color=line_color),
                        opacity=0.9 if success else 0.7,
                    ),
                    text=agent_name,
                    textposition="middle center",
                    textfont=dict(size=label_font_size, color=text_color),  # <<< читаемый цвет
                    hovertext=hover_text,
                    hoverinfo="text",
                    name=agent_name,
                    showlegend=False,
                )
            )
            if nd.get("approved"):
                approve_annotations.append(dict(
                    x=x, y=y + (node_size/80.0),
                    xref="x", yref="y",
                    text="approve ✅",
                    showarrow=False,
                    font=dict(size=max(10, label_font_size-1), color="#2c7"),
                    align="center",
                    bgcolor="rgba(255,255,255,0.75)",
                    bordercolor="rgba(0,0,0,0.1)",
                    borderwidth=1,
                    opacity=0.95,
                ))

        # 4) легенда
        for at, color in color_map.items():
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode="markers",
                    marker=dict(size=15, color=color, symbol=symbol_map.get(at, "circle")),
                    name=at.title(),
                    showlegend=True,
                )
            )

        annotations = arrow_annotations + approve_annotations + [
            dict(
                text="🟢 Зеленые стрелки = успешное выполнение | 🔴 Красные стрелки = ошибка",
                showarrow=False, xref="paper", yref="paper",
                x=0.5, y=-0.06, xanchor="center", yanchor="bottom",
                font=dict(color="#666", size=12),
            )
        ]
        if fallback_note:
            annotations.append(
                dict(
                    text="Фильтр оставил пустой граф — показан исходный без скрытия.",
                    showarrow=False, xref="paper", yref="paper",
                    x=0.5, y=-0.11, xanchor="center", yanchor="bottom",
                    font=dict(color="#a33", size=12),
                )
            )

        fig.update_layout(
            title=dict(text="Multi-Agent System Call Graph", x=0.5, xanchor="center", font=dict(size=20)),
            showlegend=True,
            legend=dict(
                yanchor="top", y=0.99, xanchor="left", x=0.01,
                bgcolor="rgba(255,255,255,0.8)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1
            ),
            hovermode="closest",
            margin=dict(b=70, l=5, r=5, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(240,240,240,0.8)",
            height=height,   # <<< увеличенная высота
            annotations=annotations,
        )

        if output_file:
            fig.write_html(output_file)
        return fig

    # # --------------------------
    # # Диаграммы/статистика
    # # --------------------------
    def get_sequence_diagram(self, output_file: Optional[str] = None) -> go.Figure:
        if not self.events:
            return go.Figure()

        # индексы START/END
        starts: Dict[str, TraceEvent] = {}
        ends_by_parent: Dict[str, TraceEvent] = {}
        for e in self.events:
            if e.event_type in (EventType.AGENT_START, EventType.TOOL_START):
                starts[e.event_id] = e
            elif e.event_type in (EventType.AGENT_END, EventType.TOOL_END) and e.parent_event_id:
                ends_by_parent[e.parent_event_id] = e

        # видимые старты (те же правила, что в rebuild)
        visible_start_ids: Set[str] = set()
        for sid, s in starts.items():
            if self._is_http_wrapper_start(s) or self._is_ephemeral_step(s, ends_by_parent):
                continue
            visible_start_ids.add(sid)

        # сгруппируем по каноническим именам: учитываем ТОЛЬКО видимые START/END
        evs_sorted = sorted(self.events, key=lambda x: x.timestamp)
        agents: Dict[str, list] = {}
        agent_y: Dict[str, int] = {}
        y = 0

        def canon(name: str) -> str:
            return self._canon_name(name)

        for e in evs_sorted:
            if e.event_type in (EventType.AGENT_START, EventType.TOOL_START):
                if e.event_id not in visible_start_ids:
                    continue
            elif e.event_type in (EventType.AGENT_END, EventType.TOOL_END):
                if e.parent_event_id not in visible_start_ids:
                    continue
            else:
                continue  # MESSAGE/ERROR/CUSTOM — обработаем позже в стрелках

            nm = canon(e.agent_name)
            if nm not in agents:
                agents[nm] = []
                agent_y[nm] = y
                y += 1
            agents[nm].append(e)

        if not agents:
            return go.Figure()

        t0 = min(e.timestamp for e in evs_sorted)
        t1 = max(e.timestamp for e in evs_sorted)

        fig = go.Figure()

        # lifelines + activation boxes + точки START/END
        for nm, yy in agent_y.items():
            fig.add_trace(go.Scatter(
                x=[datetime.fromtimestamp(t0), datetime.fromtimestamp(t1)],
                y=[yy, yy], mode="lines",
                line=dict(color="lightgray", width=1, dash="dash"),
                showlegend=False, hoverinfo="skip"
            ))
            for e in agents[nm]:
                if e.event_type in (EventType.AGENT_START, EventType.TOOL_START):
                    end = ends_by_parent.get(e.event_id)
                    if end:
                        fig.add_shape(
                            type="rect",
                            x0=datetime.fromtimestamp(e.timestamp),
                            y0=yy - 0.12,
                            x1=datetime.fromtimestamp(end.timestamp),
                            y1=yy + 0.12,
                            fillcolor="lightblue",
                            opacity=0.3, layer="below", line_width=0
                        )
                    fig.add_trace(go.Scatter(
                        x=[datetime.fromtimestamp(e.timestamp)], y=[yy],
                        mode="markers",
                        marker=dict(size=12, color="#2ECC71", symbol="circle", line=dict(width=2, color="white")),
                        hovertext=f"<b>{nm}</b><br>START<br>{datetime.fromtimestamp(e.timestamp).strftime('%H:%M:%S.%f')[:-3]}",
                        hoverinfo="text", showlegend=False
                    ))
                elif e.event_type in (EventType.AGENT_END, EventType.TOOL_END):
                    fig.add_trace(go.Scatter(
                        x=[datetime.fromtimestamp(e.timestamp)], y=[yy],
                        mode="markers",
                        marker=dict(size=12, color="#3498DB", symbol="square", line=dict(width=2, color="white")),
                        hovertext=f"<b>{nm}</b><br>END<br>{datetime.fromtimestamp(e.timestamp).strftime('%H:%M:%S.%f')[:-3]}",
                        hoverinfo="text", showlegend=False
                    ))

        # стрелки: сначала структурные (parent START -> child START), потом message_sent
        # -- поднимаем родителя к видимому предку
        def lift_to_visible(start_id: Optional[str]) -> Optional[str]:
            cur = start_id
            visited = set()
            while cur and cur not in visible_start_ids and cur not in visited:
                visited.add(cur)
                if cur not in starts:
                    return None
                cur = starts[cur].parent_event_id
            return cur if cur in visible_start_ids else None

        idx = {e.event_id: e for e in evs_sorted}
        added_event_pairs = set()     # (parent_start_id, child_start_id)
        struct_name_pairs = set()     # (from_name, to_name) — для блокировки дубля message_sent

        for sid in visible_start_ids:
            s = starts[sid]
            raw_parent = s.parent_event_id
            if not raw_parent or raw_parent not in idx:
                continue
            vis_parent = lift_to_visible(raw_parent)
            if not vis_parent:
                continue
            p = starts[vis_parent]
            frm, to = canon(p.agent_name), canon(s.agent_name)
            if frm in agent_y and to in agent_y and (vis_parent, sid) not in added_event_pairs and frm != to:
                added_event_pairs.add((vis_parent, sid))
                struct_name_pairs.add((frm, to))
                fig.add_annotation(
                    x=datetime.fromtimestamp(s.timestamp), y=agent_y[to],
                    ax=datetime.fromtimestamp(p.timestamp), ay=agent_y[frm],
                    xref="x", yref="y", axref="x", ayref="y",
                    arrowhead=3, arrowsize=1.0, arrowwidth=1.4, arrowcolor="#3498DB", opacity=0.85
                )

        # message_sent — рисуем только если НЕТ структурной связи между этими именами
        for e in evs_sorted:
            if e.event_type != EventType.MESSAGE_SENT:
                continue
            frm = canon((e.data or {}).get("from_agent", ""))
            to  = canon((e.data or {}).get("to_agent", ""))
            if not frm or not to or frm == to:
                continue
            if frm in agent_y and to in agent_y and (frm, to) not in struct_name_pairs:
                t = datetime.fromtimestamp(e.timestamp)
                fig.add_annotation(
                    x=t, y=agent_y[to], ax=t, ay=agent_y[frm],
                    xref="x", yref="y", axref="x", ayref="y",
                    arrowhead=3, arrowsize=1.0, arrowwidth=1.2, arrowcolor="#9B59B6", opacity=0.85, standoff=4
                )

        fig.update_layout(
            title=dict(text="📊 Agent Execution Sequence Diagram", x=0.5, xanchor="center", font=dict(size=18)),
            xaxis=dict(title="Время", tickformat="%H:%M:%S", showgrid=True, gridcolor="lightgray"),
            yaxis=dict(
                title="Агенты",
                tickmode="array",
                tickvals=list(agent_y.values()),
                ticktext=list(agent_y.keys()),
                showgrid=False
            ),
            hovermode="closest",
            height=max(400, len(agent_y) * 80),
            showlegend=False
        )

        if output_file:
            fig.write_html(output_file)
        return fig



    def get_timeline_viz(self, output_file: Optional[str] = None) -> go.Figure:
        if not self.events:
            return go.Figure()

        # те же правила видимости, что и в rebuild/sequence
        starts: Dict[str, TraceEvent] = {}
        ends: Dict[str, TraceEvent] = {}
        for e in self.events:
            if e.event_type in (EventType.AGENT_START, EventType.TOOL_START):
                starts[e.event_id] = e
            elif e.event_type in (EventType.AGENT_END, EventType.TOOL_END) and e.parent_event_id:
                ends[e.parent_event_id] = e

        visible: Dict[str, TraceEvent] = {}
        for sid, s in starts.items():
            if self._is_http_wrapper_start(s) or self._is_ephemeral_step(s, ends):
                continue
            visible[sid] = s

        groups: Dict[str, list] = {}
        for sid, s in visible.items():
            end = ends.get(sid)
            if not end:
                continue
            nm = self._canon_name(s.agent_name)
            groups.setdefault(nm, []).append((s, end))

        if not groups:
            return go.Figure()

        agents = list(groups.keys())
        fig = go.Figure()
        palette = px.colors.qualitative.Set3

        for i, name in enumerate(agents):
            for s, e in sorted(groups[name], key=lambda t: t[0].timestamp):
                fig.add_trace(go.Scatter(
                    x=[datetime.fromtimestamp(s.timestamp), datetime.fromtimestamp(e.timestamp)],
                    y=[i, i],
                    mode="lines+markers",
                    line=dict(width=8, color=palette[i % len(palette)]),
                    marker=dict(size=6),
                    name=name,
                    hovertemplate=f"{name}<br>Duration: {(e.duration or 0):.2f}s<br>Success: {bool(e.success)}<extra></extra>",
                    showlegend=False
                ))

        fig.update_layout(
            title="Multi-Agent System Timeline",
            xaxis_title="Time",
            yaxis_title="Agents",
            yaxis=dict(tickmode="array", tickvals=list(range(len(agents))), ticktext=agents),
            hovermode="closest",
            height=max(400, len(agents) * 60),
        )
        if output_file:
            fig.write_html(output_file)
        return fig


    # --------------------------
    # Статистика
    # --------------------------

    def get_statistics(self) -> dict:
        if not self.events:
            return {
                "total_session_time": 0,
                "total_agents": 0,
                "total_events": 0,
                "successful_events": 0,
                "failed_events": 0,
                "success_rate": 0,
                "agent_performance": {},
            }

        total_events = len(self.events)
        # успехи считаем по *_END событиям
        end_events = [e for e in self.events if e.event_type in (EventType.AGENT_END, EventType.TOOL_END)]
        successful_events = sum(1 for e in end_events if bool(e.success))
        failed_events = len(end_events) - successful_events

        timestamps = [e.timestamp for e in self.events]
        total_session_time = max(timestamps) - min(timestamps) if timestamps else 0

        unique_agents = set(e.agent_name for e in self.events)

        agent_performance: Dict[str, Dict[str, Any]] = {}
        for agent_name in unique_agents:
            agent_events = [e for e in self.events if e.agent_name == agent_name]
            starts = [e for e in agent_events if e.event_type == EventType.AGENT_START]
            ends = [e for e in agent_events if e.event_type == EventType.AGENT_END]

            durations = []
            successes = []
            for s in starts:
                end = next((ee for ee in ends if ee.parent_event_id == s.event_id), None)
                if end:
                    if end.duration is not None:
                        durations.append(end.duration)
                    successes.append(bool(end.success))

            avg_duration = (sum(durations) / len(durations)) if durations else 0
            success_rate = (sum(successes) / len(successes)) if successes else 1.0

            agent_performance[agent_name] = {
                "avg_duration": avg_duration,
                "success_rate": success_rate,
                "total_events": len(agent_events),
                "executions": len(durations),
            }

        return {
            "total_session_time": total_session_time,
            "total_agents": len(unique_agents),
            "total_events": total_events,
            "successful_events": successful_events,
            "failed_events": failed_events,
            "success_rate": successful_events / len(end_events) if end_events else 0,
            "agent_performance": agent_performance,
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        if not self.events:
            return {}

        stats = {"total_events": len(self.events), "agents": {}, "session_duration": 0, "error_count": 0}
        agent_stats: Dict[str, Dict[str, Any]] = {}

        for e in self.events:
            a = e.agent_name
            agent_stats.setdefault(a, {"calls": 0, "total_duration": 0.0, "errors": 0, "avg_duration": 0.0})
            if e.event_type == EventType.AGENT_START:
                agent_stats[a]["calls"] += 1
            elif e.event_type == EventType.AGENT_END and e.duration:
                agent_stats[a]["total_duration"] += e.duration
            elif e.event_type == EventType.ERROR:
                agent_stats[a]["errors"] += 1
                stats["error_count"] += 1

        for a, d in agent_stats.items():
            d["avg_duration"] = (d["total_duration"] / d["calls"]) if d["calls"] else 0.0

        stats["agents"] = agent_stats

        if self.events:
            start_time = min(e.timestamp for e in self.events)
            end_time = max(e.timestamp for e in self.events)
            stats["session_duration"] = end_time - start_time
        return stats

    # --------------------------
    # Загрузка/выгрузка
    # --------------------------

    def export_traces(self, filename: str, format: str = "json"):
        if format == "json":
            with open(filename, "w", encoding="utf-8") as f:
                data = {
                    "session_id": self.session_id,
                    "events": [e.to_dict() for e in self.events],
                    "stats": self.get_performance_stats(),
                }
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format == "csv":
            df = pd.DataFrame([e.to_dict() for e in self.events])
            df.to_csv(filename, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def load_traces(self, filename: str):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.session_id = data.get("session_id", self.session_id)

        for ed in data.get("events", []):
            ev = TraceEvent(
                event_id=ed["event_id"],
                timestamp=ed["timestamp"],
                event_type=EventType(ed["event_type"]),
                agent_name=ed["agent_name"],
                agent_type=AgentType(ed["agent_type"]),
                data=ed.get("data", {}),
                parent_event_id=ed.get("parent_event_id"),
                session_id=ed.get("session_id", self.session_id),
                duration=ed.get("duration"),
                success=ed.get("success", True),
            )
            self.events.append(ev)

        self._rebuild_call_graph()

    def _rebuild_call_graph(self):
        """
        Полная реконструкция invocations-графа из self.events с чисткой:
        - убираем http-обёртки (http_call::*, ::HTTP POST), пересвязываем родителя с детьми;
        - убираем мгновенные служебные шаги (например prepare_prompt с duration=0);
        - у всех узлов используем канонические имена (без http_call:: и /execute).
        """
        self.call_graph.clear()
        if not self.events:
            return

        # 1) индексы
        starts: Dict[str, TraceEvent] = {}
        end_by_parent: Dict[str, TraceEvent] = {}
        children: Dict[str, list] = {}

        for e in self.events:
            if e.event_type in (EventType.AGENT_START, EventType.TOOL_START):
                starts[e.event_id] = e
            elif e.event_type in (EventType.AGENT_END, EventType.TOOL_END) and e.parent_event_id:
                end_by_parent[e.parent_event_id] = e

        for s in starts.values():
            if s.parent_event_id:
                children.setdefault(s.parent_event_id, []).append(s.event_id)

        # 2) решаем, какие стартовые узлы оставлять
        keep: Set[str] = set()
        drop: Set[str] = set()
        for sid, s in starts.items():
            if self._is_http_wrapper_start(s) or self._is_ephemeral_step(s, end_by_parent):
                drop.add(sid)
            else:
                keep.add(sid)

        # 3) создаём узлы для keep с каноническими именами
        for sid in keep:
            s = starts[sid]
            self.call_graph.add_node(
                sid,
                agent_name=self._canon_name(s.agent_name),
                agent_type=(s.agent_type.value if isinstance(s.agent_type, AgentType) else str(s.agent_type)),
                start_time=s.timestamp,
            )

        # 4) вспомогательный «подъём» к ближ. видимому предку
        def lift_to_visible(start_id: Optional[str]) -> Optional[str]:
            cur = start_id
            visited = set()
            while cur and cur not in keep and cur not in visited:
                visited.add(cur)
                if cur not in starts:
                    return None
                cur = starts[cur].parent_event_id
            return cur if cur in keep else None

        # 5) рёбра parent -> child (для видимых)
        for sid, s in starts.items():
            if sid in drop:
                continue
            parent_vis = lift_to_visible(s.parent_event_id)
            if parent_vis and parent_vis in self.call_graph and sid in self.call_graph and parent_vis != sid:
                self.call_graph.add_edge(parent_vis, sid)

        # 6) рёбра через скрытые узлы: предок(hid) -> внук(child)
        for hid in drop:
            # ближайший видимый предок скрытого узла
            pv = lift_to_visible(starts[hid].parent_event_id if hid in starts else None)
            if not pv:
                continue
            for ch in children.get(hid, []):
                if ch in keep and pv in self.call_graph and ch in self.call_graph and pv != ch:
                    self.call_graph.add_edge(pv, ch)

        # 7) duration/success/end_time из *_END
        for sid in keep:
            end = end_by_parent.get(sid)
            if end and sid in self.call_graph:
                nd = self.call_graph.nodes[sid]
                nd["duration"] = end.duration
                nd["success"] = bool(end.success)
                nd["end_time"] = end.timestamp




# ============================================================
# Декораторы для удобной интеграции
# ============================================================

def trace_agent(tracer: MultiAgentTracer, agent_name: str, agent_type: AgentType):
    """Декоратор для автоматической трассировки агентов"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            event_id = tracer.start_trace(
                agent_name=agent_name,
                agent_type=agent_type,
                data={"args": str(args), "kwargs": str(kwargs)}
            )
            
            try:
                result = func(*args, **kwargs)
                tracer.end_trace(event_id, {"result": str(result)}, success=True)
                return result
            except Exception as e:
                tracer.log_error(agent_name, e, event_id)
                tracer.end_trace(event_id, {"error": str(e)}, success=False)
                raise
        
        return wrapper
    return decorator


def trace_async_agent(tracer: MultiAgentTracer, agent_name: str, agent_type: AgentType):
    """Декоратор для автоматической трассировки асинхронных агентов"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            event_id = tracer.start_trace(
                agent_name=agent_name,
                agent_type=agent_type,
                data={"args": str(args), "kwargs": str(kwargs)}
            )
            
            try:
                result = await func(*args, **kwargs)
                tracer.end_trace(event_id, {"result": str(result)}, success=True)
                return result
            except Exception as e:
                tracer.log_error(agent_name, e, event_id)
                tracer.end_trace(event_id, {"error": str(e)}, success=False)
                raise
        
        return wrapper
    return decorator


# ============================================================
# Интеграция с LangGraph
# ============================================================

class LangGraphTracer:
    """Специализированный трассировщик для LangGraph"""
    
    def __init__(self, base_tracer: MultiAgentTracer):
        self.tracer = base_tracer
        self.node_events = {}
    
    def trace_node(self, node_name: str, agent_type: AgentType = AgentType.CUSTOM):
        """Трассировка узла LangGraph"""
        def decorator(func):
            def wrapper(state):
                event_id = self.tracer.start_trace(
                    agent_name=node_name,
                    agent_type=agent_type,
                    data={"state": str(state)}
                )
                
                try:
                    result = func(state)
                    self.tracer.end_trace(event_id, {"result": str(result)}, success=True)
                    return result
                except Exception as e:
                    self.tracer.log_error(node_name, e, event_id)
                    self.tracer.end_trace(event_id, {"error": str(e)}, success=False)
                    raise
            
            return wrapper
        return decorator
    
    def get_sequence_diagram(self, output_file: Optional[str] = None) -> go.Figure:
        """Создание диаграммы последовательности выполнения агентов"""
        if not self.events:
            print("Нет событий для создания диаграммы последовательности")
            return go.Figure()
        
        # Сортируем события по времени
        sorted_events = sorted(self.events, key=lambda x: x.timestamp)
        
        # Создаем временную ось
        fig = go.Figure()
        
        # Группируем события по агентам
        agents = {}
        agent_positions = {}
        y_pos = 0
        
        for event in sorted_events:
            if event.agent_name not in agents:
                agents[event.agent_name] = []
                agent_positions[event.agent_name] = y_pos
                y_pos += 1
            agents[event.agent_name].append(event)
        
        # Цвета для разных типов событий
        event_colors = {
            'AGENT_START': '#2ECC71',
            'AGENT_END': '#3498DB', 
            'TOOL_START': '#F39C12',
            'TOOL_END': '#E67E22',
            'MESSAGE_SENT': '#9B59B6',
            'ERROR': '#E74C3C',
            'CUSTOM': '#95A5A6'
        }
        
        # Добавляем временные линии для каждого агента
        for agent_name, y_position in agent_positions.items():
            agent_events = agents[agent_name]
            
            # Линия жизни агента
            fig.add_trace(go.Scatter(
                x=[datetime.fromtimestamp(sorted_events[0].timestamp),
                    datetime.fromtimestamp(sorted_events[-1].timestamp)],
                y=[y_position, y_position],
                mode='lines',
                line=dict(color='lightgray', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # События агента
            for event in agent_events:
                color = event_colors.get(event.event_type.name, '#95A5A6')
                
                # Размер маркера зависит от типа события
                marker_size = 15 if event.event_type.name in ['AGENT_START', 'AGENT_END'] else 8
                
                # Символ маркера
                if event.event_type.value == 'ERROR':
                    symbol = 'x'
                elif event.event_type.value in ['AGENT_START', 'TOOL_START']:
                    symbol = 'circle'
                elif event.event_type.value in ['AGENT_END', 'TOOL_END']:
                    symbol = 'square'
                else:
                    symbol = 'diamond'
                
                # Hover информация
                hover_text = f"""
                <b>{event.agent_name}</b><br>
                Событие: {event.event_type.value}<br>
                Время: {datetime.fromtimestamp(event.timestamp).strftime('%H:%M:%S.%f')[:-3]}<br>
                Данные: {str(event.data)[:100]}{'...' if len(str(event.data)) > 100 else ''}
                """
                
                fig.add_trace(go.Scatter(
                    x=[datetime.fromtimestamp(event.timestamp)],
                    y=[y_position],
                    mode='markers',
                    marker=dict(
                        size=marker_size,
                        color=color,
                        symbol=symbol,
                        line=dict(width=2, color='white')
                    ),
                    name=event.event_type.value,
                    hovertext=hover_text,
                    hoverinfo='text',
                    showlegend=False
                ))
        
        # Добавляем стрелки для связей между агентами
        for event in sorted_events:
            if event.parent_event_id:
                # Находим родительское событие
                parent_event = next((e for e in sorted_events if e.event_id == event.parent_event_id), None)
                if parent_event and parent_event.agent_name != event.agent_name:
                    # Добавляем стрелку между агентами
                
                    fig.add_annotation(
                        x=datetime.fromtimestamp(event.timestamp),
                        y=agent_positions[event.agent_name],
                        ax=datetime.fromtimestamp(parent_event.timestamp),
                        ay=agent_positions[parent_event.agent_name],
                        xref='x', yref='y',
                        axref='x', ayref='y',
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=1.5,
                        arrowcolor='#3498DB',
                        opacity=0.7
                    )
        for e in sorted_events:
            if e.event_type == EventType.MESSAGE_SENT:
                frm = (e.data or {}).get("from_agent")
                to  = (e.data or {}).get("to_agent")
                if frm in agent_positions and to in agent_positions:
                    t = datetime.fromtimestamp(e.timestamp)
                    fig.add_annotation(
                        x=t, y=agent_positions[to],
                        ax=t, ay=agent_positions[frm],
                        xref="x", yref="y", axref="x", ayref="y",
                        arrowhead=3, arrowsize=1.0, arrowwidth=1.2,
                        arrowcolor="#9B59B6",  # тот же цвет, что и в legend для MESSAGE_SENT
                        opacity=0.85, standoff=4
                    )
        # Создаем легенду для типов событий
        for event_type, color in event_colors.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=event_type,
                showlegend=True
            ))
        
        # Настройка макета
        fig.update_layout(
            title={
                'text': '📊 Agent Execution Sequence Diagram',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis=dict(
                title='Время выполнения',
                tickformat='%H:%M:%S',
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title='Агенты',
                tickmode='array',
                tickvals=list(agent_positions.values()),
                ticktext=list(agent_positions.keys()),
                showgrid=False
            ),
            hovermode='closest',
            height=max(400, len(agents) * 80),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        if output_file:
            fig.write_html(output_file)
        
        return fig
