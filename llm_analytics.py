# llm_analytics.py
# Агент-аналитик: принимает JSON трассы MultiAgentTracer, вызывает LLM по чанкам,
# складывает оценки в поле "assessments" и возвращает новый JSON.
# Также умеет работать в офлайн-фолбэке (без API-ключа) — проставляет approve=успех.

from __future__ import annotations
import os, json, uuid
from typing import Dict, Any, List, Tuple, Set, Optional

# Опциональная зависимость: LangChain ChatOpenAI
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:
    ChatOpenAI = None

MODEL_NAME_DEFAULT = os.getenv("LLM_MODEL", "gpt-4o-mini")
MAX_EVENTS_PER_CHUNK = 14  # чуть больше, чем было

# ──────────────────────────────────────────────────────────────────────────────
# Помощники
# ──────────────────────────────────────────────────────────────────────────────

def _build_success_map(trace: Dict[str, Any]) -> Dict[str, bool]:
    """По исходной трассе строим карту: start_event_id -> success (по *_END/TOOL_END)."""
    success_by_parent: Dict[str, bool] = {}
    for e in trace.get("events", []):
        et = str(e.get("event_type", "")).lower()
        if et in ("agent_end", "tool_end"):
            parent = e.get("parent_event_id")
            success_by_parent[parent] = bool(e.get("success", True))
    return success_by_parent

def _session_view_from_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Сужаем событие до полей, полезных LLM (уменьшаем шум)."""
    session = {
        "session_id": trace.get("session_id", str(uuid.uuid4())),
        "events": []
    }
    for e in trace.get("events", []):
        session["events"].append({
            "event_id": e.get("event_id"),
            "timestamp": e.get("timestamp"),
            "event_type": e.get("event_type"),
            "agent_name": e.get("agent_name"),
            "agent_type": e.get("agent_type"),
            "data": {
                "message": (e.get("data") or {}).get("message"),
                "tool_name": (e.get("data") or {}).get("tool_name"),
                "input": (e.get("data") or {}).get("input"),
                "output": (e.get("data") or {}).get("output"),
            },
            "parent_event_id": e.get("parent_event_id"),
        })
    return session

def _extract_approved_ids(assessed_events: List[Dict[str, Any]]) -> Set[str]:
    """Собираем event_id, которым LLM поставила verdict='approve'."""
    approved: Set[str] = set()
    for item in assessed_events:
        ev_id = item.get("event_id")
        verdict = ((item.get("assessment") or {}).get("verdict") or "").lower()
        if ev_id and verdict == "approve":
            approved.add(ev_id)
    return approved

# ──────────────────────────────────────────────────────────────────────────────
# Основная функция LLM-аналитики
# ──────────────────────────────────────────────────────────────────────────────

def classify_session_with_llm(
    session_json: Dict[str, Any],
    *,
    model_name: str = MODEL_NAME_DEFAULT,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional["ChatOpenAI"] = None,
) -> Dict[str, Any]:
    """
    Вызывает LLM по чанкам и возвращает словарь:
    { "events":[{"event_id":..., "assessment":{...}}, ...],
      "session_assessment": {...} }
    Если ChatOpenAI недоступен/нет ключа — офлайн-фолбэк: approve по успеху.
    
    Для OpenRouter: Укажите base_url="https://openrouter.ai/api/v1", api_key='YOUR_KEY', model_name="qwen/qwen3-4b:free".
    """
    # Фолбэк без LLM: approve=успех (если есть карта успехов в исходной сессии)
    if model is None or ChatOpenAI is None:
        # offline: ставим approve "наугад" (или по success, если предоставили карту в session_json)
        assessed = []
        for ev in session_json.get("events", []):
            assessed.append({
                "event_id": ev.get("event_id"),
                "assessment": {"verdict": "approve"} if session_json.get("_success_map", {}).get(ev.get("event_id"), True) else {"verdict": "reject"}
            })
        return {"events": assessed, "session_assessment": {"summary": "Offline fallback: approve by success"}}

    # Если model не передан, создаем новый с учетом base_url (для OpenRouter)
    if model is None:
        if base_url:  # Для OpenRouter или аналогичных
            if not api_key:
                raise ValueError("API key required for custom base_url (e.g., OpenRouter)")
            model = ChatOpenAI(
                base_url=base_url,
                api_key=api_key,
                model=model_name,
                temperature=0
            )
        else:  # Стандартный OpenAI
            model = ChatOpenAI(
                model=model_name,
                temperature=0
            )

    # ... (остальной код classify_session_with_llm без изменений: chunks, messages, resp.invoke и т.д.)

# ──────────────────────────────────────────────────────────────────────────────
# Утилита интеграции со Streamlit: анализ «байтов» файла
# ──────────────────────────────────────────────────────────────────────────────

def analyze_trace_bytes(
    trace_bytes: bytes,
    *,
    model_name: str = MODEL_NAME_DEFAULT,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    use_llm: bool = True
) -> Tuple[bytes, Set[str]]:
    """
    Принимает байты исходного файла трассы, возвращает:
    (байты нового JSON (с assessments), set(approved_event_id))
    """
    trace = json.loads(trace_bytes.decode("utf-8"))
    session_view = _session_view_from_trace(trace)

    # в офлайне для эвристики передадим success-карту
    session_view["_success_map"] = _build_success_map(trace)

    model = None
    if use_llm and ChatOpenAI is not None and os.getenv("OPENAI_API_KEY"):
        model = ChatOpenAI(model=model_name, temperature=0)

    assessed = classify_session_with_llm(session_view, model_name=model_name, base_url=base_url, api_key=api_key, model=model)
    approved_ids = _extract_approved_ids(assessed["events"])

    # собираем новый JSON: копия трассы + блок assessments
    new_trace = trace.copy()
    new_trace["assessments"] = assessed

    out_bytes = json.dumps(new_trace, ensure_ascii=False, indent=2).encode("utf-8")
    return out_bytes, approved_ids