from __future__ import annotations

import random, time as _time
import os, json, uuid
from typing import Dict, Any, List, Tuple, Set, Optional
import re
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from jsonschema import validate, ValidationError
except Exception as e:
    print(f"Import error: {e}")
    ChatOpenAI = None

    
# llm_analytics.py
# Агент-аналитик: принимает JSON трассы MultiAgentTracer, вызывает LLM по чанкам,
# складывает оценки в поле "assessments" и возвращает новый JSON.
# Также умеет работать в офлайн-фолбэке (без API-ключа) — проставляет approve=успех.


MODEL_NAME_DEFAULT = os.getenv("LLM_MODEL", "qwen/qwen3-4b:free")
MAX_EVENTS_PER_CHUNK = 12  # оптимальный размер чанка

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "assessment": {
                        "type": "object",
                        "properties": {
                            "verdict": {"type": "string"},
                            "categories": {"type": "array"},
                            "severity": {"type": "string"},
                            "confidence": {"type": "number"},
                            "explanation": {"type": "string"},
                            "suggested_fix": {"type": "string"}
                        },
                        "required": ["verdict", "categories", "confidence"]
                    }
                },
                "required": ["event_id", "assessment"]
            }
        },
        "session_assessment": {"type": "object"}
    },
    "required": ["events", "session_assessment"]
}

# ──────────────────────────────────────────────────────────────────────────────
# Помощники
# ──────────────────────────────────────────────────────────────────────────────

def run_prechecks(session: Dict[str, Any]) -> Dict[str, Any]:
    """Выполняет предварительные проверки на корректность событий."""
    events = session.get("events", [])

    id_map = {e["event_id"]: e for e in events}
    pre_tags = {}
    
    for e in events:
        if e.get("event_type") == "message_sent":
            msg = str(e.get("data",{}).get("message",""))
            if "web_search" in msg or "SEARCH_AGENT" in msg:
                found = False
                for e2 in events:
                    if e2.get("parent_event_id") == e["event_id"] and e2.get("agent_type") == "tool":
                        found = True
                        break
                if not found:
                    pre_tags[e["event_id"]] = pre_tags.get(e["event_id"], []) + ["missing_tool_result"]

    for e in events:
        if e.get("event_type","").endswith("_end") and e.get("agent_type") == "tool":
            if e.get("data",{}).get("output") is None:
                pre_tags[e["event_id"]] = pre_tags.get(e["event_id"], []) + ["tool_no_output"]
                
    return pre_tags

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
MAX_EVENTS_PER_CHUNK = 30  # оставьте ваше значение

def classify_session_with_llm(
    session_json: Dict[str, Any],
    *,
    model_name: str = MODEL_NAME_DEFAULT,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[Any] = None,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Возвращает (assessed_dict, token_stats).
    При любой ошибке провайдера автоматически уходим в офлайн-фолбэк (approve по success).
    """
    def _try_extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Пробует вытащить JSON из ответа модели (в т.ч. из кодовых блоков)."""
        if not text:
            return None

        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    def _offline_result() -> Tuple[Dict[str, Any], Dict[str, int]]:
        success_map = session_json.get("_success_map", {})
        assessed_events: List[Dict[str, Any]] = []
        for ev in session_json.get("events", []):
            ev_id = ev.get("event_id")
            ok = success_map.get(ev_id, True)
            assessed_events.append({
                "event_id": ev_id,
                "assessment": {
                    "verdict": "approve" if ok else "reject",
                    "reason": "Offline fallback by success flag"
                }
            })
        assessed = {
            "events": assessed_events,
            "session_assessment": {
                "summary": "Offline fallback: approve by success",
                "improvement_suggestions": [
                    "Добавьте больше логирования ошибок для агентов.",
                    "Оптимизируйте последовательность вызовов между агентами.",
                    "Проверьте валидацию входных данных во всех узлах."
                ],
                "expert_review": (
                    "Сессия в целом прошла успешно. Рекомендуется усилить обработку ошибок и улучшить "
                    "диагностику причин неуспехов на уровне отдельных агентов/тулов."
                ),
            },
        }
        return assessed, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if model is None:
        return _offline_result()

    def _invoke_with_backoff(_model, _messages, max_retries: int = 4):
        last = None
        for attempt in range(max_retries + 1):
            try:
                return _model.invoke(_messages)
            except Exception as e:
                txt = str(e).lower()

                is_429 = ("rate limit" in txt) or ("429" in txt)
                is_5xx = any(s in txt for s in ["502", "503", "504"])
                if not (is_429 or is_5xx):
                    raise
                last = e
                wait = (2 ** attempt) + random.random() * 0.5
                try:
                    reset_ms = None
                    if hasattr(e, "response") and getattr(e, "response", None):
                        hdrs = getattr(e.response, "headers", {}) or {}
                        reset_ms = int(hdrs.get("X-RateLimit-Reset", "0"))
                    if reset_ms:
                        now_ms = int(_time.time() * 1000)
                        until_reset = max(0, (reset_ms - now_ms) / 1000.0)
                        wait = max(wait, min(until_reset, 90))
                except Exception:
                    pass
                if attempt < max_retries:
                    _time.sleep(wait)
                else:
                    raise last
    def _extract_json_block(text: str) -> Optional[dict]:
        """
        Пытается вытащить JSON-объект из текста:
        - ищет первый {...} блок (включая код-блоки ```json ... ```)
        - безопасно парсит в dict
        Возвращает dict или None.
        """
        if not isinstance(text, str) or not text.strip():
            return None

        fenced = re.search(r"```(?:json)?\s*({.*?})\s*```", text, flags=re.S|re.I)
        candidate = fenced.group(1) if fenced else None
        if not candidate:
            brace = re.search(r"(\{.*\})", text, flags=re.S)
            candidate = brace.group(1) if brace else None
        if not candidate:
            return None
        try:
            return json.loads(candidate)
        except Exception:
            return None

    def _coerce_session_assessment(final_text: Any) -> Dict[str, Any]:
        """
        Нормализует ответ LLM в объект с ключами summary / improvement_suggestions / expert_review.
        Принмает либо dict, либо строку.
        """
        if isinstance(final_text, dict):
            summary = final_text.get("summary") or final_text.get("overview") or final_text.get("text") or ""
            sugg = final_text.get("improvement_suggestions") or final_text.get("recommendations") or []
            review = final_text.get("expert_review") or final_text.get("analysis") or summary
            if isinstance(sugg, str):
                sugg = [s.strip("-• ").strip() for s in sugg.splitlines() if s.strip()]
            return {
                "summary": str(summary or review or ""),
                "improvement_suggestions": list(sugg) if isinstance(sugg, list) else [],
                "expert_review": str(review or summary or "")
            }

        parsed = _extract_json_block(str(final_text))
        if isinstance(parsed, dict):
            return _coerce_session_assessment(parsed)

        lines = [ln.strip() for ln in str(final_text).splitlines()]
        bullets = [ln for ln in lines if re.match(r"^(\-|\*|\d+[\.\)])\s+", ln)]
        cleaned = [re.sub(r"^(\-|\*|\d+[\.\)])\s+", "", b).strip() for b in bullets]
        return {
            "summary": str(final_text).strip(),
            "improvement_suggestions": [c for c in cleaned if c] if cleaned else [],
            "expert_review": str(final_text).strip(),
        }
    
    def _extract_usage(resp) -> Dict[str, int]:
        """
        Универсальный сбор токенов:
        - сначала пробуем стандартные места (response_metadata.usage / token_usage),
        - затем заголовки OpenRouter (x-openrouter-...),
        - затем «нативные» поля провайдера (tokens_prompt, native_tokens_*), если они прокинуты в метаданные.
        """
        prompt = 0
        completion = 0
        reasoning = 0

        meta = getattr(resp, "response_metadata", {}) or {}
        usage = (
            meta.get("token_usage")
            or meta.get("usage")
            or meta.get("openai", {}).get("usage")
            or {}
        )
        def _to_int(x): 
            try: return int(x)
            except: return 0

        prompt = _to_int(usage.get("prompt_tokens") or usage.get("input_tokens") or prompt)
        completion = _to_int(usage.get("completion_tokens") or usage.get("output_tokens") or completion)

        headers = meta.get("headers") or {}
        headers_l = {str(k).lower(): v for k, v in headers.items()} if isinstance(headers, dict) else {}

        prompt = prompt or _to_int(
            headers_l.get("x-openrouter-prompt-tokens") or
            headers_l.get("x-openrouter-tokens-prompt")
        )
        completion = completion or _to_int(
            headers_l.get("x-openrouter-completion-tokens") or
            headers_l.get("x-openrouter-tokens-completion")
        )
        reasoning = reasoning or _to_int(
            headers_l.get("x-openrouter-reasoning-tokens") or
            headers_l.get("x-openrouter-tokens-reasoning")
        )
        provider_meta = (
            meta.get("provider_meta") or
            meta.get("openrouter", {}).get("meta") or
            meta.get("venice") or
            meta.get("provider") or
            {}
        )
        if isinstance(provider_meta, dict):
            prompt = prompt or _to_int(provider_meta.get("tokens_prompt") or provider_meta.get("native_tokens_prompt"))
            completion = completion or _to_int(provider_meta.get("tokens_completion") or provider_meta.get("native_tokens_completion"))
            reasoning = reasoning or _to_int(provider_meta.get("native_tokens_reasoning"))

        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "reasoning_tokens": reasoning,
            "total_tokens": prompt + completion + reasoning,
        }

    try:
        pre_tags = run_prechecks(session_json)
    except Exception:
        pre_tags = {}

    system_prompt = (
        "You are an expert auditor for multi-agent execution traces. "
        "For each event in the provided JSON list, produce an assessment object with keys: "
        "'verdict' (approve|reject|uncertain), 'categories' (list of error categories), "
        "'severity' (low|medium|high), 'confidence' (0.0-1.0), 'explanation', 'suggested_fix'. "
        "Return JSON only: {events:[{event_id:..., assessment:{...}}], session_assessment:{...}}. "
        "Use the precheck hints when present."
    )

    events = session_json.get("events", [])
    chunk_size = MAX_EVENTS_PER_CHUNK
    event_chunks = [events[i:i + chunk_size] for i in range(0, len(events), chunk_size)]

    assessed_events: List[Dict[str, Any]] = []
    chunk_assessments: List[str] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for chunk in event_chunks:
        try:
            chunk_json = json.dumps(chunk, ensure_ascii=False, indent=2)
            human_prompt = (
                "You are scoring multi-agent trace events.\n"
                "Return STRICT JSON only with shape:\n"
                '{ "events": [ { "event_id": "<id from input>", '
                '"assessment": { "verdict": "approve|reject|uncertain", '
                '"categories": [], "severity": "low|medium|high", '
                '"confidence": 0.0, "explanation": "", "suggested_fix": "" } } ] }\n'
                "Do not include any extra text.\n\n"
                f"Precheck hints: {json.dumps(pre_tags, ensure_ascii=False)}\n\n"
                f"Events:\n{chunk_json}"
            )
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            resp = _invoke_with_backoff(model, messages, max_retries=4)
            text = resp.content if hasattr(resp, "content") else str(resp)
            usage = _extract_usage(resp)
            total_prompt_tokens += usage["prompt_tokens"]
            total_completion_tokens += usage["completion_tokens"]

            obj = _extract_json_block(text) or _try_extract_json(text)
            chunk_assessed: Dict[str, Dict[str, Any]] = {}

            if isinstance(obj, dict):
                # (опционально) строгая валидация — если есть jsonschema
                # try: validate(instance=obj, schema=OUTPUT_SCHEMA)
                # except Exception: obj = None

                for item in obj.get("events", []):
                    eid = (item or {}).get("event_id")
                    a = (item or {}).get("assessment") or {}
                    if isinstance(eid, str) and isinstance(a, dict):
                        # нормализация полей
                        verdict = str(a.get("verdict", "")).lower()
                        if verdict not in {"approve", "reject", "uncertain"}:
                            verdict = "uncertain"

                        cats = a.get("categories", [])
                        if isinstance(cats, str):
                            cats = [c.strip() for c in re.split(r"[,\n;]+", cats) if c.strip()]
                        elif not isinstance(cats, list):
                            cats = []

                        sev = str(a.get("severity", "low")).lower()
                        if sev not in {"low", "medium", "high"}:
                            sev = "low"

                        try:
                            conf = float(a.get("confidence", 0.5))
                        except Exception:
                            conf = 0.5
                        conf = max(0.0, min(1.0, conf))

                        chunk_assessed[eid] = {
                            "verdict": verdict,
                            "categories": cats,
                            "severity": sev,
                            "confidence": conf,
                            "explanation": a.get("explanation", ""),
                            "suggested_fix": a.get("suggested_fix", ""),
                        }

            # ← ЭТОГО НЕ ХВАТАЛО: сохраняем «результат чанка» для финального summary
            chunk_assessments.append(
                json.dumps(obj, ensure_ascii=False) if isinstance(obj, dict) else text
            )

        except Exception as e:
            print(f"[LLM chunk error] {e}. Falling back to offline.")
            # если хотим мгновенный оффлайн — return _offline_result()
            # либо continue с эвристикой ↓

        # Сопоставляем оценки по event_id; мягкий фолбэк для пропусков
        for ev in chunk:
            ev_id = ev.get("event_id")
            a = chunk_assessed.get(ev_id)
            if a:
                assessed_events.append({"event_id": ev_id, "assessment": a})
            else:
                assessed_events.append({
                    "event_id": ev_id,
                    "assessment": {
                        "verdict": "uncertain",
                        "explanation": "Heuristic fallback (no JSON for this id)"
                    }
                })

    try:
        final_prompt = (
            "Return STRICT JSON only with shape:\n"
            '{ "summary": "...", "improvement_suggestions": ["...","...","..."], "expert_review": "..." }.\n'
            "Do not include any extra text.\n\n"
            "Given the analyses above:"
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=final_prompt + "\n\nPrevious analyses:\n" + "\n".join(chunk_assessments)),
        ]
        final_resp = _invoke_with_backoff(model, messages, max_retries=4)
        final_text = final_resp.content if hasattr(final_resp, "content") else str(final_resp)
        usage = _extract_usage(final_resp)
        total_prompt_tokens += usage["prompt_tokens"]
        total_completion_tokens += usage["completion_tokens"]
        reasoning_here = usage.get("reasoning_tokens", 0)
    except Exception as e:
        print(f"[LLM final error] {e}. Falling back to offline.")
        return _offline_result()
    
    session_assessment = _coerce_session_assessment(final_text)
    
    if not session_assessment.get("improvement_suggestions"):
        session_assessment["improvement_suggestions"] = [
            "Optimize agent communication patterns.",
            "Implement stricter error handling and retries for tools.",
            "Add richer telemetry to trace bottlenecks and failures.",
        ]
    assessed = {"events": assessed_events, "session_assessment": session_assessment}
    token_stats = {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "reasoning_tokens": 0,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
    }

    return assessed, token_stats



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
) -> Tuple[bytes, Set[str], Dict[str, int]]:
    """
    Возвращает (байты нового JSON, set(approved_event_id), token_stats)
    """
    trace = json.loads(trace_bytes.decode("utf-8"))
    session_view = _session_view_from_trace(trace)
    session_view["_success_map"] = _build_success_map(trace)

    def _normalize_base_url(url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        u = url.rstrip("/")
        if "openrouter.ai" in u and not u.endswith("/api/v1") and not u.endswith("/v1"):
            return u + "/api/v1"
        return u

    def _maybe_fix_model_for_openrouter(mname: str, burl: Optional[str]) -> str:
        if not mname:
            return MODEL_NAME_DEFAULT
        if burl and "openrouter.ai" in burl and "/" not in mname:
            return f"openai/{mname}"
        return mname

    model = None
    token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if use_llm and ("ChatOpenAI" in globals()) and (ChatOpenAI is not None):
        base_url = _normalize_base_url(base_url or os.getenv("LLM_BASE_URL"))
        api_key = (
            api_key
            or (os.getenv("OPENROUTER_API_KEY") if (base_url and "openrouter.ai" in base_url) else None)
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
        )
        model_name = _maybe_fix_model_for_openrouter(model_name or os.getenv("LLM_MODEL") or MODEL_NAME_DEFAULT, base_url)

        if api_key:
            print(f"Initializing LLM with model={model_name}, base_url={base_url or 'default'}")
            kwargs = {
                "model": model_name,
                "temperature": 0,
                "api_key": api_key,
            }
            if base_url:
                kwargs["base_url"] = base_url
            if base_url and "openrouter.ai" in base_url:
                kwargs["default_headers"] = {
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "Tracing & Log Viz",
                }
            try:
                model = ChatOpenAI(**kwargs)
            except Exception as e:
                print(f"[LLM init error] {e}. Using offline fallback.")
                model = None
        else:
            print("API key not provided. Using offline fallback.")

    assessed, _token_stats = classify_session_with_llm(
        session_view,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        model=model,
    )

    token_stats = {
        "total_prompt_tokens": int(_token_stats.get("prompt_tokens", 0)),
        "total_completion_tokens": int(_token_stats.get("completion_tokens", 0)),
        "total_reasoning_tokens": int(_token_stats.get("reasoning_tokens", 0)),
        "total_tokens": int(_token_stats.get("total_tokens", 0)),
    }

    approved_ids = _extract_approved_ids(assessed.get("events", []))

    new_trace = trace.copy()
    new_trace["assessments"] = assessed
    new_trace["token_stats"] = token_stats

    out_bytes = json.dumps(new_trace, ensure_ascii=False, indent=2).encode("utf-8")
    return out_bytes, approved_ids, token_stats
