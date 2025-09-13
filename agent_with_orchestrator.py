import logging
import time
import threading
import uvicorn
import aiohttp
import psutil
import socket
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.llms import YandexGPT
from custom_trace import LoggingCallback, log_json, log_trace, AccessLogMiddleware
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import uuid
import aiohttp
from dotenv import load_dotenv
import os
import logging, json
import time
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from contextlib import contextmanager
from multiagent_tracer.multiagent_tracer import AgentType, MultiAgentTracer

tracer = MultiAgentTracer(
    session_id="notebook_session",
    log_file=None,                 # можно указать путь к файлу .log
    enable_real_time_viz=False,    # True — если нужно обновление графа на лету
)
@contextmanager
def trace_block(agent_name: str,
                agent_type: AgentType = AgentType.CUSTOM,
                parent_event_id: str | None = None,
                **data):
    eid = tracer.start_trace(
        agent_name=agent_name,
        agent_type=agent_type,
        data=data,
        parent_event_id=parent_event_id
    )
    try:
        yield eid
    except Exception as e:
        tracer.log_error(agent_name, e, parent_event_id=eid)
        tracer.end_trace(eid, {"error": str(e)}, success=False)
        raise
    else:
        tracer.end_trace(eid, {"success": True}, success=True)


class TaskRequest(BaseModel):
    input: str


load_dotenv()
FODLER_ID =  os.getenv("FOLDER_ID")
API_KEY =  os.getenv("API_KEY")

llm = YandexGPT(
    api_key=API_KEY,
    folder_id=FODLER_ID,
    temperature=0.2
    )
class RouteDecision(BaseModel):
    next_agent: str = Field(pattern="^(SEARCH_AGENT|ANALYSIS_AGENT|REPORT_AGENT)$")
    reason: str
    input: str

class RouterAgent:
    def __init__(self, llm):
        self.llm = llm
        self.router_prompt = PromptTemplate.from_template(
            """You are a strict router. Choose the next agent.

            Available agents:
            - SEARCH_AGENT: use ONLY when fresh/current web info is needed or no context yet.
            - ANALYSIS_AGENT: process/summarize/structure text snippets.
            - REPORT_AGENT: produce final structured answer (last step).

            Return ONLY a JSON object (no prose, no Markdown fences).

            Input query: {query}
            Current context: {context}

            JSON schema:
            {{
            "next_agent": "SEARCH_AGENT|ANALYSIS_AGENT|REPORT_AGENT",
            "reason": "string",
            "input": "string"
            }}"""
        )

    def _extract_json(self, s: str) -> dict:

        try:
            return json.loads(s)
        except Exception:
            pass

        candidates = []
        stack = []
        start = None
        for i, ch in enumerate(s):
            if ch == '{':
                if not stack:
                    start = i
                stack.append('{')
            elif ch == '}':
                if stack:
                    stack.pop()
                    if not stack and start is not None:
                        candidates.append(s[start:i+1])
        candidates.sort(key=len, reverse=True)
        for c in candidates:
            try:
                return json.loads(c)
            except Exception:
                continue
        raise ValueError("No valid JSON found in router output")

    def route(self, query: str, context: str = "No context yet"):
        try:
            chain = self.router_prompt | self.llm
            raw = chain.invoke({"query": query, "context": context})
            data = self._extract_json(raw)
            decision = RouteDecision(**data)  # валидация
            return decision.model_dump()
        except Exception as e:
            logging.error(f"Router LLM failed: {e}. Fallback engaged.")
            return self._smart_fallback(query, context)

    def _smart_fallback(self, query, context):
        q = query.lower()
        c = context.lower()

        if "search_agent output" in c or "найдено" in c or "результат поиска" in c:
            return {
                "next_agent": "ANALYSIS_AGENT",
                "reason": "Post-search analysis step",
                "input": context
            }

        if "analysis_agent output" in c or "анализ" in c or "summary" in c:
            return {
                "next_agent": "REPORT_AGENT",
                "reason": "Finalize into report",
                "input": context
            }
        # Триггеры «нужна свежесть»
        if any(t in q for t in ["нов", "свеж", "сегодня", "вчера", "202", "цена", "курс", "последн", "latest", "today", "news"]):
            return {"next_agent": "SEARCH_AGENT", "reason": "Fresh info likely needed", "input": query}
        
        return {"next_agent": "ANALYSIS_AGENT", "reason": "Direct analysis", "input": query}


TRACE_SESSION = "X-Trace-Session"
TRACE_PARENT = "X-Parent-Event"

class TraceHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        sid = request.headers.get(TRACE_SESSION)
        if sid:
            try:
                tracer.session_id = sid
            except Exception:
                pass
        response = await call_next(request)
        return response
    
search_app = FastAPI(title="search-agent")
search_app.add_middleware(AccessLogMiddleware, service_name="search-agent", log_fn=log_json)

duck = DuckDuckGoSearchRun()

translate_prompt = PromptTemplate.from_template(
    "Translate to English for web search. Output ONLY the translation, no quotes, no comments:\n{data}"
)


# 2. ПОИСКОВЫЙ АГЕНТ (отдельный сервис)
@search_app.post("/execute")
async def execute_search(task: TaskRequest, request: Request):
    parent = request.headers.get("X-Parent-Event")
    root_eid = tracer.start_trace(
        agent_name="search-agent/execute",
        agent_type=AgentType.SEARCH,
        data={
            "trace_headers_present": bool(parent),
            "input_preview": (task.input or "")[:200]
        },
        parent_event_id=parent
    )

    cb = LoggingCallback(log_fn=log_json)
    try:
        # 1) translate
        with trace_block("translate", AgentType.TOOL, parent_event_id=root_eid, original_query=task.input) as trans_eid:
            model_name = getattr(llm, "model_name", type(llm).__name__)
            tool_eid = tracer.log_tool_start(
                agent_name="translate",
                tool_name=str(model_name),
                tool_input={"data_preview": (task.input or "")[:200]},
                parent_event_id=trans_eid
            )
            try:
                english_query = (translate_prompt | llm).invoke({"data": task.input}, config={"callbacks":[cb]}).strip()
                if english_query.startswith('"') and english_query.endswith('"'):
                    english_query = english_query[1:-1].strip()
                tracer.log_tool_end(
                    tool_eid,
                    tool_output={"translation_preview": english_query[:400]},
                    success=True
                )
            except Exception as e:
                tracer.log_tool_end(tool_eid, tool_output={"error": str(e)[:400]}, success=False)
                tracer.log_error("translate", e, parent_event_id=trans_eid)
                tracer.end_trace(root_eid, {"error": str(e)}, success=False)
                raise HTTPException(status_code=500, detail=str(e))

        # 2) web_search
        with trace_block("web_search", AgentType.SEARCH, parent_event_id=root_eid, query=english_query) as search_eid:
            tool_eid = tracer.log_tool_start(
                agent_name="web_search",
                tool_name="DuckDuckGoSearchRun",
                tool_input={"query": english_query},
                parent_event_id=search_eid
            )
            try:
                result = duck.run(english_query)
                tracer.log_tool_end(
                    tool_eid,
                    tool_output={"result_preview": str(result)[:400]},
                    success=True
                )
            except Exception as e:
                tracer.log_tool_end(tool_eid, tool_output={"error": str(e)[:400]}, success=False)
                tracer.log_error("web_search", e, parent_event_id=search_eid)
                tracer.end_trace(root_eid, {"error": str(e)}, success=False)
                raise HTTPException(status_code=500, detail=str(e))

        tracer.end_trace(root_eid, {"success": True}, success=True)
        return {"result": result, "agent": "SEARCH_AGENT", "translated_query": english_query}

    except HTTPException:
        raise
    except Exception as e:
        tracer.log_error("search/execute", e, parent_event_id=root_eid)
        tracer.end_trace(root_eid, {"error": str(e)}, success=False)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tracer.export_traces("search_agent.json", format="json")


# 3. АГЕНТ-АНАЛИТИК (отдельный сервис)
analysis_app = FastAPI()
analysis_app.add_middleware(AccessLogMiddleware, service_name="analysis-agent", log_fn=log_json)

@analysis_app.post("/execute")
async def execute_analysis(task: TaskRequest, request: Request):
    parent = request.headers.get("X-Parent-Event")
    root_eid = tracer.start_trace(
        agent_name="analysis-agent/execute",
        agent_type=AgentType.ANALYSIS,
        data={
            "trace_headers_present": bool(parent),
            "input_preview": (task.input or "")[:200]
        },
        parent_event_id=parent
    )

    cb = LoggingCallback(log_fn=log_json)
    try:
        # prepare_prompt
        with trace_block(
            "prepare_prompt",
            AgentType.CUSTOM,
            parent_event_id=root_eid,
            template="Analyze this information and extract key insights in Russian"
        ) as prep_eid:
            prompt = PromptTemplate.from_template("""
            Analyze this information and extract key insights in Russian:

            {data}

            Provide comprehensive analysis in Russian.
            """)
            chain = prompt | llm

        # llm_inference
        with trace_block(
            "llm_inference",
            AgentType.ANALYSIS,
            parent_event_id=root_eid,  # <-- важное: родитель — текущий блок, а не prep_eid
            tokens_estimate=len((task.input or ""))  # просто пример метадаты
        ) as infer_eid:
            model_name = getattr(llm, "model_name", type(llm).__name__)
            tool_eid = tracer.log_tool_start(
                agent_name="llm_inference",              # <-- это имя пойдёт в граф (будет "llm_inference::MODEL")
                tool_name=str(model_name),
                tool_input={"data_preview": (task.input or "")[:200]},
                parent_event_id=infer_eid                # <-- правильный parent
            )
            try:
                result = chain.invoke({"data": task.input}, config={"callbacks":[cb]})
                tracer.log_tool_end(
                    tool_eid,
                    tool_output={"result_preview": (str(result) if result is not None else "")[:400]},
                    success=True
                )
            except Exception as e:
                tracer.log_tool_end(
                    tool_eid,
                    tool_output={"error": str(e)[:400]},
                    success=False
                )
                tracer.log_error("llm_inference", e, parent_event_id=infer_eid)
                tracer.end_trace(root_eid, {"error": str(e)}, success=False)
                raise HTTPException(status_code=500, detail=str(e))

        tracer.end_trace(root_eid, {"success": True}, success=True)
        return {"result": result, "agent": "ANALYSIS_AGENT"}

    except HTTPException:
        # уже залогировано выше
        raise
    except Exception as e:
        tracer.log_error("analysis-agent/execute", e, parent_event_id=root_eid)
        tracer.end_trace(root_eid, {"error": str(e)}, success=False)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tracer.export_traces("analysis_agent.json", format="json")

# 4. АГЕНТ-ОТЧЕТ (отдельный сервис)
report_app = FastAPI()
report_app.add_middleware(AccessLogMiddleware, service_name="report-agent", log_fn=log_json)

@report_app.post("/execute")
async def execute_report(task: TaskRequest, request: Request):
    parent = request.headers.get("X-Parent-Event")
    root_eid = tracer.start_trace(
        agent_name="report-agent/execute",
        agent_type=AgentType.REPORT,
        data={
            "trace_headers_present": bool(parent),
            "input_preview": (task.input or "")[:300]
        },
        parent_event_id=parent
    )
    try:
        cb = LoggingCallback(log_fn=log_json)

        # подготовка промпта
        with trace_block("prepare_prompt", AgentType.CUSTOM, parent_event_id=root_eid) as prep_eid:
            prompt = PromptTemplate.from_template(
                """Generate comprehensive final answer in Russian based on this information:

                {data}

                Answer in clear, structured Russian."""
            )
            chain = prompt | llm

        # инференс LLM (как инструмент: start -> invoke -> end)
        with trace_block("llm_inference", AgentType.REPORT, parent_event_id=prep_eid) as infer_eid:
            model_name = getattr(llm, "model_name", type(llm).__name__)
            tool_eid = tracer.log_tool_start(
                agent_name="llm_inference",
                tool_name=str(model_name),
                tool_input={"data_preview": (task.input or "")[:200]},
                parent_event_id=infer_eid
            )
            try:
                result = chain.invoke({"data": task.input}, config={"callbacks": [cb]})
                tracer.log_tool_end(
                    tool_eid,
                    tool_output={"result_preview": (str(result) if result is not None else "")[:400]},
                    success=True
                )
            except Exception as e:
                tracer.log_tool_end(
                    tool_eid,
                    tool_output={"error": str(e)[:400]},
                    success=False
                )
                raise

        tracer.end_trace(root_eid, {"success": True}, success=True)
        return {"result": result, "agent": "REPORT_AGENT"}

    except Exception as e:
        tracer.log_error("report-agent/execute", e, parent_event_id=root_eid)
        tracer.end_trace(root_eid, {"error": str(e)}, success=False)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tracer.export_traces("report_agent.json", format="json")

# 5. ГЛАВНЫЙ ОРКЕСТРАТОР (управляет всей системой)
orchestrator_app = FastAPI()
orchestrator_app.add_middleware(AccessLogMiddleware, service_name="orchestrator", log_fn=log_json)


router = RouterAgent(llm)

# Маппинг агентов на их URL
AGENT_URLS = {
    "SEARCH_AGENT": "http://localhost:8001/execute",
    "ANALYSIS_AGENT": "http://localhost:8002/execute", 
    "REPORT_AGENT": "http://localhost:8003/execute"
}

for _app in [orchestrator_app, search_app, analysis_app, report_app]:
    try:
        _app.add_middleware(TraceHeadersMiddleware)
    except Exception:
        pass


@orchestrator_app.post("/orchestrate")
async def orchestrate(task: TaskRequest):
    request_id = str(uuid.uuid4())
    execution_chain = []
    current_data = task.input
    context = f"[{request_id}] Original query: {task.input}"
    last_agent = None
    search_runs = 0

    # корневое событие запроса оркестратора
    root_eid = tracer.start_trace(
        agent_name="orchestrator/orchestrate",
        agent_type=AgentType.ORCHESTRATOR,
        data={"request_id": request_id, "input_preview": (task.input or "")[:300]}
    )

    try:
        for step in range(6):
            # ── Роутинг шага
            with trace_block(
                agent_name="router",
                agent_type=AgentType.CLARIFICATION,
                parent_event_id=root_eid,
                step=step + 1
            ) as route_eid:

                routing = router.route(task.input if step == 0 else current_data, context)
                next_agent = routing["next_agent"]
                agent_input = routing["input"]

                # Жёсткие перила
                if last_agent == "SEARCH_AGENT" and next_agent == "SEARCH_AGENT":
                    next_agent = "ANALYSIS_AGENT"
                    agent_input = f"Analyze following search snippets:\n{current_data[:5000]}"
                if last_agent == "ANALYSIS_AGENT" and next_agent == "ANALYSIS_AGENT":
                    next_agent = "REPORT_AGENT"
                    agent_input = f"Generate report from analysis:\n{current_data[:5000]}"

                if next_agent == "SEARCH_AGENT":
                    search_runs += 1
                    if search_runs > 2:
                        next_agent = "ANALYSIS_AGENT"
                        agent_input = f"Analyze following search snippets:\n{current_data[:5000]}"

                agent_url = AGENT_URLS.get(next_agent)
                if not agent_url:
                    raise RuntimeError(f"No URL for agent {next_agent}")

                # лог «сообщения» (для диаграммы последовательности)
                tracer.log_message(
                    from_agent="orchestrator",
                    to_agent=next_agent,
                    message=f"step={step+1}, reason={routing.get('reason','')}, input_preview={agent_input[:180]}",
                    parent_event_id=route_eid
                )
                await log_trace(
                    event="route",
                    request_id=request_id,
                    step=step + 1,
                    next_agent=next_agent,
                    reason=routing.get("reason", ""),
                    agent_input_preview=agent_input[:300]
                )

            with trace_block(
                agent_name=f"http_call::{next_agent}",
                agent_type=AgentType.TOOL,
                parent_event_id=route_eid,
                url=agent_url
            ) as http_eid:

                HTTP_HEADERS = {
                    "X-Trace-Session": tracer.session_id,
                    "X-Parent-Event": http_eid,
                }

                tool_eid = tracer.log_tool_start(
                    agent_name=f"http_call::{next_agent}",   
                    tool_name="HTTP POST",
                    tool_input={"url": agent_url, "json": {"input": agent_input[:500]}},
                    parent_event_id=http_eid
                )

                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(agent_url, json={"input": agent_input}, headers=HTTP_HEADERS) as resp:
                            body_text = await resp.text()
                            if resp.status != 200:
                                tracer.log_tool_end(
                                    tool_eid,
                                    tool_output={"status": resp.status, "body": body_text[:400]},
                                    success=False
                                )
                                tracer.log_error(f"http_call::{next_agent}",
                                                RuntimeError(f"HTTP {resp.status}: {body_text[:400]}"),
                                                parent_event_id=http_eid)
                                raise RuntimeError(f"{next_agent} failed: {body_text}")

                            payload = await resp.json()
                            current_data = payload["result"]
                            tracer.log_tool_end(
                                tool_eid,
                                tool_output={"status": resp.status, "result_preview": str(current_data)[:400]},
                                success=True
                            )

                except Exception as e:
                    # чтобы end-тул под любой ошибкой точно был написан (если не успели выше)
                    try:
                        tracer.log_tool_end(tool_eid, tool_output={"error": str(e)[:400]}, success=False)
                    except Exception:
                        pass
                    tracer.log_error(f"http_call::{next_agent}", e, parent_event_id=http_eid)
                    raise

            # обновляем контекст и цепочку
            context += f" | {next_agent} output: {current_data[:400]}"
            execution_chain.append({
                "step": step + 1,
                "agent": next_agent,
                "input": (agent_input or "")[:250],
                "output": (current_data or "")[:250]
            })
            last_agent = next_agent

            if next_agent == "REPORT_AGENT":
                break

        # успех — закрываем корень
        tracer.end_trace(root_eid, {"success": True}, success=True)
        return {
            "final_result": current_data,
            "execution_chain": execution_chain,
            "trace_id": request_id,
            "status": "completed" if last_agent == "REPORT_AGENT" else "max_steps_reached"
        }

    except Exception as e:
        # ошибка — логируем и закрываем корень с флагом fail
        logging.exception(f"[{request_id}] Orchestration error: {e}")
        tracer.log_error("orchestrator/orchestrate", e, parent_event_id=root_eid)
        tracer.end_trace(root_eid, {"error": str(e)}, success=False)
        return {
            "final_result": f"Error: {e}",
            "execution_chain": execution_chain,
            "trace_id": request_id,
            "status": "error"
        }
    finally:
        tracer.export_traces("orchestrator_trace.json", format="json")


@orchestrator_app.post("/test_router")
async def test_router(task: TaskRequest):
    """Тестовый эндпоинт для проверки работы роутера"""
    root_eid = tracer.start_trace(
        agent_name="orchestrator/test_router",
        agent_type=AgentType.ORCHESTRATOR,
        data={"input_preview": (task.input or "")[:300]}
    )
    try:
        # Отдельный блок для принятия решения роутером
        with trace_block(
            agent_name="router",
            agent_type=AgentType.CLARIFICATION,
            parent_event_id=root_eid
        ) as route_eid:
            routing = router.route(task.input)

            # Лог «сообщения» для диаграммы последовательности
            tracer.log_message(
                from_agent="orchestrator",
                to_agent=routing.get("next_agent", "UNKNOWN"),
                message=f"router_decision: {routing}",
                parent_event_id=route_eid
            )

        response = {
            "input": task.input,
            "routing_decision": routing,
            "agent_url": AGENT_URLS.get(routing["next_agent"])
        }

        # Успешное закрытие корневого эвента
        tracer.end_trace(root_eid, {"success": True}, success=True)
        return response

    except Exception as e:
        # Ошибочное закрытие + запись ошибки
        tracer.log_error("orchestrator/test_router", e, parent_event_id=root_eid)
        tracer.end_trace(root_eid, {"error": str(e)}, success=False)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tracer.export_traces("orchestrator_test_router_trace.json", format="json")


@orchestrator_app.get("/health")
async def health():
    return {"status": "healthy", "service": "orchestrator"}

@search_app.get("/health")
async def search_health():
    return {"status": "healthy", "service": "search_agent"}

@analysis_app.get("/health")
async def analysis_health():
    return {"status": "healthy", "service": "analysis_agent"}

@report_app.get("/health")
async def report_health():
    return {"status": "healthy", "service": "report_agent"}

# def run_server(app, port, log_file):
#     """Запускает FastAPI сервер на указанном порту"""
#     logging.basicConfig(
#         filename=log_file, 
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )
    
#     try:
#         logging.info(f"Starting server on port {port}")
#         uvicorn.run(
#             app, 
#             host="0.0.0.0", 
#             port=port, 
#             log_level="info",
#             access_log=False  # Убираем лишние логи
#         )
#     except Exception as e:
#         logging.error(f"Server failed to start on port {port}: {str(e)}")
#         raise

# # Освобождаем порты перед запуском
# def free_port(port):
#     """Освобождает порт если он занят"""
#     try:
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#             if s.connect_ex(('localhost', port)) == 0:
#                 # Порт занят, пытаемся освободить
#                 for proc in psutil.process_iter(['pid', 'name']):
#                     try:
#                         for conn in proc.net_connections():
#                             if conn.laddr.port == port:
#                                 proc.terminate()
#                                 proc.wait(timeout=2)
#                                 logging.info(f"Freed port {port} from process {proc.pid}")
#                     except (psutil.NoSuchProcess, psutil.AccessDenied):
#                         pass
#     except Exception as e:
#         logging.warning(f"Could not free port {port}: {str(e)}")

# # Освобождаем все нужные порты
# for port in [8000, 8001, 8002, 8003]:
#     free_port(port)
#     time.sleep(1)

# # Запускаем все сервисы
# services = [
#     (orchestrator_app, 8000, "orchestrator"),
#     (search_app, 8001, "search_agent"), 
#     (analysis_app, 8002, "analysis_agent"),
#     (report_app, 8003, "report_agent")
# ]

# for app, port, name in services:
#     threading.Thread(
#         target=run_server, 
#         args=(app, port, f"{name}.log"),
#         daemon=True  # Демонизируем потоки
#     ).start()
#     time.sleep(2)  # Пауза между запуском сервисов

# logging.info("All agents started successfully!")
# print("Multi-agent system is running on ports 8000-8003")
