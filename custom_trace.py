
import logging
import time
import json
import uuid
from typing import Callable, Optional
import logging, sys, json
from logging.handlers import RotatingFileHandler
import time
from starlette.middleware.base import BaseHTTPMiddleware
import uuid
from typing import Callable, Optional
from langchain.callbacks.base import BaseCallbackHandler
from starlette.responses import Response


logging.basicConfig(
    filename='system.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

MAX_LOG_CHARS = 2000

def _safe_text(b: bytes) -> str:
    try:
        return b.decode("utf-8", errors="replace")
    except Exception:
        return "<binary>"

def _truncate(s: str, n: int = MAX_LOG_CHARS) -> str:
    return s if len(s) <= n else s[:n] + "...<truncated>"

class AccessLogMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, service_name: str, log_fn: Callable[[dict], None]):
        super().__init__(app)
        self.service_name = service_name
        self.log_fn = log_fn

    async def dispatch(self, request, call_next):
        trace_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
        start_ts = time.time()

        # --- Request logging
        raw_req = await request.body()
        self.log_fn(
            service=self.service_name,
            event="request_in",
            trace_id=trace_id,
            method=request.method,
            path=str(request.url.path),
            body=_truncate(_safe_text(raw_req)),
        )

        # --- Call downstream
        resp = await call_next(request)

        # Drain body_iterator to log response body
        body = b""
        async for chunk in resp.body_iterator:
            body += chunk

        # Preserve headers & content type
        headers = dict(resp.headers)
        if "content-type" not in {k.lower(): v for k, v in headers.items()}:
            # Умная догадка: если это JSON-подобно — ставим JSON
            if body[:1] in (b"{", b"["):
                headers["content-type"] = "application/json; charset=utf-8"
            else:
                headers["content-type"] = "text/plain; charset=utf-8"

        # Rebuild response WITHOUT media_type arg, чтобы не перезаписать content-type
        new_resp = Response(
            content=body,
            status_code=resp.status_code,
            headers=headers,
        )

        # --- Response logging
        self.log_fn(
            service=self.service_name,
            event="response_out",
            trace_id=trace_id,
            status=resp.status_code,
            duration_ms=int((time.time() - start_ts) * 1000),
            body=_truncate(_safe_text(body)),
        )
        return new_resp
def setup_logging(log_path="orchestrator.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        sh = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt); sh.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(sh)

def log_json(**kwargs):
    logging.info(json.dumps(kwargs, ensure_ascii=False))

async def log_trace(event: str, **fields):
    fields.update({
        "ts": time.time(),
        "event": event
    })
    log_json(**fields)

class LoggingCallback(BaseCallbackHandler):
    def __init__(self, log_fn: Optional[Callable[..., None]] = None, log_file: str = "trace_log.log"):
        self.log_fn = log_fn
        self.log_file = log_file

    def _emit(self, **fields):
        # в файл (чтобы всегда что-то писалось)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] " + " ".join(f"{k}={repr(v)[:500]}" for k, v in fields.items())
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        # в внешний логгер, если передан
        if self.log_fn:
            try:
                self.log_fn(**fields)  # ИМЕНОВАННЫЕ аргументы!
            except Exception as e:
                logging.warning(f"LoggingCallback log_fn failed: {e}")

    # LLM события
    def on_llm_start(self, serialized, prompts, **kwargs):
        self._emit(event="llm_start", prompts=prompts)

    def on_llm_end(self, response, **kwargs):
        try:
            text = response.generations[0][0].text
        except Exception:
            text = str(response)
        self._emit(event="llm_end", text=text[:2000])

    # Цепочки
    def on_chain_start(self, serialized, inputs, **kwargs):
        self._emit(event="chain_start", inputs=inputs)

    def on_chain_end(self, outputs, **kwargs):
        self._emit(event="chain_end", outputs=str(outputs)[:2000])

    # Инструменты
    def on_tool_start(self, serialized, input_str, **kwargs):
        self._emit(event="tool_start", tool=serialized.get("name"), input=str(input_str)[:500])

    def on_tool_end(self, output, **kwargs):
        self._emit(event="tool_end", output=str(output)[:500])