"""FastAPI backend.

Endpoints:
  GET  /api/health       -> health probe
  POST /api/chat         -> JSON response (full result at once)
  POST /api/chat/stream  -> SSE stream: trace events, cost estimate, answer tokens
  GET  /                 -> React chat UI (built by `npm run build` in frontend/,
                            output lands in static/ and is mounted at the bottom
                            of this file so it doesn't shadow the /api routes)

Sessions are kept in-memory so a clarification round-trip reuses the already
extracted content instead of re-processing the uploads.
"""
from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.agent.cost import estimate_cost
from app.agent.graph import AGENT, execute_node, plan_node, stream_synthesize
from app.agent.state import AgentState
from app.config import get_settings
from app.ingestion.pipeline import ingest_files
from app.schemas import ChatResponse, ExtractedItem, TraceEvent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic Multimodal Assistant", version="1.0.0")

_settings = get_settings()
if not _settings.groq_api_key:
    logger.warning(
        "No GROQ_API_KEY found. Get a free key at console.groq.com and set it "
        "in a .env file next to this app, or as an environment variable / "
        "docker --env-file flag. Requests will fail until a key is provided."
    )

# session_id -> {"history": [...], "extracted": [ExtractedItem], "ts": float}
SESSIONS: dict[str, dict] = {}


def _session(session_id: str | None) -> tuple[str, dict]:
    ttl = get_settings().session_ttl_seconds
    now = time.time()
    for sid in [s for s, v in SESSIONS.items() if now - v["ts"] > ttl]:
        SESSIONS.pop(sid, None)

    if session_id and session_id in SESSIONS:
        sess = SESSIONS[session_id]
        sess["ts"] = now
        return session_id, sess
    sid = session_id or uuid.uuid4().hex[:12]
    SESSIONS[sid] = {"history": [], "extracted": [], "ts": now}
    return sid, SESSIONS[sid]


async def _read_uploads(files: list[UploadFile]) -> list[tuple[str, bytes, str]]:
    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    out = []
    for f in files or []:
        raw = await f.read()
        if len(raw) > max_bytes:
            raise HTTPException(413, f"{f.filename} exceeds {get_settings().max_upload_mb} MB limit")
        if raw:
            out.append((f.filename or "upload", raw, f.content_type or ""))
    return out


def _prepare_state(sid: str, sess: dict, query: str,
                   uploads: list[tuple[str, bytes, str]]) -> tuple[AgentState, list[TraceEvent]]:
    new_items, ingest_trace = ingest_files(uploads) if uploads else ([], [])
    sess["extracted"].extend(new_items)
    sess["history"].append({"role": "user", "content": query or "(files uploaded)"})
    state: AgentState = {
        "query": query,
        "history": list(sess["history"]),
        "extracted": list(sess["extracted"]),
        "trace": list(ingest_trace),
    }
    return state, ingest_trace


@app.get("/api/health")
def health():
    st = get_settings()
    return {"status": "ok" if st.groq_api_key else "missing_api_key",
            "model": st.llm_model, "key_present": bool(st.groq_api_key)}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    query: str = Form(""),
    session_id: str | None = Form(None),
    files: list[UploadFile] = File(default=[]),
):
    sid, sess = _session(session_id)
    uploads = await _read_uploads(files)
    if not query.strip() and not uploads and not sess["extracted"]:
        raise HTTPException(400, "Provide a query, at least one file, or both.")

    state, _ = _prepare_state(sid, sess, query.strip(), uploads)
    try:
        result: AgentState = AGENT.invoke(state)
    except Exception as exc:
        logger.exception("Agent run failed")
        return ChatResponse(session_id=sid, kind="error",
                            answer=f"The agent hit an unexpected error: {exc}",
                            extracted=state["extracted"], trace=state["trace"])

    sess["history"].append({"role": "assistant", "content": result.get("answer", "")})
    cost = estimate_cost(query, state["extracted"], result.get("plan"))
    return ChatResponse(
        session_id=sid,
        kind=result.get("kind", "answer"),
        answer=result.get("answer", ""),
        extracted=state["extracted"],
        trace=result.get("trace", []),
        cost_estimate=cost,
    )


@app.post("/api/chat/stream")
async def chat_stream(
    query: str = Form(""),
    session_id: str | None = Form(None),
    files: list[UploadFile] = File(default=[]),
):
    """SSE stream. Event payloads (JSON):
    {"type":"session","session_id":...}
    {"type":"trace","event":{...}}          # real-time tool/plan visualization
    {"type":"extracted","items":[...]}
    {"type":"cost","estimate":{...}}        # pre-execution cost estimate
    {"type":"token","text":"..."}           # final answer, token by token
    {"type":"done","kind":"answer|clarify|error","answer":"..."}
    """
    sid, sess = _session(session_id)
    uploads = await _read_uploads(files)
    if not query.strip() and not uploads and not sess["extracted"]:
        raise HTTPException(400, "Provide a query, at least one file, or both.")

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    def run():
        yield sse({"type": "session", "session_id": sid})
        state, ingest_trace = _prepare_state(sid, sess, query.strip(), uploads)
        for ev in ingest_trace:
            yield sse({"type": "trace", "event": ev.model_dump()})
        yield sse({"type": "extracted",
                   "items": [i.model_dump() for i in state["extracted"]]})
        try:
            # PLAN
            before = len(state["trace"])
            state.update(plan_node(state))
            for ev in state["trace"][before:]:
                yield sse({"type": "trace", "event": ev.model_dump()})

            cost = estimate_cost(query, state["extracted"], state.get("plan"))
            yield sse({"type": "cost", "estimate": cost.model_dump()})

            plan = state["plan"]
            if plan.action == "clarify":
                q = plan.clarify_question or "Could you clarify what you'd like me to do?"
                sess["history"].append({"role": "assistant", "content": q})
                yield sse({"type": "done", "kind": "clarify", "answer": q})
                return

            # EXECUTE (emit each tool trace as it happens)
            before = len(state["trace"])
            state.update(execute_node(state))
            for ev in state["trace"][before:]:
                yield sse({"type": "trace", "event": ev.model_dump()})

            # SYNTHESIZE (token streaming)
            answer_parts: list[str] = []
            for token in stream_synthesize(state):
                answer_parts.append(token)
                yield sse({"type": "token", "text": token})
            answer = "".join(answer_parts).strip()
            sess["history"].append({"role": "assistant", "content": answer})
            yield sse({"type": "trace", "event": TraceEvent(
                stage="synthesize", title="Composed final answer",
                detail=f"{len(answer)} chars").model_dump()})
            yield sse({"type": "done", "kind": "answer", "answer": answer})
        except Exception as exc:
            logger.exception("Streaming agent run failed")
            yield sse({"type": "done", "kind": "error",
                       "answer": f"The agent hit an unexpected error: {exc}"})

    return StreamingResponse(run(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})



app.mount("/", StaticFiles(directory="static", html=True), name="static")
