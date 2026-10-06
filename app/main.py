from fastapi import FastAPI, Query
from pydantic import BaseModel
import os
import asyncio
from dotenv import load_dotenv
from evaluation.custom_evaluation import custom_evaluator
from app.support_agent_workflow import call_workflow_start, call_workflow_resume
from logger.audit_logger import _audit
from datetime import datetime, timezone
from uuid import uuid4

load_dotenv()


class AskRequest(BaseModel):
    query: str
    session_id: str | None = str(uuid4())   # pass back on subsequent turns for memory


class ReviewRequest(BaseModel):
    thread_id: str
    decision: str   # "approve" | "takeover" | free-text feedback


app = FastAPI(
    title="ResolveAI API",
    description="AI-powered customer support API",
    version="0.2.0",
)


def _safe_eval(query: str, response: str, context) -> None:
    """Run custom_evaluator, swallowing all errors (non-critical)."""
    try:
        custom_evaluator(query, response, context)
    except Exception as e:
        print(f"[custom_evaluator] skipped (non-critical): {e}")


def _build_response(result: dict, thread_id: str, session_id: str = "") -> dict:
    """Normalise the LangGraph result into the API response shape."""
    interrupt_val = None
    raw = result.get("__interrupt__")
    if raw:
        try:
            interrupt_val = raw[0].value
        except Exception:
            interrupt_val = None

    return {
        "thread_id":           thread_id,
        "session_id":          session_id,
        "query":               result.get("query"),
        "response":            result.get("draft"),
        "human_required":      result.get("human_required"),
        "hitl_reason":         result.get("hitl_reason"),
        "is_approved":         result.get("is_approved"),
        "human_feedback":      result.get("review_feedback"),
        "historical_evidence": result.get("historical_evidence"),
        "intent":              result.get("intent_name"),
        "intent_description":  result.get("intent_description"),
        "memory_turns":        len(_memory_turns(result)),
        "interrupt_data":      interrupt_val,
    }


def _memory_turns(result: dict) -> list:
    ctx = result.get("memory_context") or ""
    return [l for l in ctx.splitlines() if l.startswith("[Turn")]


@app.get("/")
def root():
    return {"status": "ok", "service": "ResolveAI"}


@app.post("/ask")
async def ask(request: AskRequest):
    """
    Start the support workflow.

    Returns immediately even when human review is required.
    ``human_required=True`` means the caller should display the draft and
    POST to ``/review`` with the human's decision.
    """
    # Run the blocking LangGraph workflow in a thread so the event loop
    # stays free (prevents uvicorn worker exhaustion on long LLM calls).
    result, thread_id, session_id = await asyncio.to_thread(
        call_workflow_start, request.query, request.session_id
    )

    payload = _build_response(result, thread_id, session_id)

    # Background eval — only when a final response exists
    response = payload["response"]
    context  = payload["historical_evidence"]
    if response and context and not payload["human_required"]:
        asyncio.create_task(
            asyncio.to_thread(_safe_eval, request.query, response, context)
        )

    return payload


@app.post("/review")
async def review(request: ReviewRequest):
    """
    Resume a paused workflow with the human reviewer's decision.

    ``decision``:
      - ``"approve"``   — accept the draft as-is
      - ``"takeover"``  — human handles the customer directly
      - <any other text> — treat as feedback and regenerate
    """
    result = await asyncio.to_thread(
        call_workflow_resume, request.thread_id, request.decision
    )

    payload = _build_response(result, request.thread_id)

    response = payload["response"]
    context  = payload["historical_evidence"]
    if response and context:
        asyncio.create_task(
            asyncio.to_thread(
                _safe_eval, result.get("query", ""), response, context
            )
        )

    return payload


# ── Audit Trail ───────────────────────────────────────────────────────────────

@app.get("/audit")
async def get_audit(
    session_id: str | None = Query(default=None),
    thread_id:  str | None = Query(default=None),
):
    """
    Return audit events.
    - No params    → all events
    - session_id   → events for that session
    - thread_id    → events for that thread
    """
    if session_id:
        events = _audit.get_by_session(session_id)
    elif thread_id:
        events = _audit.get_by_thread(thread_id)
    else:
        events = _audit.get_all()

    return {"total": len(events), "events": events}


@app.delete("/audit/clear")
async def clear_audit():
    """Admin: wipe the in-memory audit log."""
    _audit.clear()
    return {"status": "cleared"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }