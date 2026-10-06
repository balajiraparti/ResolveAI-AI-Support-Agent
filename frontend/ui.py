import requests
import streamlit as st
from pathlib import Path
import sys

# ── Project path ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Evaluator imports ─────────────────────────────────────────────────────────
try:
    from evaluation.custom_evaluation import custom_evaluator
    _custom_ok = True
except Exception as e:
    _custom_ok = False
    _custom_err = str(e)


# ── Constants ─────────────────────────────────────────────────────────────────
# API_URL    = "https://resolveai-ai-support-agent-production.up.railway.app/ask"
# REVIEW_URL = "https://resolveai-ai-support-agent-production.up.railway.app/review"
# AUDIT_URL  = "https://resolveai-ai-support-agent-production.up.railway.app/audit"
API_URL    = "http://localhost:8000/ask"
REVIEW_URL = "http://localhost:8000/review"
AUDIT_URL  = "http://localhost:8000/audit"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="ResolveAI", page_icon="🎧", layout="wide")

st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Agent response bubble */
    .agent-bubble {
        background: linear-gradient(135deg, #1a1d2e 0%, #1e2130 100%);
        border: 1px solid #2d3348;
        border-left: 4px solid #6366f1;
        border-radius: 0 14px 14px 14px;
        padding: 1.1rem 1.4rem;
        margin: 0.5rem 0 1rem 0;
        font-size: 0.95rem;
        line-height: 1.75;
        color: #e2e8f8;
    }
    .agent-bubble p  { margin: 0.5em 0; }
    .agent-bubble ul { padding-left: 1.4em; margin: 0.5em 0; }
    .agent-bubble ol { padding-left: 1.4em; margin: 0.5em 0; }
    .agent-bubble li { margin-bottom: 0.3em; }
    .agent-bubble strong { color: #a5b4fc; }
    .agent-bubble code { background:#2d3252; border-radius:4px; padding:0 4px; font-size:0.88em; }

    /* Metadata tag row */
    .tag-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.6rem 0 1rem; }
    .tag {
        display: inline-flex; align-items: center; gap: 0.3rem;
        background: #2d3348; border-radius: 20px;
        padding: 0.2rem 0.75rem; font-size: 0.75rem; color: #9aa5c4;
    }
    .tag-val { color: #c4cbe8; font-weight: 600; }

    /* Metric card */
    .metric-card {
        background: #1e2130; border-radius: 12px;
        padding: 1.2rem 1.4rem; margin-bottom: 0.8rem;
        border-left: 4px solid #4f8ef7;
    }
    .metric-label { color: #9aa5c4; font-size: 0.78rem; text-transform: uppercase;
                    letter-spacing: 0.06em; margin-bottom: 0.3rem; }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .score-high   { color: #4ade80; }
    .score-mid    { color: #fbbf24; }
    .score-low    { color: #f87171; }

    /* Section pill */
    .pill {
        display: inline-block; background: #2d3348; border-radius: 20px;
        padding: 0.25rem 0.9rem; font-size: 0.78rem; font-weight: 600;
        color: #9aa5c4; margin-bottom: 0.5rem; letter-spacing: 0.05em;
    }

    /* Context chunk card */
    .chunk-card {
        background: #1e2130; border-radius: 8px; padding: 0.7rem 1rem;
        margin-bottom: 0.4rem; border-left: 3px solid #6366f1;
        font-size: 0.82rem; color: #c4cbe8; line-height: 1.6;
    }
    .chunk-index { color: #6366f1; font-weight: 700; margin-right: 0.4rem; }

    /* HITL warning banner */
    .hitl-banner {
        background: #2d1f0e; border: 1px solid #f59e0b; border-radius: 10px;
        padding: 1rem 1.2rem; margin: 0.8rem 0; color: #fcd34d;
    }
    .hitl-title  { font-weight: 700; font-size: 1rem; margin-bottom: 0.4rem; }
    .hitl-reason { font-size: 0.88rem; color: #fbbf24; }

    /* User query bubble */
    .user-bubble {
        background: #1a2a1e; border: 1px solid #2d4838;
        border-left: 4px solid #4ade80;
        border-radius: 14px 0 14px 14px;
        padding: 0.75rem 1.2rem; margin-bottom: 0.6rem;
        font-size: 0.92rem; color: #d1fae5;
    }

    /* ── Audit Trail badges ───────────────────────────────── */
    .aud-badge {
        display: inline-block; border-radius: 20px;
        padding: 0.18rem 0.7rem; font-size: 0.72rem; font-weight: 700;
        letter-spacing: 0.04em; white-space: nowrap;
    }
    .aud-SUCCESS  { background: #14532d; color: #86efac; border: 1px solid #16a34a; }
    .aud-ESCALATED{ background: #78350f; color: #fde68a; border: 1px solid #d97706; }
    .aud-WARNING  { background: #7f1d1d; color: #fca5a5; border: 1px solid #dc2626; }
    .aud-PENDING  { background: #1e293b; color: #94a3b8; border: 1px solid #475569; }
    .aud-FAILURE  { background: #450a0a; color: #ff6b6b; border: 1px solid #ef4444; }

    /* audit event row — 7 columns: ID | Timestamp | Session | Node | EventType | Status | Message */
    .aud-row {
        display: grid;
        grid-template-columns: 120px 145px 120px 170px 190px 115px 1fr;
        gap: 0.4rem; align-items: start;
        padding: 0.55rem 0.9rem; border-bottom: 1px solid #1e2130;
        font-size: 0.79rem; color: #c4cbe8;
    }
    .aud-header {
        background: #161929; color: #6b7ba4;
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.06em;
        border-radius: 8px 8px 0 0;
    }
    .aud-row:hover { background: #1a1d2e; }
    .aud-event-id { color: #6366f1; font-family: monospace; font-size: 0.76rem; }
    .aud-node     { color: #a5b4fc; font-weight: 600; }
    .aud-type     { color: #e2e8f8; }
    .aud-table    { border: 1px solid #2d3348; border-radius: 8px; overflow: hidden; margin-top: 0.5rem; }
    .aud-msg      { color: #94a3b8; font-size: 0.76rem; line-height: 1.6; word-break: break-word; }
    .aud-msg.failure { color: #ff6b6b; }
    /* key-value pills inside message column */
    .aud-kv       { display: inline-flex; flex-wrap: wrap; gap: 0.25rem 0.5rem; }
    .aud-kv-pair  { display: inline-flex; align-items: baseline; gap: 0.25rem;
                    background: #1e2130; border-radius: 6px;
                    padding: 0.1rem 0.5rem; font-size: 0.73rem; }
    .aud-kv-key   { color: #6366f1; font-weight: 600; }
    .aud-kv-sep   { color: #475569; }
    .aud-kv-val   { color: #e2e8f8; max-width: 30ch; overflow: hidden;
                    text-overflow: ellipsis; white-space: nowrap; }
    .aud-kv-val.bool-true  { color: #4ade80; }
    .aud-kv-val.bool-false { color: #f87171; }
    .aud-kv-val.null-val   { color: #64748b; font-style: italic; }
</style>
""", unsafe_allow_html=True)

st.title("🎧 ResolveAI")
tab_chat, tab_eval, tab_audit = st.tabs(["💬 Chat", "🔬 Evaluate", "🔍 Audit Trail"])


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _call_ask(query: str, session_id: str | None = None) -> dict | None:
    try:
        r = requests.post(
            API_URL,
            json={"query": query, "session_id": session_id},
            timeout=(10, 300),
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API call failed: {e}")
        return None


def _call_review(thread_id: str, decision: str) -> dict | None:
    try:
        r = requests.post(REVIEW_URL, json={"thread_id": thread_id, "decision": decision}, timeout=(10, 300))
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Review API call failed: {e}")
        return None


def _normalise_context(raw) -> list:
    if isinstance(raw, list):
        return [str(c) for c in raw if c]
    if isinstance(raw, str) and raw.strip():
        return [raw]
    return []


def _score_class(val):
    try:
        v = float(val)
        return "score-high" if v >= 0.7 else ("score-mid" if v >= 0.4 else "score-low")
    except (TypeError, ValueError):
        return "score-mid"


def _fmt(val):
    try:
        return f"{float(val):.3f}"
    except (TypeError, ValueError):
        return str(val) if val is not None else "N/A"


def _render_agent_response(response_text: str) -> None:
    """Render the agent's reply with proper markdown formatting in a styled bubble."""
    # Wrap in a styled container — st.markdown renders markdown natively
    st.markdown('<div class="agent-bubble">', unsafe_allow_html=True)
    st.markdown(response_text or "*(no response)*")
    st.markdown('</div>', unsafe_allow_html=True)


def _render_metadata_tags(data: dict) -> None:
    """Render intent / approved / HITL as a compact tag row."""
    tags = []
    if data.get("intent"):
        tags.append(("🎯 Intent", data["intent"]))
    hr = data.get("human_required")
    if hr is not None:
        tags.append(("👤 Human required", "Yes" if hr else "No"))
    appr = data.get("is_approved")
    if appr is not None:
        tags.append(("✅ Approved", "Yes" if appr else "No"))
    if data.get("hitl_reason") and not data.get("human_required"):
        tags.append(("ℹ️ HITL", data["hitl_reason"][:60]))

    if tags:
        html = '<div class="tag-row">' + "".join(
            f'<span class="tag">{k}: <span class="tag-val">{v}</span></span>'
            for k, v in tags
        ) + "</div>"
        st.markdown(html, unsafe_allow_html=True)


def _render_context_expander(context_chunks: list) -> None:
    """Show retrieved context in a collapsible expander with chunk cards."""
    n = len(context_chunks)
    label = f"📚 Retrieved context  ({n} chunk{'s' if n != 1 else ''})"
    with st.expander(label, expanded=False):
        if context_chunks:
            for i, chunk in enumerate(context_chunks, 1):
                st.markdown(
                    f'<div class="chunk-card"><span class="chunk-index">#{i}</span>{chunk}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No context returned by the API.")


def _render_response_block(data: dict):
    """Render response + context + HITL banner from an API response dict."""
    response_text  = data.get("response") or data.get("draft") or ""
    context_chunks = _normalise_context(data.get("historical_evidence", []))

    st.markdown('<div class="pill">🤖 Agent Response</div>', unsafe_allow_html=True)
    _render_agent_response(response_text)
    _render_metadata_tags(data)
    _render_context_expander(context_chunks)

    if data.get("human_required"):
        st.markdown(
            f'<div class="hitl-banner">'
            f'<div class="hitl-title">⚠️ Human Review Required</div>'
            f'<div class="hitl-reason">{data.get("hitl_reason", "")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    return response_text, context_chunks


def _run_evaluator(eval_query, reference, response_text, context_chunks):
    """Run custom evaluator and display metric cards."""
    with st.spinner("Running Custom Evaluator…"):
        try:
            if not _custom_ok:
                st.error(f"Custom evaluator failed to import: {_custom_err}")
                return
            cp, ar, fth = custom_evaluator(eval_query, response_text, context_chunks)
            scores = {"Context Precision": cp, "Answer Relevance": ar, "Faithfulness": fth}
        except Exception as exc:
            st.error(f"Evaluator error: {exc}")
            return

    st.markdown('<div class="pill">⑥ Evaluation Scores</div>', unsafe_allow_html=True)
    st.caption("Evaluated with: **Custom Evaluator (GPT-4o-mini)**")
    cols = st.columns(len(scores))
    for col, (label, value) in zip(cols, scores.items()):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value {_score_class(value)}">{_fmt(value)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CHAT TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.header("Chat with ResolveAI")

    # ── Session state ──────────────────────────────────────────────────────────
    for _k, _v in [
        ("chat_data", None),
        ("chat_thread_id", None),
        ("chat_session_id", None),   # persists across turns for memory
        ("chat_awaiting_review", False),
    ]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    chat_query = st.text_input(
        "Customer message", key="chat_query",
        placeholder="Describe your Spotify issue…",
    )

    if st.button("Ask ➜", key="ask_button") and chat_query.strip():
        with st.spinner("Thinking…"):
            data = _call_ask(chat_query, st.session_state["chat_session_id"])
        if data:
            st.session_state["chat_data"] = data
            st.session_state["chat_thread_id"]  = data.get("thread_id")
            st.session_state["chat_session_id"] = data.get("session_id")  # keep for next turn
            st.session_state["chat_awaiting_review"] = bool(data.get("human_required"))

    data = st.session_state.get("chat_data")
    if data:
        resp_text      = data.get("response") or data.get("draft") or ""
        context_chunks = _normalise_context(data.get("historical_evidence", []))

        # ── User query echo ────────────────────────────────────────────────────
        st.markdown(
            f'<div class="user-bubble">🧑 {chat_query or st.session_state.get("_last_chat_query", "Your query")}</div>',
            unsafe_allow_html=True,
        )

        # ── Agent response (markdown-rendered) ─────────────────────────────────
        st.markdown('<div class="pill">🤖 Agent Response</div>', unsafe_allow_html=True)
        _render_agent_response(resp_text)
        _render_metadata_tags(data)
        _render_context_expander(context_chunks)
        # ── Human review panel ─────────────────────────────────────────────────
        if st.session_state["chat_awaiting_review"]:
            st.markdown(
                f'<div class="hitl-banner">'
                f'<div class="hitl-title">⚠️ This response requires human review</div>'
                f'<div class="hitl-reason">{data.get("hitl_reason", "")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            interrupt_data = data.get("interrupt_data") or {}
            if interrupt_data.get("draft"):
                with st.expander("📝 Draft response to review", expanded=True):
                    st.markdown(
                        f'<div class="text-box">{interrupt_data["draft"]}</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("**Your decision:**")
            rc1, rc2, rc3 = st.columns(3)

            with rc1:
                if st.button("✅ Approve", key="chat_approve"):
                    with st.spinner("Submitting approval…"):
                        result = _call_review(st.session_state["chat_thread_id"], "approve")
                    if result:
                        st.session_state["chat_data"] = result
                        st.session_state["chat_awaiting_review"] = bool(result.get("human_required"))
                        st.rerun()

            with rc2:
                if st.button("🙋 Takeover", key="chat_takeover"):
                    with st.spinner("Handing over…"):
                        result = _call_review(st.session_state["chat_thread_id"], "takeover")
                    if result:
                        st.session_state["chat_data"] = result
                        st.session_state["chat_awaiting_review"] = False
                        st.rerun()

            with rc3:
                feedback_text = st.text_input("Or type feedback to regenerate:", key="chat_feedback_text")
                if st.button("🔄 Send Feedback", key="chat_send_feedback") and feedback_text.strip():
                    with st.spinner("Regenerating…"):
                        result = _call_review(st.session_state["chat_thread_id"], feedback_text.strip())
                    if result:
                        st.session_state["chat_data"] = result
                        st.session_state["chat_awaiting_review"] = bool(result.get("human_required"))
                        st.rerun()

        with st.expander("Raw JSON", expanded=False):
            st.json(data)


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATE TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.header("Evaluation Harness")
    st.caption(
        "Enter a query and ground-truth answer. "
        "The app calls the ResolveAI API, then scores the result with your chosen evaluator."
    )

    # ── Session state ──────────────────────────────────────────────────────────
    for _k, _v in [
        ("eval_data", None),
        ("eval_thread_id", None),
        ("eval_awaiting_review", False),
        ("eval_ran", False),
        ("eval_results_ready", False),
    ]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # ── Inputs ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="pill">① Inputs</div>', unsafe_allow_html=True)
    col_q, col_gt = st.columns([1, 1])
    with col_q:
        eval_query = st.text_area(
            "User query", key="eval_query_input",
            placeholder="e.g. Why is my Spotify not working on my phone?", height=100,
        )
    with col_gt:
        reference = st.text_area(
            "Ground truth / Reference answer", key="eval_reference_input",
            placeholder="The expected correct answer…", height=100,
        )

    # ── Run button ─────────────────────────────────────────────────────────────
    if st.button(
        "▶  Generate & Evaluate", key="run_eval", type="primary",
        disabled=not (eval_query.strip() and reference.strip()),
    ):
        with st.spinner("Calling ResolveAI API…"):
            data = _call_ask(eval_query)

        if data:
            st.session_state["eval_data"] = data
            st.session_state["eval_thread_id"] = data.get("thread_id")
            st.session_state["eval_awaiting_review"] = bool(data.get("human_required"))
            st.session_state["eval_ran"] = True
            st.session_state["eval_results_ready"] = not bool(data.get("human_required"))

    # ── Results area ───────────────────────────────────────────────────────────
    if st.session_state["eval_ran"]:
        data = st.session_state["eval_data"]
        if not data:
            st.warning("No data — re-run the query.")
        else:
            st.divider()
            st.markdown('<div class="pill">③ API Response</div>', unsafe_allow_html=True)
            response_text, context_chunks = _render_response_block(data)

            st.markdown('<div class="pill">④ Ground Truth</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="text-box">{reference or "(none provided)"}</div>', unsafe_allow_html=True)

            # ── Human review inside eval tab ───────────────────────────────────
            if st.session_state["eval_awaiting_review"]:
                interrupt_data = data.get("interrupt_data") or {}

                if interrupt_data.get("draft"):
                    with st.expander("📝 Draft to review", expanded=True):
                        st.markdown(f'<div class="text-box">{interrupt_data["draft"]}</div>', unsafe_allow_html=True)

                st.markdown("**⑤ Human Review Decision:**")
                ev1, ev2, ev3 = st.columns(3)

                with ev1:
                    if st.button("✅ Approve", key="eval_approve"):
                        with st.spinner("Submitting…"):
                            result = _call_review(st.session_state["eval_thread_id"], "approve")
                        if result:
                            st.session_state["eval_data"] = result
                            st.session_state["eval_awaiting_review"] = bool(result.get("human_required"))
                            st.session_state["eval_results_ready"] = not bool(result.get("human_required"))
                            st.rerun()

                with ev2:
                    if st.button("🙋 Takeover", key="eval_takeover"):
                        with st.spinner("Handing over…"):
                            result = _call_review(st.session_state["eval_thread_id"], "takeover")
                        if result:
                            st.session_state["eval_data"] = result
                            st.session_state["eval_awaiting_review"] = False
                            st.rerun()

                with ev3:
                    eval_feedback_text = st.text_input("Feedback to regenerate:", key="eval_feedback_text")
                    if st.button("🔄 Send Feedback", key="eval_send_feedback") and eval_feedback_text.strip():
                        with st.spinner("Regenerating…"):
                            result = _call_review(st.session_state["eval_thread_id"], eval_feedback_text.strip())
                        if result:
                            st.session_state["eval_data"] = result
                            st.session_state["eval_awaiting_review"] = bool(result.get("human_required"))
                            st.session_state["eval_results_ready"] = not bool(result.get("human_required"))
                            st.rerun()

            # ── Run evaluator once human review is resolved ────────────────────
            if st.session_state["eval_results_ready"]:
                final_data = st.session_state["eval_data"]
                final_response = final_data.get("response") or final_data.get("draft") or ""
                final_context  = _normalise_context(final_data.get("historical_evidence", []))
                _run_evaluator(eval_query, reference, final_response, final_context)

    st.divider()
    st.caption("ResolveAI Evaluation Harness · Query + Ground Truth → Generate → (Review if needed) → Custom Evaluate")


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT TRAIL TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_audit:
    import time, json, io

    st.header("🔍 Audit Trail Stream")
    st.caption("Append-only log of every decision made by the ResolveAI agent — reconstructable chronologically.")

    # ── Controls ───────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([1.8, 1.8, 1.2, 1.2, 1.2])
    with ctrl1:
        filter_session = st.text_input("Filter by Session ID", key="aud_session", placeholder="session-uuid…")
    with ctrl2:
        filter_thread  = st.text_input("Filter by Thread ID",  key="aud_thread",  placeholder="thread-uuid…")
    with ctrl3:
        filter_status  = st.selectbox("Status", ["ALL", "SUCCESS", "ESCALATED", "WARNING", "PENDING"], key="aud_status")
    with ctrl4:
        live_mode = st.checkbox("⚡ Live (5s)", key="aud_live")
    with ctrl5:
        refresh_btn = st.button("⟳ Refresh", key="aud_refresh")

    # ── Fetch ──────────────────────────────────────────────────────────────────
    def _fetch_audit(session_id=None, thread_id=None):
        try:
            params = {}
            if session_id and session_id.strip():
                params["session_id"] = session_id.strip()
            elif thread_id and thread_id.strip():
                params["thread_id"] = thread_id.strip()
            r = requests.get(AUDIT_URL, params=params, timeout=(5, 10))
            r.raise_for_status()
            return r.json().get("events", [])
        except Exception as e:
            st.warning(f"Could not reach audit endpoint: {e}")
            return []

    events = _fetch_audit(filter_session, filter_thread)

    # Apply status filter client-side
    if filter_status != "ALL":
        events = [e for e in events if e.get("status") == filter_status]

    # Show newest first
    events = list(reversed(events))

    # ── Stats bar ──────────────────────────────────────────────────────────────
    total = len(events)
    st1, st2, st3, st4, st5 = st.columns(5)
    st1.metric("Total Events",  total)
    st2.metric("✅ SUCCESS",    sum(1 for e in events if e["status"] == "SUCCESS"))
    st3.metric("⚠️ WARNING",   sum(1 for e in events if e["status"] == "WARNING"))
    st4.metric("🔶 ESCALATED", sum(1 for e in events if e["status"] == "ESCALATED"))
    st5.metric("⏳ PENDING",   sum(1 for e in events if e["status"] == "PENDING"))

    # Failure count shown prominently if any exist
    fail_count = sum(1 for e in events if e["status"] == "FAILURE")
    if fail_count:
        st.error(f"❌ {fail_count} FAILURE event(s) detected — check the Message column below for exception details.")

    st.divider()

    if not events:
        st.info("No audit events yet. Send a query in the Chat tab to start generating events.")
    else:
        # ── Table header ───────────────────────────────────────────────────────
        st.markdown(
            '<div class="aud-table">'
            '<div class="aud-row aud-header">'
            '<span>Event ID</span>'
            '<span>Timestamp</span>'
            '<span>Session</span>'
            '<span>Node / Actor</span>'
            '<span>Event Type</span>'
            '<span>Status</span>'
            '<span>Message</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        STATUS_ICON = {
            "SUCCESS":  "✅",
            "WARNING":  "⚠️",
            "ESCALATED":"🔶",
            "PENDING":  "⏳",
            "FAILURE":  "❌",
        }

        for ev in events:
            status = ev.get("status", "")
            icon   = STATUS_ICON.get(status, "🔵")
            ts     = ev.get("ts", "").replace("T", " ").split("+")[0]
            session_short = (ev.get("session_id") or "")[:10] + "…"
            raw_msg = ev.get("message") or ""
            msg_class = "failure" if status == "FAILURE" else ""

            # Try to parse JSON message — render as key-value pills
            kv_html = ""
            try:
                kv = json.loads(raw_msg)
                if isinstance(kv, dict):
                    pills = []
                    for k, v in kv.items():
                        if v is None:
                            val_cls = "null-val"
                            v_str   = "null"
                        elif isinstance(v, bool):
                            val_cls = "bool-true" if v else "bool-false"
                            v_str   = str(v).lower()
                        else:
                            val_cls = ""
                            v_str   = str(v)[:60] + ("…" if len(str(v)) > 60 else "")
                        pills.append(
                            f'<span class="aud-kv-pair">'
                            f'<span class="aud-kv-key">{k}</span>'
                            f'<span class="aud-kv-sep">:</span>'
                            f'<span class="aud-kv-val {val_cls}">{v_str}</span>'
                            f'</span>'
                        )
                    kv_html = f'<span class="aud-kv">{"".join(pills)}</span>'
                else:
                    kv_html = f'<span class="aud-msg {msg_class}">{raw_msg}</span>'
            except (json.JSONDecodeError, TypeError):
                # Plain string (e.g. exception message from FAILURE events)
                kv_html = f'<span class="aud-msg {msg_class}">{raw_msg}</span>'

            st.markdown(
                f'<div class="aud-row">'
                f'<span class="aud-event-id">{ev["event_id"]}</span>'
                f'<span>{ts}</span>'
                f'<span style="font-family:monospace;font-size:0.72rem">{session_short}</span>'
                f'<span class="aud-node">{ev.get("node","")}</span>'
                f'<span class="aud-type">{ev.get("event_type","")}</span>'
                f'<span><span class="aud-badge aud-{status}">{icon} {status}</span></span>'
                f'<span>{kv_html}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Detail expander ────────────────────────────────────────────────────
        st.markdown("")
        with st.expander("🔎 Full Event Details (select from list)", expanded=False):
            event_options = {f"{e['event_id']} — {e['event_type']}": e for e in events}
            chosen_label  = st.selectbox("Event", list(event_options.keys()), key="aud_detail_sel")
            if chosen_label:
                st.json(event_options[chosen_label])

        # ── CSV export ─────────────────────────────────────────────────────────
        import csv
        buf = io.StringIO()
        if events:
            writer = csv.DictWriter(
                buf,
                fieldnames=["event_id","ts","session_id","thread_id","node","event_type","status","message","details"],
                extrasaction="ignore",
            )
            writer.writeheader()
            for ev in events:
                writer.writerow({**ev, "details": json.dumps(ev.get("details", {}))})
        st.download_button(
            "⬇ Export CSV", data=buf.getvalue(),
            file_name="resolveai_audit.csv", mime="text/csv", key="aud_csv",
        )

    # ── Live refresh ───────────────────────────────────────────────────────────
    if live_mode:
        time.sleep(5)
        st.rerun()