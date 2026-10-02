"""
source/reference:
https://github.com/AkarshVyas/Agentic-AI-youtube/blob/main/humanintheloop.py

   
"""
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import (
    StateGraph,
    START,
    END,
)


from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel,Field
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver,InMemorySaver
from langgraph.types import interrupt, Command
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path
from langchain_ollama import ChatOllama
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from intent.intent_classification import classify_intent
# from retrieval.muliturn_retrieval_sample import TurnChainRetriever
from retrieval.multiturn_retrieval import TurnChainRetriever
import datetime
import uuid
load_dotenv()


# ============================================================
# CONVERSATION MEMORY
# ============================================================

class ConversationMemory:
    """
    Lightweight in-process memory store.

    Keeps one compact record per turn, keyed by session_id.
    Each record captures the decisions made during that turn so that
    subsequent turns have enough context to continue coherently.
    """

    def __init__(self):
        # { session_id: [turn_record, ...] }
        self._store: dict[str, list[dict]] = {}

    # ── write ─────────────────────────────────────────────────────────────────

    def record_turn(
        self,
        session_id: str,
        query: str,
        intent: str,
        retrieval_required: bool,
        escalated: bool,
        hitl_reason: str,
        context_summary: str,
        response: str,
        approved: bool | None,
    ) -> None:
        """Append a compact record for one completed turn."""
        if session_id not in self._store:
            self._store[session_id] = []

        # Summarise long fields to 1 sentence each (truncate — no extra LLM call)
        self._store[session_id].append({
            "turn":        len(self._store[session_id]) + 1,
            "ts":          datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "query":       query[:200],
            "intent":      intent[:120],
            "retrieval":   retrieval_required,
            "escalated":   escalated,
            "hitl_reason": hitl_reason[:200] if hitl_reason else "",
            "context":     context_summary[:300] if context_summary else "(no context)",
            "response":    response[:300] if response else "(no response)",
            "approved":    approved,
        })

    # ── read ──────────────────────────────────────────────────────────────────

    def get_context_block(self, session_id: str, max_turns: int = 5) -> str:
        """
        Return a compact memory block (last N turns) formatted as a
        system-message string that can be prepended to the LLM context.
        """
        turns = self._store.get(session_id, [])
        if not turns:
            return ""

        recent = turns[-max_turns:]
        lines = ["=== CONVERSATION HISTORY ==="]
        for t in recent:
            lines.append(
                f"[Turn {t['turn']}] "
                f"User: {t['query']} | "
                f"Intent: {t['intent']} | "
                f"Retrieval: {t['retrieval']} | "
                f"Escalated: {t['escalated']} | "
                f"Response summary: {t['response']}"
            )
        lines.append("===========================")
        return "\n".join(lines)

    def has_session(self, session_id: str) -> bool:
        return bool(self._store.get(session_id))

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)


# Module-level singleton — shared across all requests in this process
_memory = ConversationMemory()


langfuse = get_client()

langfuse_handler = CallbackHandler()
# ============================================================
# LLM
# ============================================================
def get_llm():
     return ChatOpenAI(
        model="gpt-5.4",
        temperature=0.2,
    )

SUPPORT_SYSTEM_PROMPT="""
You are a helpful ai support agent  for spotify brand. Your job is to solve the customer issue based on the how the the same issue is solved in past history.
                Be professional and polite. Always respond with empathy and provide clear solutions. Make sure that you only answer from given historical evidences and do not invent new solutions/things on your own.You need to answer strictly on provided evidences.
                If the the user asking general questions then respond with appropriate message.
                
                Tone:
                - Use simple language. Avoid Robotic Phrasing 
                - Acknowledge customer inquiries first with warm message

                Rules:
                - If the user is greeting or showing gratitude then respond with war message
                - Use given historical evidence to generate effective solution if present
                - If a user asks a question outside your defined role or responsibilities, respond with:- "This question falls outside the scope of my current knowledge"
    

                Input Format:
                - you are given a given multiple conversation between customer and brand in multi-turn fashion customer->brand->customer->brand
                - Intent of the customer with description
"""
class HumanEscalationSchema(BaseModel):
    human_escalation:bool = Field(description="True if the message should be escalated to a human, False if it can be auto-handled")
    escalation_reason:str = Field(description="The stated reason for why the message was escalated or auto-handled")
# ============================================================
# STATE
# ============================================================
class RetrievalSchema(BaseModel):
    is_retrieval_required:bool
class SupportState(TypedDict):
    query: str
    session_id: str              # persistent across multi-turn conversation

    messages: Annotated[list, add_messages]

    # Retrieved historical evidence
    historical_evidence: str
    intent_name: str
    intent_description: str
    # Generated response
    draft: str

    # Human feedback
    review_feedback: str

    # HITL state
    is_approved: bool
    human_required: bool
    hitl_reason: str
    is_retrieval_required: bool

    # Retrieval information
    retrieval_results: list

    # Memory context block injected at start of each turn
    memory_context: str


# ============================================================
# SYSTEM PROMPT
# ============================================================

# ============================================================
# STEP 1
# RETRIEVE HISTORICAL EVIDENCE
# ============================================================
def route_after_retrieval_decision(state: SupportState):

    if state.get("is_retrieval_required", False):

        print(
            "\n[Router] Retrieval required -> RETRIEVAL"
        )

        return "retrieval"

    print(
        "\n[Router] Retrieval not required -> WRITER"
    )

    return "writer"
def decision_node(state:SupportState):
        query = state["messages"]
        intent = state.get("intent_name", "")
        intent_description = state.get("intent_description", "")
    
        
        llm=get_llm().with_structured_output(RetrievalSchema)
        prompt=ChatPromptTemplate.from_messages(
                 [(
                    "system",
                    """You are a customer support AI agent for Spotify. Your role is to help users resolve issues related to albums, songs, accounts, bugs, content availability, account access, Family Plan issues, subscriptions, billing and plan charges, and playback problems.
Guidelines:

-Clarify Vague Queries: If the user's input does not provide enough context to understand their problem, ask a clear follow-up question to get more details.

-Determine Retrieval Needs: Decide whether database or information retrieval is necessary based on the user's intent:

-Do not trigger retrieval (is_retrieval_required: false) if the user is simply greeting you, expressing gratitude, or offering praise.

-Do trigger retrieval (is_retrieval_required: true) if the user is asking for help with an issue or troubleshooting.

Output Format:
Return ONLY valid JSON.

    """,
                ),
                ("human", "{query}"),
            ]
            )
        evidence_check_chain=prompt | llm 
        response=evidence_check_chain.invoke({"query":f"""
    CUSTOMER QUERY:
    
    {query}
    
    
    CUSTOMER INTENT:
    
    {intent[0]}
    
    
    INTENT DESCRIPTION:
    
    {intent_description[0]}
    
    
  
    """},config={
        "callbacks": [langfuse_handler]
    })
        
        langfuse.flush()
        return {
            "is_retrieval_required":response.is_retrieval_required
        }
def retrieval_node(state: SupportState) -> dict:
    """
    Retrieve historically similar customer-support conversations.
    """

    query = state["query"]

    print("\n[Retrieval]")
    print(f"Customer query: {query}")

    retriever = TurnChainRetriever()

    results = retriever.search(
        query,
        top_k_children=10,
        top_n_parents=5,
    )

    if not results:
        print("[Retrieval] No historical evidence found.")
        
        return {
            "retrieval_results": [],
            "historical_evidence": "",
            "human_required": True,
        }

    historical_evidence = "\n\n".join(
        [
            f"""
--- HISTORICAL THREAD {i + 1} ---

{result["parent_thread_text"]}
"""
            for i, result in enumerate(results)
        ]
    )

    print(
        f"[Retrieval] Retrieved {len(results)} historical threads."
    )

    
    state["retrieval_results"]= results
    state["historical_evidence"]=historical_evidence
    
    return state


# ============================================================
# STEP 2
# DETERMINE WHETHER EVIDENCE IS SUFFICIENT
# ============================================================

def evidence_check_node(state: SupportState) -> dict:
    """
    Basic deterministic evidence gate.
    """
    query = state["query"]
    intent = state.get("intent_name", "")
    intent_description = state.get("intent_description", "")
    evidence = state.get("historical_evidence", "")

    
    llm=get_llm().with_structured_output(HumanEscalationSchema)
    prompt=ChatPromptTemplate.from_messages(
             [(
                "system",
                """
You are a quality-control gatekeeper for Spotify's customer support system. You are shown a
customer's message, the system's classified intent for that message (with a confidence
score), and one or more retrieved passages that a retrieval system selected as potentially
relevant to answering the customer.

Your job is NOT to write the customer-facing reply. Your job is to decide whether the
retrieved passages are sufficient and safe to auto-approve a response built from them, or
whether this case must be escalated to a human agent instead.

AUTO_APPROVE if any of the following are true:
- If the customer asks for general, non-account-specific information such as rate limits, plan limits, feature availability, supported devices, or general list/name information, treat the request as low-risk and eligible for AUTO_APPROVE when the answer is supported by reliable evidence.
- Do not require human review for general informational questions unless the information is unavailable, conflicting, outdated, or requires access to the customer's private account.
- If the requested information or solution is currently unavailable but can be obtained by asking the customer for missing details, do not escalate immediately; ask a concise clarifying question and continue the workflow
- If the user greets customer or starts the conversation  
- The answer is generic/informational and requires no account-specific, order-specific,
  or case-specific lookup or action to be correct.
- The topic is not billing, refunds, cancellations, account security, legal, safety,
  self-harm, harassment, or any sign of customer distress requiring human judgment.
- The customer's message is not sarcastic, ambiguous, multi-part, or dependent on
  thread context you weren't given.
- if the customer complementing the service or sending feedback or saying gratitude
- if the spotify is down, then the brand response may include changing browser or using incognito mode
- Customer may want to contact to spotify support but the there is lack of context given 
- example brand may provides information to solve the problem. they may request customer to visit to link to check if the unavailable song album are there are added and restarting device,use incognito mode or reinstalling the latest spotify version to check the device compatibility issue. if the user request is unclear or vague like not available without any context given where the brand may ask additional information like device, os and app version.


Decide ESCALATE rather than AUTO_APPROVE if ANY of the following are true:
- The retrieved passages do not directly and completely address the customer's specific
  question or problem (partial overlap is not enough).
- Answering correctly requires account-specific, order-specific, or case-specific action or
  information that is not present in the retrieved passages (e.g. checking a real account,
  issuing a refund, verifying identity, looking up a specific order or payment).
- The message involves billing disputes, refunds, cancellations, account security or
  suspected account compromise, legal threats, safety concerns, self-harm, harassment,hacking or
  any indication the customer is distressed and needs a human's judgment or empathy rather
  than a templated answer.
- for example brand may ask for additional information from customer like sending username and email address. they brand may note the feedback from the customer which is later analyzed by human assistant. the brand may say the phone support is not availble and send mail to spotify service where team can analyze the issue

Otherwise, if the retrieved passages clearly, completely, and safely answer exactly what the
customer asked, decide AUTO_APPROVE.

When in doubt, escalate. A missed auto-approval costs a few minutes of human review time; a
wrong auto-approved answer costs customer trust and may require correcting misinformation
later. Bias toward caution.

Return ONLY valid JSON.

""",
            ),
            ("human", "{evidence}"),
        ]
        )
    evidence_check_chain=prompt | llm 
    response=evidence_check_chain.invoke({"evidence":f"""
CUSTOMER QUERY:

{query}


CUSTOMER INTENT:

{intent}


INTENT DESCRIPTION:

{intent_description}


HISTORICAL SUPPORT EVIDENCE:

{evidence}
"""},config={
        "callbacks": [langfuse_handler]
    })
    # Guard: structured-output parsing can return None if the model
    # response doesn't match the schema (e.g. malformed JSON).
    # Default to human escalation — the safer fallback.
    if response is None:
        print(
            "[evidence_check_node] WARNING: structured-output returned None "
            "(LLM response did not match HumanEscalationSchema). "
            "Defaulting to human escalation."
        )
        langfuse.flush()
        return {
            "human_required": True,
            "hitl_reason": "Structured-output parsing failed; escalating as a precaution.",
        }

    if response.human_escalation:
        return {
            "human_required": True,
            "hitl_reason": response.escalation_reason,
        }

    langfuse.flush()
    return {
        "human_required": False,
    }


# ============================================================
# STEP 3
# GENERATE RESPONSE
# ============================================================

def writer_node(state: SupportState) -> dict:
    """
    Generate a customer-support response from historical evidence.
    """

    writer_llm=get_llm()
    query = state["messages"]
    evidence = state.get("historical_evidence","")
    feedback = state.get("review_feedback", "")
    intent=state.get("intent_name","")
    intent_description=state.get("intent_description","")

    user_message = f"""
Customer message:

{query}


Historical customer-support evidence:

{evidence}


Previous response was rejected by human support.

Human reviewer feedback:

{feedback}
intent:
{intent}
intent description:
{intent_description}

Generate a NEW response.

Fix every issue mentioned by the reviewer.

The new response must still be strictly grounded in the
historical evidence.
"""

    response = writer_llm.invoke(
        [
            ("system", SUPPORT_SYSTEM_PROMPT),
            ("human", user_message),
        ],config={
        "callbacks": [langfuse_handler]
    }
    )

    langfuse.flush()
    draft = response.content

    print("\n[Generated response]")
    print("-" * 60)
    print(draft)
    print("-" * 60)

    return {
        "draft":    draft,
        "messages": response,    # AIMessage — add_messages handles accumulation
    }


# ============================================================
# STEP 3b  — MEMORY RECORDER
# Runs after writer_node; commits the turn to ConversationMemory.
# ============================================================

# def memory_node(state: SupportState) -> dict:
#     """
#     Record the completed turn into ConversationMemory and return
#     an updated memory_context so the NEXT turn has history.
#     """
#     session_id = state.get("session_id", "default")

#     # Summarise context to first 300 chars of first thread
#     raw_ctx = state.get("historical_evidence", "") or ""
#     ctx_summary = raw_ctx[:300].replace("\n", " ").strip() if raw_ctx else "(no context)"

#     _memory.record_turn(
#         session_id        = session_id,
#         query             = state.get("query", ""),
#         intent            = state.get("intent_name", ""),
#         retrieval_required= bool(state.get("is_retrieval_required", False)),
#         escalated         = bool(state.get("human_required", False)),
#         hitl_reason       = state.get("hitl_reason", ""),
#         context_summary   = ctx_summary,
#         response          = (state.get("draft") or "")[:300],
#         approved          = state.get("is_approved"),
#     )

#     new_context = _memory.get_context_block(session_id)
#     print(f"[Memory] Session '{session_id}' — "
#           f"{len(_memory._store.get(session_id, []))} turn(s) recorded.")

#     return {"memory_context": new_context}


# ============================================================
# STEP 4
# HUMAN REVIEW
# ============================================================

def human_review_node(state: SupportState) -> dict:
    """
    Pause the graph and request human review.
    """


    human_response = interrupt(
        {
            "type": "support_response_review",

            "customer_message": state["query"],

            "historical_evidence": state[
                "historical_evidence"
            ],

            "draft": state.get("draft","No draft Available"),


            "instruction": (
                "Approve the response, reject it with feedback, "
                "or take over the conversation."
            ),

            "allowed_actions": [
                "approve",
                "reject",
                "takeover",
            ],
        }
    )

    # --------------------------------------------------------
    # HUMAN INPUT
    # --------------------------------------------------------

    if isinstance(human_response, str):

        response = human_response.strip()

        if response.lower() in {
            "approve",
            "approved",
            "yes",
            "ok",
        }:

            return {
                "is_approved": True,
                "human_required": False,
                "review_feedback": "Approved by human.",
            }

        if response.lower() in {
            "takeover",
            "take over",
            "human",
        }:

            return {
                "is_approved": False,
                "human_required": True,
                "review_feedback": "Human takeover requested.",
            }

        return {
            "is_approved":False,
            "human_required": False,
            "review_feedback": response,
        }
    return {
        "is_approved": False,
        "human_required": True,
        "review_feedback": "Human takeover requested.",
    }



# ============================================================
# STEP 5
# ROUTING
# ============================================================

def route_after_evidence(state: SupportState):

    if state.get("human_required", False):

        print(
            "\n[Router] Evidence insufficient -> HUMAN REVIEW"
        )

        return "human_review"

    print(
        "\n[Router] Evidence sufficient -> WRITER"
    )

    return "writer"


def route_after_review(state: SupportState):

    if state.get("is_approved", False):

        print(
            "\n[Router] Response approved -> END"
        )
        return END
    if state.get("human_required", False):

        print(
            "\n[Router] Human takeover -> END"
        )

        return END



 

    return "writer"


# ============================================================
# GRAPH
# ============================================================

graph = StateGraph(SupportState)

graph.add_node("retrieval",     retrieval_node)
graph.add_node("evidence_check", evidence_check_node)
graph.add_node("writer",         writer_node)
# graph.add_node("memory",         memory_node)   # ← new memory recorder
graph.add_node("decide_retrieval", decision_node)
graph.add_node("human_review",   human_review_node)

# START → decide_retrieval
graph.add_edge(START, "decide_retrieval")

# decide_retrieval → retrieval | writer
graph.add_conditional_edges(
    "decide_retrieval",
    route_after_retrieval_decision,
    {"retrieval": "retrieval", "writer": "writer"},
)

# retrieval → evidence_check
graph.add_edge("retrieval", "evidence_check")

# evidence_check → writer | human_review
graph.add_conditional_edges(
    "evidence_check",
    route_after_evidence,
    {"writer": "writer", "human_review": "human_review"},
)

# human_review → writer | END
graph.add_conditional_edges(
    "human_review",
    route_after_review,
    {"writer": "writer", END: END},
)

# writer → memory → END
# graph.add_edge("writer", "memory")
# graph.add_edge("memory", END)


# ============================================================
# SHARED APP-LEVEL GRAPH  (compiled once; checkpointer persists
# across requests so /review can resume a paused thread)
# ============================================================

_checkpointer = InMemorySaver()
_compiled_app  = None   # lazy-initialised on first request


def build_app():
    """Return the compiled LangGraph app (singleton per process)."""
    global _compiled_app
    if _compiled_app is None:
        _compiled_app = graph.compile(checkpointer=_checkpointer)
    return _compiled_app


def _build_initial_state(user_query: str, session_id: str) -> dict:
    """Classify intent and build the initial SupportState dict."""
    intent_result = classify_intent(user_query)

    valid_intents       = []
    intent_descriptions = []

    for item in intent_result:
        llm_validation = item.get("llm_validation", {}) or {}
        llm_response   = llm_validation.get("response", {}) or {}

        if llm_validation.get("is_correct") is True:
            valid_intents.append(str(item.get("intent_label", "")))
        elif llm_response.get("corrected_label"):
            valid_intents.append(str(llm_response.get("corrected_label", "")))
        elif item.get("intent_label"):
            valid_intents.append(str(item.get("intent_label", "")))

        if llm_response.get("description"):
            intent_descriptions.append(str(llm_response.get("description", "")))
        elif str(item.get("description", "")).strip():
            intent_descriptions.append(str(item.get("description", "")))

    # Inject memory from previous turns of this session as a system message
    # memory_ctx = _memory.get_context_block(session_id)
    # initial_messages = []
    # if memory_ctx:
    #     initial_messages.append({"role": "system", "content": memory_ctx})
    # initial_messages.append({"role": "user", "content": user_query})

    return {
        "query":      user_query,
        "session_id": session_id,
        "messages":  [{"role":"user","content":user_query}],
        # "messages": initial_messages ,
        # "memory_context": memory_ctx,
        "intent_name": (
            "\n".join(valid_intents) if valid_intents
            else (intent_result[0].get("intent_label", "") if intent_result else "")
        ),
        "intent_description": (
            "\n".join(intent_descriptions) if intent_descriptions
            else (intent_result[0].get("description", "") if intent_result else "")
        ),
    }


def call_workflow_start(user_query: str, session_id: str | None = None):
    """
    Start a new workflow for *user_query*.

    Parameters
    ----------
    user_query : str
    session_id : str, optional
        Persistent session identifier for multi-turn memory.
        If omitted, a new session is created.

    Returns
    -------
    result    : dict  — LangGraph state (may contain ``__interrupt__``)
    thread_id : str   — pass to ``call_workflow_resume`` if interrupted
    session_id: str   — pass back on subsequent turns
    """
    if not session_id:
        session_id = str(uuid.uuid4())
    # thread_id = str(uuid.uuid4())
    thread_id = session_id
    config    = {"configurable": {"thread_id": thread_id},"callbacks":[langfuse_handler]}
    result    = build_app().invoke(
        _build_initial_state(user_query, session_id), config=config
    )
    _log_result(result)
    return result, thread_id, session_id


def call_workflow_resume(thread_id: str, human_decision: str) -> dict:
    """
    Resume a paused workflow with the human reviewer's decision.

    Parameters
    ----------
    thread_id      : returned by ``call_workflow_start``
    human_decision : "approve" | "takeover" | free-text feedback
    """
    config = {"configurable": {"thread_id": thread_id},"callbacks":[langfuse_handler]}
    result = build_app().invoke(Command(resume=human_decision), config=config)
    _log_result(result)
    return result


def _log_result(result: dict) -> None:
    """Print a concise server-console summary."""
    print("\n" + "=" * 70)
    print("WORKFLOW RESULT")
    print("=" * 70)
    print(f"Query:    {result.get('query', '')}")
    print(f"Draft:    {str(result.get('draft', 'No draft'))[:200]}")
    print(f"Approved: {result.get('is_approved')}")
    print(f"HITL:     {result.get('hitl_reason', '')}")
    print(f"Feedback: {result.get('review_feedback', '')}")
    if "__interrupt__" in result:
        print("[INTERRUPTED — awaiting human review via /review endpoint]")
    print("=" * 70 + "\n")


# ============================================================
# CLI ENTRY-POINT — blocking input() loop for local script runs
# Do NOT call from FastAPI; use call_workflow_start / call_workflow_resume
# ============================================================

def call_workflow(user_query: str) -> dict:
    result, thread_id = call_workflow_start(user_query)

    while "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0].value

        print("\n" + "=" * 70)
        print("HUMAN REVIEW REQUIRED")
        print("=" * 70)
        print(f"\nCustomer:\n{interrupt_data.get('customer_message', '')}")
        print(f"\nEvidence:\n{interrupt_data.get('historical_evidence', '')}")
        print(f"\nDraft:\n{interrupt_data.get('draft', '')}")
        print("\nActions:")
        print("  approve   -> approve response")
        print("  feedback  -> provide feedback and regenerate")
        print("  takeover  -> human handles customer")

        human_input = input("\nYour decision:\n> ").strip()
        result = call_workflow_resume(thread_id, human_input)

    return result

if __name__ == "__main__":
    call_workflow("emailed about the double charges two weeks back")

