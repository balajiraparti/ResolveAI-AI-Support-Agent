"""
source/reference:
https://github.com/AkarshVyas/Agentic-AI-youtube/blob/main/humanintheloop.py

   
"""
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from deepeval.tracing import observe,update_current_span
from langgraph.graph import (
    StateGraph,
    START,
    END,
)


from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import BaseModel,Field
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from deepeval.metrics import AnswerRelevancyMetric,FaithfulnessMetric
import sys
from pathlib import Path
from langchain_ollama import ChatOllama
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from intent.intent_classification import classify_intent
from retrieval.muliturn_retrieval_sample import TurnChainRetriever
# from retrieval.multiturn_retrieval import TurnChainRetriever
load_dotenv()
relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
faithfulness_metric = FaithfulnessMetric(threshold=0.8)

# ============================================================
# LLM
# ============================================================
def get_llm():
     return ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
    )

SUPPORT_SYSTEM_PROMPT="""
You are a helpful ai support agent expert for spotify brand. Your job is to solve the customer issue based on the how the the same issue is solved in past history.
                Be professional and polite. Always respond with empathy and provide clear solutions. Make sure that you only answer from given historical evidences and do not invent new solutions/things on your own.You need to answer strictly on provided evidences
                
                Tone:
                - Use simple language. Avoid Robotic Phrasing 
                - Acknowledge customer inquiries first with warm message

                Rules:
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

    messages: Annotated[list, add_messages]

    # Retrieved historical evidence
    historical_evidence: str
    intent_name: str
    intent_description: str
    # Generated response
    draft: str =None

    # Human feedback
    review_feedback: str
   
    # HITL state
    is_approved: bool
    human_required: bool
    hitl_reason:str
    is_retrieval_required:bool
 

    # Retrieval information
    retrieval_results: list


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
        query = state["query"]
        intent = state.get("intent_name", "")
        intent_description = state.get("intent_description", "")
    
        
        llm=get_llm().with_structured_output(RetrievalSchema)
        prompt=ChatPromptTemplate.from_messages(
                 [(
                    "system",
                    """You are customer support ai agent for spotify company. who helps to solve the issue realated album,songs, account, bug, content availability, account access,family plan issue, subscription problem, plan charging problem, playback issue 
     Decide if for the given query,intent and intent description the retrival is required is or not.
    if the user is greeting or having natural conversation then do not set is_retrieval_required=True
    if the user asking for any issue then is_retrieval_required=True   
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
    
    {intent}
    
    
    INTENT DESCRIPTION:
    
    {intent_description}
    
    
  
    """})
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
        update_current_span(
            retrieval_context=[],
            metadata={
                "retrieval_count": 0,
                "retrieval_status": "empty"
            }
        )

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
"""})
    if response.human_escalation:
        return {
            "human_required": True,
            "hitl_reason":response.escalation_reason
        }

    

    return {
        "human_required": False
    }


# ============================================================
# STEP 3
# GENERATE RESPONSE
# ============================================================
@observe(type="llm",metrics=[relevancy_metric,faithfulness_metric])
def writer_node(state: SupportState) -> dict:
    """
    Generate a customer-support response from historical evidence.
    """

    writer_llm=get_llm()
    query = state["query"]
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
        ]
    )

    draft = response.content

    print("\n[Generated response]")
    print("-" * 60)
    print(draft)
    print("-" * 60)

    return {
        "draft": draft,
       "messages":[response]
    }


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


graph.add_node(
    "retrieval",
    retrieval_node,
)

graph.add_node(
    "evidence_check",
    evidence_check_node,
)

graph.add_node(
    "writer",
    writer_node,
)
graph.add_node("decide_retrieval",decision_node)
graph.add_node(
    "human_review",
    human_review_node,
)

# START
graph.add_edge(
    START,
    "decide_retrieval",
)


# Retrieval → Evidence Check
graph.add_edge(
    "retrieval",
    "evidence_check",
)
graph.add_conditional_edges("decide_retrieval",route_after_retrieval_decision,{"retrieval":"retrieval","writer":"writer"})

# Evidence → Writer OR Human
graph.add_conditional_edges(
    "evidence_check",
    route_after_evidence,
    {
        "writer": "writer",
        "human_review": "human_review",
    },
)




# Human Review → Writer OR END
graph.add_conditional_edges(
    "human_review",
    route_after_review,
    {
        "writer": "writer",
        END: END,
    },
)


# ============================================================
# CHECKPOINTER
# ============================================================
@observe(type="agent")
def call_workflow(user_query:str):
    checkpointer = MemorySaver()
    intent_result=classify_intent(user_query)
    app = graph.compile(
        checkpointer=checkpointer

    )
    output_path = Path("resolveai_graph.png")

    png_data = app.get_graph().draw_mermaid_png()

    output_path.write_bytes(png_data)
    # config = {"configurable": {"thread_id": "spotify_session_15"}}
    # state={"query":user_query,"messages":[{"role":"user","content":user_query}],"intent_name":"\n".join(i["intent_label"]for i in intent_result),"intent_description":"\n".join(i["description"]for i in intent_result)}
    # result=app.invoke(state,config=config)

    # print(result.get("draft",None))
    # while "__interrupt__" in result:

    #         interrupt_data = (
    #             result["__interrupt__"][0].value
    #         )

    #         print("\n")
    #         print("=" * 70)
    #         print("HUMAN REVIEW REQUIRED")
    #         print("=" * 70)

    #         print(
    #             f"\nCustomer:\n"
    #             f"{interrupt_data['customer_message']}"
    #         )
    #         print(
    #             f"\nEvidence:\n"
    #             f"{interrupt_data['historical_evidence']}"
    #         )
    #         print(
    #             f"\nDraft:\n"
    #             f"{interrupt_data['draft']}"
    #         )

    #         print(
    #             "\nActions:"
    #         )

    #         print(
    #             "  approve  -> approve response"
    #         )

    #         print(
    #             "  feedback -> provide feedback and regenerate"
    #         )

    #         print(
    #             "  takeover -> human handles customer"
    #         )

    #         human_input = input(
    #             "\nYour decision:\n> "
    #         ).strip()

    #         result = app.invoke(
    #             Command(
    #                 resume=human_input
    #             ),
    #             config=config,
    #         )

    # print("\n")
    # print("=" * 70)
    # print("FINAL RESULT")
    # print("=" * 70)

    # print(
    #     f"\nCustomer:\n{result['query']}"
    # )

    # print(
    #     f"\nResponse:\n{result.get("draft","No Draft Available")}"
    # )
    # print(f"\n review")
    # print(f"\n Review Feedback:{result.get("review_feedback","No Review Available")}")
    # print(f"\n Human-in-the-loop reason:{result.get("hitl_reason","")}")
    # print(
    #     f"Approved: {result.get("is_approved",None)}"
    # )
    # print(f"\n evidence: {result.get("historical_evidence","No Evidence")}")
    # print(f"\n intent:{result.get("intent_name","No intent")} \n intent_description:{result.get("intent_description","No description")}")
    # return result

if __name__=="__main__":
    call_workflow("hi")
# # Export graph visualization
# output_path = Path("resolveai_graph.png")

# png_data = app.get_graph().draw_mermaid_png()

# output_path.write_bytes(png_data)