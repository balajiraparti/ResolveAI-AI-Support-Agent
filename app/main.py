from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from evaluation.custom_evaluation import custom_evaluator
from app.support_agent_workflow import call_workflow
load_dotenv()


class AskRequest(BaseModel):
    query: str
    
app = FastAPI(
    title="ResolveAI API",
    description="AI-powered customer support API",
    version="0.1.0"
)


    

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "ResolveAI"
    }



@app.post("/ask")
def ask(request: AskRequest):
    query=request.query
    result=call_workflow(request)
    query=result.get("query",None)
    response=result.get("draft",None)
    context=result.get("historical_evidence",None)
    if response and context:
        custom_evaluator(query)    
    return {
        "query":result.get("query",None),
        "response": result.get("draft",None),
        "human_required":result.get("human_required",None),
        "hitl_reason":result.get("hitl_reason",None),
        "is_approved":result.get("is_approved",None),
        "human_feedback":result.get("review_feedback",None),
        "historical_evidence":result.get("historical_evidence",None),
        "intent":result.get("intent_name",None),
        "intent_description":result.get("intent_description",None)      
    }