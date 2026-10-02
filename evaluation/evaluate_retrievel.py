from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision, ContextRecall, AnswerCorrectness
from retrieval.multiturn_retrieval import TurnChainRetriever
from langchain_openai import OpenAIEmbeddings
import sys
from langfuse.openai import AsyncOpenAI
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.support_agent_workflow import call_workflow

# Setup LLM
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)
recall_scorer = ContextRecall(llm=llm)
embeddings = OpenAIEmbeddings()
correctness_scorer = AnswerCorrectness(llm=llm, embeddings=embeddings)
precision_scorer = ContextPrecision(llm=llm)


class RetrievelEval:
    def __init__(self, retriever, user_input, response):
        self.context = retriever.search(user_input, top_k_children=5, top_n_parents=3)
        self.response = response

    def precision_score_retrievel(self, user_input, reference):
        # Bug fix: initialise result to None so the print below never hits UnboundLocalError
        result = None
        if self.response:
            result = precision_scorer.score(
                user_input=user_input,
                reference=reference,
                retrieved_contexts=[
                    r["parent_thread_text"] for r in self.context
                ],
            )
        value = result.value if result is not None else None
        print(f"Context Precision Score: {value}")
        return value

    def recall_score_retrievel(self, user_input, reference):
        # Bug fix: initialise result to None so the print below never hits UnboundLocalError
        result = None
        if self.response:
            result = recall_scorer.score(
                user_input=user_input,
                reference=reference,
                retrieved_contexts=[
                    r["parent_thread_text"] for r in self.context
                ],
            )
        value = result.value if result is not None else None
        print(f"Context Recall Score: {value}")
        return value

    def answer_correctness_eval(self, user_input, reference):
        # Bug fix: method now returns the score value
        result = None
        if self.response:
            result = correctness_scorer.score(
                user_input=user_input,
                reference=reference,
                response=self.response,
            )
        value = result.value if result is not None else None
        print(f"Answer Correctness Score: {value}")
        return value  # Bug fix: was missing return


def eval_retrieval(user_query, reference, response):
    retriever = TurnChainRetriever()
    eval_retriver = RetrievelEval(retriever, user_query, response)

    # Bug fix: compute each metric once and store — avoids double API calls
    # Bug fix: last line called eval_retrieval (the function) instead of eval_retriver (instance)
    precision  = eval_retriver.precision_score_retrievel(user_query, reference)
    recall     = eval_retriver.recall_score_retrievel(user_query, reference)
    correctness = eval_retriver.answer_correctness_eval(user_query, reference)

    return precision, recall, correctness