"""
    Used Custom Evaluator code from https://atalupadhyay.wordpress.com/2026/01/30/rag-evaluation-from-bleu-scores-to-production-ready-metrics/ 
"""
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama
from typing import List, Dict
import re
from langchain_huggingface import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings()
class CustomRAGEvaluator:
    def __init__(self, embedding_model):
        self.embeddings = embedding_model
        self.llm = ChatOllama(model="llama3.2:1b", temperature=0)  # Strong model for evaluation

    def context_precision(self, query: str, retrieved_chunks: List[str]) -> float:
        """
        Binary relevance scoring using LLM as judge
        """
        prompt = f"""Rate the relevance of the following text chunk to the query on a scale of 0-1,
        where 1 means highly relevant and 0 means completely irrelevant.
        Query: {query}
        Chunk: {{chunk}}
        Respond with only a number between 0 and 1."""
        scores = []
        for chunk in retrieved_chunks:
            response = self.llm.invoke(prompt.format(chunk=chunk))
            try:
                score = float(response.content.strip())
                scores.append(score)
            except:
                scores.append(0.0)
        return np.mean(scores) if scores else 0.0
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        emb1 = self.embeddings.embed_query(text1)
        emb2 = self.embeddings.embed_query(text2)
        return cosine_similarity([emb1], [emb2])[0][0]
    def answer_relevance(self, query: str, answer: str) -> float:
        """
        Measure semantic similarity between query and answer
        with penalty for redundant information
        """
        base_similarity = self.semantic_similarity(query, answer)
        # Penalty for length (redundancy detection simplified)
        query_len = len(query.split())
        answer_len = len(answer.split())
        if answer_len > query_len * 5:  # Answer much longer than needed
            penalty = 0.1
        else:
            penalty = 0
        return max(0, base_similarity - penalty)
    def extract_claims(self, text: str) -> List[str]:
        """Extract atomic claims from text using LLM"""
        prompt = f"""Break down the following text into individual factual claims.
        Each claim should be a single, verifiable statement.
        Text: {text}
        Format as a list with one claim per line starting with "- """
        response = self.llm.invoke(prompt)
        claims = [line.strip("- ").strip() for line in response.content.split("\n") if line.strip().startswith("-")]
        return claims
    def verify_claim(self, claim: str, context: str) -> Dict:
        """Verify if a claim is supported by context"""
        prompt = f"""Determine if the CLAIM is supported by the CONTEXT.
        Respond with one of: SUPPORTS, CONTRADICTS, or NOT_ENOUGH_INFO
        CONTEXT: {context}
        CLAIM: {claim}
        Reasoning:"""
        response = self.llm.invoke(prompt)
        verdict = response.content.strip().upper()
        if "SUPPORT" in verdict:
            return {"verdict": "SUPPORTS", "confidence": 1.0}
        elif "CONTRADICT" in verdict:
            return {"verdict": "CONTRADICTS", "confidence": 1.0}
        else:
            return {"verdict": "UNKNOWN", "confidence": 0.5}
    def faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Check what proportion of answer claims are supported by context
        """
        claims = self.extract_claims(answer)
        if not claims:
            return 1.0  # No claims made = vacuously faithful
        supported = 0
        context_text = "\n".join(contexts)
        for claim in claims:
            result = self.verify_claim(claim, context_text)
            if result["verdict"] == "SUPPORTS":
                supported += 1
        return supported / len(claims)
# Initialize evaluator
def custom_evaluator(query,response,context,ground_truth):
    evaluator = CustomRAGEvaluator(embeddings)
    test_query = query
    test_answer = response
    test_contexts = context
    print("=== Custom Metrics ===")
    print(f"Context Precision: {evaluator.context_precision(test_query, test_contexts):.3f}")
    print(f"Answer Relevance: {evaluator.answer_relevance(test_query, test_answer):.3f}")
    print(f"Faithfulness: {evaluator.faithfulness(test_answer, test_contexts):.3f}")