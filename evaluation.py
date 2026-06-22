
import os
import json
import time
from typing import List, Dict
import pandas as pd
from dotenv import load_dotenv
 
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
 
from vector_store import RAGVectorStore
from generation import generate_answer
 
load_dotenv()
 
# Use the fast, cheap 8B model as the "judge" so the whole pipeline stays free.
JUDGE_MODEL = "llama-3.1-8b-instant"
 
 
def load_golden_dataset(path: str = "golden_dataset.json") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
 
 
def run_evaluation(vector_store: RAGVectorStore, golden_dataset_path: str = "golden_dataset.json",
                    retrieval_mode: str = "hybrid", k: int = 4) -> pd.DataFrame:
    """
    Runs every question in the golden dataset through the full RAG
    pipeline (retrieve -> generate), then scores the results with RAGAS.
    Returns a DataFrame with one row per question and a score per metric,
    so you can both see aggregate numbers and drill into individual
    failures.
    """
    golden_data = load_golden_dataset(golden_dataset_path)
 
    questions, answers, contexts, ground_truths = [], [], [], []
 
    for item in golden_data:
        question = item["question"]
        ground_truth = item["ground_truth"]
 
        retrieved = vector_store.retrieve(question, k=k, mode=retrieval_mode)
        result = generate_answer(question, retrieved)
 
        questions.append(question)
        answers.append(result["answer"])
        contexts.append([c["text"] for c in retrieved])
        ground_truths.append(ground_truth)
 
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # Wire RAGAS up to use Groq (free) instead of defaulting to OpenAI.
    judge_llm = ChatGroq(model=JUDGE_MODEL, api_key=os.environ.get("GROQ_API_KEY"))
    ragas_llm = LangchainLLMWrapper(judge_llm)
    judge_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
 
    result = evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ragas_llm,
        embeddings=judge_embeddings,
    )
 
    results_df = result.to_pandas()
    return results_df
 
 
def summarize_results(results_df: pd.DataFrame) -> Dict:
    """Compute simple average scores per metric for the dashboard."""
    summary = {}
    for col in ["faithfulness", "answer_relevancy", "context_precision"]:
        if col in results_df.columns:
            summary[col] = round(results_df[col].mean(), 3)
    return summary
 
 
def save_results(results_df: pd.DataFrame, label: str):
    """Save timestamped results to /results so you can compare runs later."""
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = f"results/{label}_{timestamp}.csv"
    results_df.to_csv(path, index=False)
    print(f"Saved evaluation results to {path}")
    return path


