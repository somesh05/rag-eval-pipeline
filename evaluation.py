
import json
import os
import time
from typing import Dict, List

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from vector_store import RAGVectorStore
from generation import generate_answer

load_dotenv()

# Use the fast, cheap 8B model as the "judge" so the whole pipeline stays free.
JUDGE_MODEL = "openai/gpt-oss-120b"


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
    """
    Compute average scores per metric for the dashboard.

    RAGAS sometimes fails to score an individual question for a given
    metric (the judge LLM doesn't return a parseable result) and records
    that as NaN rather than raising an error. A plain .mean() silently
    skips those rows, so the "average" can end up computed over a tiny,
    unrepresentative slice of the dataset without any indication of that
    having happened. This version reports the valid/total count alongside
    every average so a low sample size is impossible to miss, and flags
    metrics where fewer than half the questions produced a usable score.
    """
    total = len(results_df)
    summary = {}
    for col in ["faithfulness", "answer_relevancy", "context_precision"]:
        if col not in results_df.columns:
            continue
        valid = results_df[col].dropna()
        n_valid = len(valid)
        mean_score = round(valid.mean(), 3) if n_valid > 0 else None
        summary[col] = mean_score
        summary[f"{col}_n_valid"] = n_valid
        summary[f"{col}_n_total"] = total
        summary[f"{col}_low_sample"] = n_valid < (total / 2)
    return summary


def save_results(results_df: pd.DataFrame, label: str):
    """Save timestamped results to /results so you can compare runs later."""
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = f"results/{label}_{timestamp}.csv"
    results_df.to_csv(path, index=False)
    print(f"Saved evaluation results to {path}")
    return path
