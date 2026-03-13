# evaluation/ragas_eval.py

import json
import time
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from agents.graph import run_query


def load_test_questions(path: str = "evaluation/test_questions.json") -> list[dict]:
    """Loads the ground truth test set."""
    with open(path) as f:
        return json.load(f)


def run_agent_on_question(question: dict) -> dict:
    """
    Runs the agent on a single test question and collects:
    - The generated answer
    - The retrieved contexts (raw chunk content)
    - The ground truth answer
    - Metadata for analysis

    RAGAS needs all four of these to compute its metrics.
    """
    print(f"  Running: {question['question'][:60]}...")

    result = run_query(
        query=question["question"],
        ticker=question["ticker"],
        year=question["year"],
        quarter=question["quarter"],
    )

    # Extract raw content from retrieved chunks for context evaluation
    contexts = []
    for chunk in (result.get("retrieved_chunks") or []):
        contexts.append(chunk["content"][:500])

    # For comparison queries, also include comparison chunks
    tool_results = result.get("tool_results") or {}
    if "comparison" in tool_results:
        for chunk in tool_results["comparison"].get("comparison", []):
            contexts.append(chunk["content"])

    return {
        "question": question["question"],
        "answer": result.get("final_answer", "")[:1000],
        "contexts": contexts if contexts else ["No context retrieved"],
        "ground_truth": question["ground_truth"],
        "query_type": question["query_type"],
        "question_id": question["id"],
    }


def build_ragas_dataset(results: list[dict]) -> Dataset:
    """
    Converts our results into the HuggingFace Dataset format
    that RAGAS expects.
    """
    return Dataset.from_dict({
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })


def run_evaluation(
    test_path: str = "evaluation/test_questions.json",
    output_path: str = f"evaluation/results_{time.strftime('%Y%m%d_%H%M%S')}_reranked.json",
    sleep_between: float = 2.0,
) -> dict:
    """
    Full evaluation pipeline:
    1. Load test questions
    2. Run agent on each question
    3. Score with RAGAS
    4. Save and return results

    sleep_between: seconds to wait between questions to avoid rate limits
    """
    print("\n" + "="*60)
    print("🧪 FinSight RAG Evaluation")
    print("="*60)

    questions = load_test_questions(test_path)
    print(f"\n📋 Running {len(questions)} test questions...\n")

    # Run agent on all questions
    results = []
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q['query_type'].upper()}")
        try:
            result = run_agent_on_question(q)
            results.append(result)
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results.append({
                "question": q["question"],
                "answer": f"ERROR: {e}",
                "contexts": ["Error during retrieval"],
                "ground_truth": q["ground_truth"],
                "query_type": q["query_type"],
                "question_id": q["id"],
            })

        # Avoid hitting OpenAI rate limits
        if i < len(questions) - 1:
            time.sleep(sleep_between)

    print(f"\n✅ Agent runs complete. Scoring with RAGAS...\n")

    # Build dataset and score
    dataset = build_ragas_dataset(results)

    # scores = evaluate(
    #     dataset,
    #     metrics=[
    #         faithfulness,
    #         answer_relevancy,
    #         context_precision,
    #         context_recall,
    #     ],
    # )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    scorer_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    scorer_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    scores = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    llm=scorer_llm,
    embeddings=scorer_embeddings,
    )
    # # Build results summary
    # score_dict = scores.to_pandas().mean().to_dict()

    # summary = {
    #     "overall_scores": {
    #         "faithfulness": round(score_dict.get("faithfulness", 0), 3),
    #         "answer_relevancy": round(score_dict.get("answer_relevancy", 0), 3),
    #         "context_precision": round(score_dict.get("context_precision", 0), 3),
    #         "context_recall": round(score_dict.get("context_recall", 0), 3),
    #     },
    #     "per_question": results,
    #     "total_questions": len(questions),
    # }
    # Newer RAGAS versions return numeric columns mixed with string columns
    # Select only numeric columns before computing mean
    df = scores.to_pandas()
    numeric_df = df.select_dtypes(include="number")
    score_dict = numeric_df.mean().to_dict()

    summary = {
        "overall_scores": {
            "faithfulness": round(score_dict.get("faithfulness", 0), 3),
            "answer_relevancy": round(score_dict.get("answer_relevancy", 0), 3),
            "context_precision": round(score_dict.get("context_precision", 0), 3),
            "context_recall": round(score_dict.get("context_recall", 0), 3),
        },
        "per_question": results,
        "total_questions": len(questions),
    }
    # Save to disk
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary