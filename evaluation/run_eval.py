# evaluation/run_eval.py

from evaluation.ragas_eval import run_evaluation


def print_scorecard(summary: dict):
    scores = summary["overall_scores"]

    print("\n" + "="*45)
    print("         FINSIGHT RAG — EVAL SCORECARD")
    print("="*45)
    print(f"  {'Metric':<25} {'Score':>8}")
    print("-"*45)
    print(f"  {'Faithfulness':<25} {scores['faithfulness']:>8.3f}")
    print(f"  {'Answer Relevancy':<25} {scores['answer_relevancy']:>8.3f}")
    print(f"  {'Context Precision':<25} {scores['context_precision']:>8.3f}")
    print(f"  {'Context Recall':<25} {scores['context_recall']:>8.3f}")
    print("-"*45)
    overall = sum(scores.values()) / len(scores)
    print(f"  {'Overall':<25} {overall:>8.3f}")
    print("="*45)
    print(f"\n  Questions evaluated: {summary['total_questions']}")
    print(f"  Results saved to: evaluation/results.json\n")


if __name__ == "__main__":
    summary = run_evaluation()
    print_scorecard(summary)