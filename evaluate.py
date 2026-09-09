"""Reproducible retrieval evaluation; generation review is deliberately separate."""

import argparse
import json
from pathlib import Path
import platform
import time

from rag import DenseIndex, EMBEDDING_MODEL, KeywordIndex, demo_chunks

ROOT = Path(__file__).parent


def evaluate(retriever="bm25", split="heldout"):
    cases = json.loads((ROOT / "eval_questions.json").read_text())
    selected = [case for case in cases if split == "all" or case["split"] == split]
    started = time.perf_counter()
    index = DenseIndex(demo_chunks()) if retriever == "semantic" else KeywordIndex(demo_chunks())
    index_seconds = time.perf_counter() - started
    rows = []
    latencies = []
    for case in selected:
        start = time.perf_counter()
        hits = index.search(case["question"], k=3)
        latencies.append(time.perf_counter() - start)
        retrieved = [chunk.source for chunk, _ in hits]
        expected = case["expected_sources"]
        recall = (len(set(retrieved) & set(expected)) / len(expected)) if expected else None
        rows.append({**case, "retrieved": retrieved, "recall_at_3": recall,
                     "top1_source_correct": bool(retrieved and retrieved[0] == expected[0]) if len(expected) == 1 else None,
                     "no_match": not retrieved if not case["answerable"] else None})
    answerable = [row for row in rows if row["answerable"]]
    single_source = [row for row in answerable if len(row["expected_sources"]) == 1]
    unsupported = [row for row in rows if not row["answerable"]]
    return {
        "scope": f"{retriever} retrieval on five original notes; {split} hand-authored split; generation not scored",
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "retriever": retriever, "embedding_model": EMBEDDING_MODEL if retriever == "semantic" else None},
        "counts": {"questions": len(rows), "answerable": len(answerable), "unsupported": len(unsupported)},
        "metrics": {
            "recall_at_3": sum(row["recall_at_3"] for row in answerable) / len(answerable) if answerable else None,
            "top1_source_accuracy": sum(row["top1_source_correct"] for row in single_source) / len(single_source) if single_source else None,
            "unsupported_no_match_rate": sum(row["no_match"] for row in unsupported) / len(unsupported) if unsupported else None,
            "index_seconds": index_seconds,
            "mean_query_ms": 1000 * sum(latencies) / len(latencies) if latencies else None,
        },
        "definitions": {
            "recall_at_3": "Mean fraction of expected source documents appearing in the first three retrieved chunks for answerable questions.",
            "top1_source_accuracy": "Fraction of single-source answerable questions whose first chunk comes from the expected source.",
            "unsupported_no_match_rate": "Fraction of unsupported questions for which retrieval returns zero chunks; this is not generation abstention.",
        },
        "cases": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", choices=["bm25", "semantic"], default="bm25")
    parser.add_argument("--split", choices=["development", "heldout", "all"], default="heldout")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.retriever, args.split)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)
