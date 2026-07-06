"""Evaluation harness (LangChain variant).

Scores the LangChain RAG pipeline over the same golden Q/A set as the
from-scratch project and writes reports to `eval/results/` (JSON + Markdown) so
the Evaluation dashboard can display them:

  * Retrieval metrics (keyless): context_recall@k, recall@1, MRR, refusal
    accuracy, avg cost/latency, plus per-question detail.
  * Generation metrics (need a real LLM, via Ragas): faithfulness,
    answer_relevancy, context_precision, context_recall.
  * An A/B mode (`--compare`) that scores vector vs hybrid retrieval.

Usage:
  python -m eval.run_eval                      # single run (+ Ragas if available)
  python -m eval.run_eval --no-ragas           # retrieval metrics only
  python -m eval.run_eval --min-recall 0.8     # fail (exit 1) if recall drops -> CI gate
  python -m eval.run_eval --compare            # A/B benchmark: vector vs hybrid
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag_lc.config import get_settings  # noqa: E402
from src.rag_lc.pipeline import RAGPipelineLC  # noqa: E402

GOLDEN = Path(__file__).parent / "golden_set.jsonl"
# Single source of truth with the API's /eval-results reader (repo-anchored).
RESULTS_DIR = Path(get_settings().eval_results_dir)
# Eval queries write their traces here (gitignored), not to the app's trace file,
# so /metrics keeps reflecting real user traffic only.
EVAL_TRACE_DIR = Path(__file__).parent / "traces"
REFUSAL_MARKER = "don't have enough information"


def load_golden() -> list[dict]:
    """Load the golden Q/A set (one JSON object per non-empty line)."""
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def _ordered_sources(ans) -> list[str]:
    """Retrieved sources in rank order, de-duplicated (first occurrence wins)."""
    seen: list[str] = []
    for c in ans.citations:
        if c.source not in seen:
            seen.append(c.source)
    return seen


def retrieval_metrics(rows: list[dict], pipeline: RAGPipelineLC) -> tuple[dict, list[dict]]:
    """Score the pipeline over the golden set.

    Returns (aggregate_metrics, per_question_detail). context_recall@k / recall@1 /
    MRR are over answerable questions; refusal accuracy over out-of-scope ones.
    """
    answerable_total = answerable_hits = top1_hits = 0
    rr_sum = 0.0
    refusal_total = refusal_correct = 0
    per_q: list[dict] = []

    for r in rows:
        ans = pipeline.answer(r["question"])
        ordered = _ordered_sources(ans)
        expected = set(r.get("expected_sources", []))
        is_refusal_q = len(expected) == 0
        refused = REFUSAL_MARKER in ans.answer.lower()
        rank = next((i for i, s in enumerate(ordered) if s in expected), None)

        if is_refusal_q:
            refusal_total += 1
            refusal_correct += int(refused)
            hit = refused
        else:
            answerable_total += 1
            answerable_hits += int(rank is not None)
            top1_hits += int(rank == 0)
            rr_sum += (1.0 / (rank + 1)) if rank is not None else 0.0
            hit = rank is not None

        per_q.append({
            "question": r["question"],
            "answer": ans.answer,
            "ground_truth": r.get("ground_truth", ""),
            "expected_sources": sorted(expected),
            "retrieved_sources": ordered,
            "refusal_question": is_refusal_q,
            "correct": hit,
            "first_relevant_rank": (rank + 1) if rank is not None else None,
            "cost_usd": round(ans.cost_usd, 8),
            "latency_ms": round(ans.latency_ms, 1),
        })

    metrics = {
        "context_recall_at_k": round(answerable_hits / answerable_total, 3) if answerable_total else None,
        "recall_at_1": round(top1_hits / answerable_total, 3) if answerable_total else None,
        "mrr": round(rr_sum / answerable_total, 3) if answerable_total else None,
        "refusal_accuracy": round(refusal_correct / refusal_total, 3) if refusal_total else None,
        "answerable_questions": answerable_total,
        "refusal_questions": refusal_total,
        "avg_cost_usd": round(sum(q["cost_usd"] for q in per_q) / len(per_q), 8) if per_q else 0,
        "avg_latency_ms": round(sum(q["latency_ms"] for q in per_q) / len(per_q), 1) if per_q else 0,
    }
    return metrics, per_q


def ragas_metrics(rows: list[dict], pipeline: RAGPipelineLC) -> dict | None:
    """Generation-quality metrics. Requires a real LLM and the `ragas` package;
    skips cleanly (returns None) otherwise."""
    if pipeline.settings.llm_provider != "openai":
        print("[ragas] skipped (LLM_PROVIDER != openai)")
        return None
    # Ragas builds its own LangChain ChatOpenAI / OpenAIEmbeddings judges, which read
    # OPENAI_API_KEY from the OS environment -- NOT our pydantic-settings `.env`. The
    # pipeline passes the key explicitly (so answers/embeddings already worked), but the
    # Ragas judge can't see it. Bridge the loaded key into the env var Ragas expects.
    if pipeline.settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = pipeline.settings.openai_api_key
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ragas] skipped (import failed: {exc})")
        return None

    top = pipeline.settings.top_k
    records = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for r in rows:
        if not r.get("expected_sources"):
            continue
        ans = pipeline.answer(r["question"])
        docs = list(pipeline.retriever.invoke(r["question"]))[:top]
        contexts = [d.page_content for d in docs] or [c.snippet for c in ans.citations]
        records["question"].append(r["question"])
        records["answer"].append(ans.answer)
        records["contexts"].append(contexts)
        records["ground_truth"].append(r["ground_truth"])

    try:
        ds = Dataset.from_dict(records)
        result = evaluate(
            ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
        )
        return {k: round(float(v), 3) for k, v in dict(result).items()}
    except Exception as exc:  # noqa: BLE001
        print(f"[ragas] skipped (evaluation failed: {exc}); ensure ragas<0.2 is installed")
        return None


def _providers(settings) -> dict:
    return {
        "llm": settings.llm_provider,
        "embedding": settings.embedding_provider,
        "framework": "langchain",
        "retrieval_mode": settings.retrieval_mode,
        "top_k": settings.top_k,
    }


def write_report(metrics: dict, ragas: dict | None, per_q: list[dict]) -> Path:
    """Write eval results to eval/results/ as a JSON file + a Markdown table."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    settings = get_settings()
    semantic = settings.embedding_provider in ("openai", "hf")

    payload = {
        "timestamp": stamp,
        "providers": _providers(settings),
        "retrieval_metrics": metrics,
        "ragas_metrics": ragas,
        "per_question": per_q,
    }
    (RESULTS_DIR / f"eval-{stamp}.json").write_text(json.dumps(payload, indent=2))

    refusal_note = "" if semantic else " _(needs semantic embeddings; not meaningful on fake path)_"
    lines = [
        "# Evaluation Results (LangChain variant)",
        "",
        f"- Run: `{stamp}`",
        f"- Providers: framework=`langchain`, llm=`{settings.llm_provider}`, "
        f"embedding=`{settings.embedding_provider}`, mode=`{settings.retrieval_mode}`, "
        f"top_k=`{settings.top_k}`",
        "",
        "## Retrieval metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Context recall@k (answerable) | {metrics['context_recall_at_k']} |",
        f"| Recall@1 | {metrics['recall_at_1']} |",
        f"| MRR | {metrics['mrr']} |",
        f"| Refusal accuracy (out-of-scope) | {metrics['refusal_accuracy']}{refusal_note} |",
        f"| Avg cost / query (USD) | {metrics['avg_cost_usd']} |",
        f"| Avg latency / query (ms) | {metrics['avg_latency_ms']} |",
    ]
    if ragas:
        lines += ["", "## Generation metrics (Ragas)", "", "| Metric | Value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in ragas.items()]
    md_path = RESULTS_DIR / f"eval-{stamp}.md"
    md_path.write_text("\n".join(lines) + "\n")
    return md_path


def run_compare(rows: list[dict]) -> Path:
    """A/B benchmark: vector vs hybrid retrieval over the same corpus."""
    base = get_settings()
    results: dict[str, dict] = {}
    for mode in ("vector", "hybrid"):
        s = base.model_copy(
            update={"retrieval_mode": mode, "enable_cache": False, "trace_dir": EVAL_TRACE_DIR}
        )
        pipeline = RAGPipelineLC(settings=s, reset_index=True)
        metrics, _ = retrieval_metrics(rows, pipeline)
        results[mode] = metrics
        print(f"\n=== mode={mode} ===")
        for key in ("context_recall_at_k", "recall_at_1", "mrr", "avg_latency_ms"):
            print(f"  {key}: {metrics[key]}")

    def delta(metric: str) -> str:
        v, h = results["vector"][metric], results["hybrid"][metric]
        if v is None or h is None:
            return "n/a"
        d = round(h - v, 3)
        return f"+{d}" if d > 0 else str(d)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (RESULTS_DIR / f"compare-{stamp}.json").write_text(
        json.dumps(
            {
                "timestamp": stamp,
                "providers": {"embedding": base.embedding_provider, "top_k": base.top_k},
                "results": results,
            },
            indent=2,
        )
    )
    lines = [
        "# Retrieval A/B: vector vs hybrid (LangChain variant)",
        "",
        f"- Run: `{stamp}` · embedding=`{base.embedding_provider}` · top_k=`{base.top_k}`",
        "",
        "| Metric | vector | hybrid | delta |",
        "|---|---|---|---|",
        f"| Context recall@k | {results['vector']['context_recall_at_k']} | "
        f"{results['hybrid']['context_recall_at_k']} | {delta('context_recall_at_k')} |",
        f"| Recall@1 | {results['vector']['recall_at_1']} | "
        f"{results['hybrid']['recall_at_1']} | {delta('recall_at_1')} |",
        f"| MRR | {results['vector']['mrr']} | {results['hybrid']['mrr']} | {delta('mrr')} |",
        "",
        "_Hybrid = BM25 + vector fused with LangChain's EnsembleRetriever._",
    ]
    out = RESULTS_DIR / f"compare-{stamp}.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ragas", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Benchmark vector vs hybrid.")
    parser.add_argument("--min-recall", type=float, default=0.0,
                        help="Fail (exit 1) if context_recall@k falls below this (CI gate).")
    args = parser.parse_args()

    rows = load_golden()
    print(f"Loaded {len(rows)} questions (framework=langchain).")

    if args.compare:
        out = run_compare(rows)
        print(f"\nA/B report written to {out}")
        return

    settings = get_settings().model_copy(update={"trace_dir": EVAL_TRACE_DIR})
    pipeline = RAGPipelineLC(settings=settings, reset_index=True)
    metrics, per_q = retrieval_metrics(rows, pipeline)
    ragas = None if args.no_ragas else ragas_metrics(rows, pipeline)
    md_path = write_report(metrics, ragas, per_q)

    print("\n=== Retrieval metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    if ragas:
        print("\n=== Ragas metrics ===")
        for k, v in ragas.items():
            print(f"  {k}: {v}")
    print(f"\nReport written to {md_path}")

    recall = metrics["context_recall_at_k"] or 0.0
    if args.min_recall and recall < args.min_recall:
        print(f"\nFAIL: context_recall@k {recall} < required {args.min_recall}")
        sys.exit(1)


if __name__ == "__main__":
    main()
