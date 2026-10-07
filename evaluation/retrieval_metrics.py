"""Retrieval metrics over a saved eval run. No LLM, no network.

Usage:
    python evaluation/retrieval_metrics.py [path_to_run_file]

With no path, uses the newest evaluation/results/run_*.json
(ignores *_retrieval.json).
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "evaluation" / "golden_dataset.json"
RESULTS_DIR = REPO_ROOT / "evaluation" / "results"


def normalize_text(text):
    """Lowercase and collapse whitespace for quote containment checks."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def normalize_idk_answer(answer):
    """Strip, lower, drop a trailing period, straighten curly apostrophes."""
    text = (answer or "").strip().lower()
    if text.endswith("."):
        text = text[:-1].strip()
    return text.replace("\u2019", "'")


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def newest_run_path():
    candidates = [
        path
        for path in RESULTS_DIR.glob("run_*.json")
        if not path.name.endswith("_retrieval.json")
    ]
    if not candidates:
        raise SystemExit(f"No run_*.json files found in {RESULTS_DIR.as_posix()}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_run_path(argv):
    if len(argv) > 1:
        path = Path(argv[1])
        if not path.is_file():
            raise SystemExit(f"Run file not found: {path}")
        return path
    return newest_run_path()


def corpus_chunk_count(dataset):
    """N for the chance baseline — from the golden file, not a hardcoded 9."""
    verified = dataset.get("verified_against") or {}
    count = verified.get("chunk_count")
    if not isinstance(count, int) or count <= 0:
        raise SystemExit(
            "golden_dataset.json is missing verified_against.chunk_count "
            "(needed for the 1/N and k/N baseline)."
        )
    return count


def evidence_covered(chunk, evidence):
    """Return True if this returned chunk covers one evidence entry."""
    id_match = (
        chunk["sectionNumber"] == evidence["section_number"]
        and chunk["chunkNumber"] == evidence["chunk_number"]
    )
    quote_match = normalize_text(evidence["quote"]) in normalize_text(
        chunk["chunk_text"]
    )
    return id_match, quote_match


def score_item(item, result):
    """Metrics for one answerable, error-free result over context_chunks."""
    evidence = item.get("evidence") or []
    chunks = result.get("context_chunks") or []
    k = len(chunks)

    covered = [False] * len(evidence)
    relevant_ranks = []
    chunk_matches = []

    for rank, chunk in enumerate(chunks, start=1):
        matched_id = False
        matched_quote = False
        for index, entry in enumerate(evidence):
            id_match, quote_match = evidence_covered(chunk, entry)
            if id_match or quote_match:
                covered[index] = True
            matched_id = matched_id or id_match
            matched_quote = matched_quote or quote_match

        if matched_id or matched_quote:
            if matched_id and matched_quote:
                matched_by = "both"
            elif matched_id:
                matched_by = "id"
            else:
                matched_by = "quote"
            relevant_ranks.append(rank)
            chunk_matches.append(
                {
                    "rank": rank,
                    "sectionNumber": chunk["sectionNumber"],
                    "chunkNumber": chunk["chunkNumber"],
                    "matched_by": matched_by,
                }
            )
        else:
            chunk_matches.append(
                {
                    "rank": rank,
                    "sectionNumber": chunk["sectionNumber"],
                    "chunkNumber": chunk["chunkNumber"],
                    "matched_by": None,
                }
            )

    hit_at_1 = bool(relevant_ranks and relevant_ranks[0] == 1)
    hit_at_k = bool(relevant_ranks)
    recall = (sum(covered) / len(evidence)) if evidence else 0.0
    reciprocal_rank = (1.0 / relevant_ranks[0]) if relevant_ranks else 0.0

    ranks_label = [
        f"{chunk['sectionNumber']}.{chunk['chunkNumber']}" for chunk in chunks
    ]
    relevant_label = [
        f"{chunk['sectionNumber']}.{chunk['chunkNumber']}"
        for chunk, match in zip(chunks, chunk_matches)
        if match["matched_by"]
    ]

    return {
        "id": item["id"],
        "k": k,
        "ranks": ranks_label,
        "relevant_ranks": relevant_label,
        "relevant_rank_numbers": relevant_ranks,
        "chunk_matches": chunk_matches,
        "hit_at_1": hit_at_1,
        "hit_at_k": hit_at_k,
        "recall": recall,
        "reciprocal_rank": reciprocal_rank,
        "evidence_total": len(evidence),
        "evidence_covered": sum(covered),
    }


def unanswerable_pass(answer):
    return normalize_idk_answer(answer) == "i don't know"


def mean(values):
    return sum(values) / len(values) if values else 0.0


def evaluate(dataset, run):
    by_id = {item["id"]: item for item in dataset["items"]}
    n_chunks = corpus_chunk_count(dataset)

    evaluated = []
    exclusions = []
    unanswerable_checks = []

    for result in run["results"]:
        item_id = result["id"]
        item = by_id.get(item_id)
        if item is None:
            exclusions.append(
                {
                    "id": item_id,
                    "reason": "missing_from_dataset",
                    "detail": "no matching golden item",
                }
            )
            continue

        if result.get("error") is not None:
            error_text = str(result["error"])
            exclusions.append(
                {
                    "id": item_id,
                    "reason": "error",
                    "detail": error_text[:100],
                }
            )
            continue

        if not item.get("answerable", result.get("answerable")):
            passed = unanswerable_pass(result.get("answer"))
            unanswerable_checks.append(
                {
                    "id": item_id,
                    "pass": passed,
                    "answer": result.get("answer"),
                }
            )
            exclusions.append(
                {
                    "id": item_id,
                    "reason": "unanswerable",
                    "detail": "no evidence; retrieval metrics do not apply",
                }
            )
            continue

        evaluated.append(score_item(item, result))

    hit1 = [row["hit_at_1"] for row in evaluated]
    hitk = [row["hit_at_k"] for row in evaluated]
    recalls = [row["recall"] for row in evaluated]
    rrs = [row["reciprocal_rank"] for row in evaluated]
    mean_k = mean([row["k"] for row in evaluated])

    totals = {
        "items_evaluated": len(evaluated),
        "items_excluded": len(exclusions),
        "mean_hit_at_1": mean(hit1),
        "mean_hit_at_k": mean(hitk),
        "mean_recall": mean(recalls),
        "mrr": mean(rrs),
        "mean_k": mean_k,
        "corpus_chunk_count": n_chunks,
        "chance_hit_at_1": (1.0 / n_chunks) if n_chunks else 0.0,
        "chance_hit_at_k": (mean_k / n_chunks) if n_chunks else 0.0,
    }

    return {
        "run_id": run.get("run_id"),
        "dataset_path": DATASET_PATH.as_posix(),
        "run_path": None,
        "items": evaluated,
        "exclusions": exclusions,
        "unanswerable_checks": unanswerable_checks,
        "totals": totals,
    }


def print_report(report):
    for row in report["items"]:
        relevant = ",".join(row["relevant_ranks"]) if row["relevant_ranks"] else "-"
        ranks = ",".join(row["ranks"]) if row["ranks"] else "-"
        print(
            f"id={row['id']} ranks=[{ranks}] relevant=[{relevant}] "
            f"hit@1={int(row['hit_at_1'])} hit@k={int(row['hit_at_k'])} "
            f"recall={row['recall']:.3f} rr={row['reciprocal_rank']:.3f}"
        )

    totals = report["totals"]
    print()
    print(f"Items evaluated: {totals['items_evaluated']}")
    print(f"Items excluded: {totals['items_excluded']}")
    for exclusion in report["exclusions"]:
        print(
            f"  exclude id={exclusion['id']} reason={exclusion['reason']}: "
            f"{exclusion['detail']}"
        )
    print(f"Mean hit@1: {totals['mean_hit_at_1']:.3f}")
    print(f"Mean hit@k: {totals['mean_hit_at_k']:.3f}")
    print(f"Mean recall: {totals['mean_recall']:.3f}")
    print(f"MRR: {totals['mrr']:.3f}")
    print(
        f"Chance baseline: 1/N={totals['chance_hit_at_1']:.3f}, "
        f"k/N={totals['chance_hit_at_k']:.3f} "
        f"(N={totals['corpus_chunk_count']}, mean k={totals['mean_k']:.3f})"
    )

    print()
    print("Unanswerable check:")
    if not report["unanswerable_checks"]:
        print("  (none)")
    for check in report["unanswerable_checks"]:
        status = "pass" if check["pass"] else "fail"
        print(f"  id={check['id']} {status}: {check['answer']!r}")


def save_report(report, run_path):
    run_id = report["run_id"] or Path(run_path).stem
    out_path = RESULTS_DIR / f"{run_id}_retrieval.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    payload["run_id"] = run_id
    payload["run_path"] = Path(run_path).as_posix()
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print()
    print(f"Wrote {out_path.as_posix()}")
    return out_path


def main(argv=None):
    argv = sys.argv if argv is None else argv
    run_path = resolve_run_path(argv)
    dataset = load_json(DATASET_PATH)
    run = load_json(run_path)
    report = evaluate(dataset, run)
    print_report(report)
    save_report(report, run_path)


if __name__ == "__main__":
    main()
