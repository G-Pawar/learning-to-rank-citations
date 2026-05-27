"""
main.py – title‑based evaluation of curated pairs
 • Excludes pairs where Paper A’s title isn’t found in Paper B’s references
 • Shows Top‑10 and Bottom‑5 lists in a neat table
"""

import argparse
from config     import BibliographyRankerConfig
from ranker     import BibliographyRanker
from evaluation import TABLE_PAIRS_DATASET, title_mrr, title_ndcg, plot_evaluation_results


# ──────────────────────────── pretty printing ─────────────────────────────
def _make_table(rows, header):
    topbar = "╒════╤═════════╤" + "═"*80 + "╕"
    midbar = "╞════╪═════════╪" + "═"*80 + "╡"
    botbar = "╘════╧═════════╧" + "═"*80 + "╛"
    lines  = [topbar, f"│ # │  Score  │ {header}", midbar]
    lines += rows
    lines.append(botbar)
    return "\n".join(lines)

def pretty_tables(refs, top_k=10, bottom_k=5):
    top_rows = [f"│{i:2d} │ {r.similarity:7.4f} │ {r.title}" for i, r in enumerate(refs[:top_k], 1)]
    bot_rows = [f"│{i:2d} │ {r.similarity:7.4f} │ {r.title}"
                for i, r in enumerate(refs[-bottom_k:], 1)]
    return _make_table(top_rows, "Top references") + "\n\n" + _make_table(bot_rows, "Bottom references")


# ───────────────────────────── evaluation ────────────────────────────────
def evaluate(ranker):
    kept = {}   # paper_b_id → (mrr, ndcg)

    for idx, (b_id, info) in enumerate(TABLE_PAIRS_DATASET.items(), 1):
        if info.get("exclude"):
            continue
        print(f"\n\033[1mPair {idx}\033[0m ▸  Paper B: {info['paper_b_title']}")
        refs = ranker.run_pipeline_by_openalex_id(b_id)
        if not refs:
            info["exclude"] = True
            continue

        print(pretty_tables(refs, 10, 5))

        if not any(r.title.lower() == info['paper_a_title'].lower() for r in refs):
            print("⟹ Paper A title NOT present → pair excluded.")
            info["exclude"] = True
            continue

        mrr  = title_mrr(refs, info['paper_a_title'])
        ndcg = title_ndcg(refs, info['paper_a_title'])
        print(f"→  MRR {mrr:.4f}   nDCG@10 {ndcg:.4f}")
        kept[b_id] = (mrr, ndcg)

    # summary
    print("\n\033[1mSummary of included pairs\033[0m")
    for i, (bid, sc) in enumerate(kept.items(), 1):
        print(f"{i}. {TABLE_PAIRS_DATASET[bid]['paper_b_title']}  –  MRR {sc[0]:.3f}  nDCG {sc[1]:.3f}")

    plot_evaluation_results(kept)


# ─────────────────────────────── CLI ─────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Title‑based evaluation of curated pairs")
    p.add_argument("--model_name",    default="allenai/scibert_scivocab_uncased")
    p.add_argument("--pooling_method", default="mean", choices=["mean", "cls"])
    p.add_argument("--use_gpu",       action="store_true")
    p.add_argument("--batch_size",    type=int, default=8)
    p.add_argument("--chunk_size",    type=int, default=50)
    p.add_argument("--sleep",         type=float, default=0.2)
    args = p.parse_args()

    cfg = BibliographyRankerConfig(
        model_name=args.model_name, use_gpu=args.use_gpu,
        batch_size=args.batch_size, chunk_size=args.chunk_size,
        sleep_seconds=args.sleep, pooling_method=args.pooling_method
    )
    evaluate(BibliographyRanker(cfg))


if __name__ == "__main__":
    main()
