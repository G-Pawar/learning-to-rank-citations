"""
evaluation.py   ·  title‑based ground‑truth for Paper‑pairs
If Paper A is not found in Paper B’s references, the pair will be
marked  “exclude: True” in‑memory by main.py and omitted from plots.
"""

import numpy as np
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────────
#  TABLE OF PAIRS  (Paper B ID ➜ dict{ … })
#  Paper titles are written exactly as they appear in OpenAlex metadata.
# ────────────────────────────────────────────────────────────────────────────
TABLE_PAIRS_DATASET = {
    # machine‑translation
    "https://api.openalex.org/works/W1902237438": {
        "paper_b_title": "Effective Approaches to Attention-based Neural Machine Translation",
        "paper_a_title": "Neural Machine Translation by Jointly Learning to Align and Translate",
        "paper_a_id":    "https://api.openalex.org/works/W2964308564",
        "exclude": False,
    },
    # doc2vec
    "https://api.openalex.org/works/W2131744502": {
        "paper_b_title": "Distributed Representations of Sentences and Documents",
        "paper_a_title": "Distributed Representations of Words and Phrases and their Compositionality",
        "paper_a_id":    "https://api.openalex.org/works/W2153579005",
        "exclude": False,
    },
    # BERT vs AttentionIsAllYouNeed
    "https://api.openalex.org/works/W2896457183": {
        "paper_b_title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "paper_a_title": "Attention Is All You Need",
        "paper_a_id":    "https://api.openalex.org/works/W4385245566",
        "exclude": False,
    },
    # Rainbow vs Atari DQN
    "https://api.openalex.org/works/W2761873684": {
        "paper_b_title": "Rainbow: Combining Improvements in Deep Reinforcement Learning",
        "paper_a_title": "Playing Atari with Deep Reinforcement Learning",
        "paper_a_id":    "https://api.openalex.org/works/W4298857966",
        "exclude": False,
    },
    # YOLO v1  → YOLO9000
    "https://api.openalex.org/works/W2570343428": {
        "paper_b_title": "YOLO9000: Better, Faster, Stronger",
        "paper_a_title": "You Only Look Once: Unified, Real-Time Object Detection",
        "paper_a_id":    "https://api.openalex.org/works/W2963037989",
        "exclude": False,
    },
    # YOLO9000 → YOLOv3
    "https://api.openalex.org/works/W2796347433": {
        "paper_b_title": "YOLOv3: An Incremental Improvement.",
        "paper_a_title": "YOLO9000: Better, Faster, Stronger",
        "paper_a_id":    "https://api.openalex.org/works/W2570343428",
        "exclude": False,
    },
    # R‑CNN line
    "https://api.openalex.org/works/W1536680647": {
        "paper_b_title": "Fast R-CNN",
        "paper_a_title": "Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation",
        "paper_a_id":    "https://api.openalex.org/works/W2102605133",
        "exclude": False,
    },
    "https://api.openalex.org/works/W639708223": {
        "paper_b_title": "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks",
        "paper_a_title": "Fast R-CNN",
        "paper_a_id":    "https://api.openalex.org/works/W1536680647",
        "exclude": False,
    },
    "https://api.openalex.org/works/W2963150697": {
        "paper_b_title": "Mask R-CNN",
        "paper_a_title": "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks",
        "paper_a_id":    "https://api.openalex.org/works/W639708223",
        "exclude": False,
    },
    # ResNet line
    "https://api.openalex.org/works/W2302255633": {
        "paper_b_title": "Identity Mappings in Deep Residual Networks",
        "paper_a_title": "Deep Residual Learning for Image Recognition",
        "paper_a_id":    "https://api.openalex.org/works/W2194755991",
        "exclude": False,
    },
}

# ────────────────────────────────  metrics  ────────────────────────────────
def title_mrr(ranked_papers, target_title):
    target = target_title.lower()
    for i, p in enumerate(ranked_papers):
        if p.title.lower() == target:
            return 1.0 / (i + 1)
    return 0.0

def title_ndcg(ranked_papers, target_title, k=10):
    target = target_title.lower()
    for i, p in enumerate(ranked_papers[:k]):
        if p.title.lower() == target:
            return 1.0 / np.log2(i + 2)          # DCG of a single hit
    return 0.0                                   # not found → 0

def plot_evaluation_results(results):
    if not results:
        print("Nothing to plot – all pairs were excluded.")
        return
    import numpy as np, matplotlib.pyplot as plt
    lbls = [f"B{i+1}" for i in range(len(results))]
    mrr  = [v[0] for v in results.values()]
    ndcg = [v[1] for v in results.values()]

    x = np.arange(len(lbls))
    w = 0.35
    plt.figure(figsize=(10, 4))
    plt.bar(x-w/2, mrr,  w, label="MRR")
    plt.bar(x+w/2, ndcg, w, label="nDCG@10")
    plt.xticks(x, lbls, rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Title‑based evaluation (pairs that contain Paper A)")
    plt.legend()
    plt.tight_layout()
    plt.show()
