# Learning to Rank What Matters

> A semantic approach to citation relevance - re-ranking a paper's bibliography by how closely each cited work relates to the paper itself.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Grade](https://img.shields.io/badge/MSci_Grade-76%25-success)

This is the implementation of my MSci final-year project at the University of Birmingham (2024–25). The system takes a research paper (by DOI, OpenAlex ID, or title) and automatically re-orders its bibliography so that the most semantically related prior work appears at the top - rather than the arbitrary alphabetical or chronological order it usually arrives in.

The full dissertation (8,352 words) is included in this repository.

---

## The problem

Modern academic papers cite 50-100+ references each, and they're usually listed alphabetically by author surname. This means there's no signal from the ordering about *which* of those citations matter most. A reader trying to trace a paper's intellectual lineage has to inspect every entry - a task that grows linearly with reference list length and quadratically when surveying multiple papers.

This project addresses that by **automatically ranking a paper's references by semantic relevance** to the paper itself, using sentence-embedding models. The system is query-free: the only inputs are the paper's title and abstract, and the titles and abstracts of its references - exactly the information a real reader has when opening a bibliography.

---

## Results

The system was evaluated on a curated set of **10 well-known "extended-version" paper pairs**, where Paper B is a recognised follow-up or extension of Paper A (e.g. YOLOv3 extending YOLO9000, BERT building on Attention Is All You Need). The expected behaviour is that Paper A should rank near the top of Paper B's bibliography when sorted by semantic similarity.

The best-performing configuration achieved:

| Metric | Score |
|--------|-------|
| **Mean Reciprocal Rank (MRR)** | **0.83** |
| **nDCG@10** | **0.83** |
| **Success@10** (Paper A appears in top 10) | **90%** |
| **Rank 1 hits** | **8 out of 10 pairs** |

This substantially outperformed all baselines tested during development - including TF-IDF, BM25, and alternative embedding models. Full per-pair results are in the `results/` directory.

### Configuration comparison

Four model × pooling combinations were evaluated on the same benchmark:

| Configuration | Mean MRR | Mean nDCG@10 | Success@10 |
|---|---|---|---|
| **all-mpnet-base-v2 + mean pooling** ⭐ | **0.83** | **0.83** | **90%** |
| all-mpnet-base-v2 + CLS pooling | 0.72 | 0.74 | 80% |
| SciBERT + mean pooling | 0.27 | 0.30 | 50% |
| SciBERT + CLS pooling | 0.06 | 0.10 | 20% |

**Two clear findings:** sentence-transformer embeddings outperform vanilla SciBERT by a wide margin on this task, and pooling strategy matters more than the underlying architecture - mean pooling beats CLS pooling on every model tested.

---

## How it works

```
       User input (DOI / OpenAlex ID / title)
                       │
                       ▼
              OpenAlex /works API
        (fetches paper + reference metadata)
                       │
                       ▼
            Abstract reconstruction
       (inverted-index → plain text)
                       │
                       ▼
        Concatenate title + abstract
               for paper & each ref
                       │
                       ▼
        Sentence-Transformer encoding
         (all-mpnet-base-v2, mean pooling)
                       │
                       ▼
              Cosine similarity
       (paper vector ↔ each ref vector)
                       │
                       ▼
        Ranked bibliography (CLI table)
```

### Key technical decisions

- **OpenAlex** was chosen over Semantic Scholar because its API is fully open with no authentication required. Semantic Scholar's API key application process had a median turnaround of eight weeks during the project window.
- **`all-mpnet-base-v2`** (Sentence-Transformers) was selected after testing four configurations. It outperformed domain-specific SciBERT by 0.56 MRR - evidence that general-purpose sentence encoders fine-tuned with Siamese losses excel even in scientific similarity tasks.
- **Mean pooling** is the default. CLS-token pooling consistently underperformed because the [CLS] token over-represents document preambles and loses distal sentence content in longer abstracts.
- **Exponential backoff** wraps every HTTP request. OpenAlex rate-limits unauthenticated traffic to ~60 requests/minute; the `safe_get` wrapper retries 429 and 5xx responses with delays of 1s, 2s, 4s, 8s (capped at 32s). 97% of transient errors resolved on the second attempt during testing.
- **Batch reference fetching** (default 50 IDs per request) reduces what would be thousands of HTTP round-trips down to dozens.
- **In-memory caching** of fetched metadata yielded a ~2.7× speed-up on repeated evaluation runs without the complexity of on-disk serialisation.

---

## Getting started

### Requirements

- Python 3.11
- ~4 GB free RAM
- No GPU required - entire pipeline runs on CPU
- Internet connection (uses OpenAlex public API)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/G-Pawar/learning-to-rank-citations.git
cd learning-to-rank-citations

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Usage

**Run the full curated-pair evaluation** (reproduces the headline results):

```bash
python main.py
```

**Rank references for a specific paper by OpenAlex ID:**

```bash
python main.py --openalex_id "W2896457183"
```

**Use a different model:**

```bash
python main.py --model_name "allenai/scibert_scivocab_uncased" --pooling_method cls
```

### Available CLI options

| Flag | Description | Default |
|------|-------------|---------|
| `--model_name` | HuggingFace model checkpoint | `allenai/scibert_scivocab_uncased` |
| `--pooling_method` | `mean` or `cls` | `mean` |
| `--use_gpu` | Enable CUDA acceleration if available | `False` |
| `--batch_size` | Encoder batch size | `8` |
| `--chunk_size` | OpenAlex reference fetch batch size | `50` |
| `--sleep` | Seconds between API calls | `0.2` |

For best results, use:

```bash
python main.py --model_name "sentence-transformers/all-mpnet-base-v2" --pooling_method mean
```

---

## Project structure

```
learning-to-rank-citations/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── config.py           # BibliographyRankerConfig + OpenAlex base URL
│   ├── data_structures.py  # Paper dataclass (title, abstract, embedding, similarity)
│   ├── ranker.py           # BibliographyRanker - fetch, embed, rank pipeline
│   ├── evaluation.py       # 10-pair benchmark + MRR/nDCG/Success metrics
│   ├── search.py           # Interactive title-based paper search
│   └── main.py             # CLI entrypoint + orchestration
│
├── docs/
│   └── dissertation.pdf    # Full MSci report (8,352 words)
│
└── results/
    └── results_summary.md  # Detailed per-pair benchmark results
```

### Module responsibilities

| Module | Responsibility |
|--------|----------------|
| `config.py` | Central runtime parameters - model name, pooling, batch sizes, API delays |
| `data_structures.py` | The `Paper` dataclass - lightweight, holds title/abstract/embedding/similarity |
| `ranker.py` | The core pipeline - fetches from OpenAlex, reconstructs abstracts, embeds text, computes cosine similarity, returns ranked references |
| `evaluation.py` | The curated 10-pair ground-truth table plus title-based MRR, nDCG@10, and plotting utilities |
| `search.py` | Title-based search wrapper around OpenAlex `/works` - used when the user doesn't have a DOI |
| `main.py` | Argument parsing, CLI orchestration, ASCII table rendering for top-10 / bottom-5 output |

---

## The 10-pair benchmark dataset

The evaluation set covers three sub-fields of deep learning, each with paper pairs where Paper B is a known follow-up to Paper A:

| # | Paper A (precursor) | Paper B (extension) |
|---|---------------------|---------------------|
| 1 | Neural Machine Translation by Jointly Learning to Align and Translate | Effective Approaches to Attention-based NMT |
| 2 | Distributed Representations of Words and Phrases | Distributed Representations of Sentences and Documents |
| 3 | Playing Atari with Deep RL | Rainbow: Combining Improvements in Deep RL |
| 4 | You Only Look Once | YOLO9000: Better, Faster, Stronger |
| 5 | YOLO9000 | YOLOv3: An Incremental Improvement |
| 6 | Rich Feature Hierarchies (R-CNN) | Fast R-CNN |
| 7 | Fast R-CNN | Faster R-CNN |
| 8 | Faster R-CNN | Mask R-CNN |
| 9 | Deep Residual Learning (ResNet) | Identity Mappings in Deep Residual Networks |
| 10 | Attention Is All You Need | BERT |

This dataset is released under the same MIT licence as the code, providing a lightweight benchmark for future citation-ranking research.

---

## Limitations

The dissertation is transparent about three constraints that bound the work:

- **Title + abstract only.** The system ignores signal from full paper text, section headings, methodology, and results. References whose abstracts are generic but whose full text reveals strong overlap can be mis-ranked.
- **Single ground-truth assumption.** The benchmark assumes Paper A is the *single* most relevant reference for Paper B. This is intuitive for extension pairs but may not hold for more complex citation relationships.
- **Sample size.** Ten pairs is enough to surface clear architecture-level trends but too small for cross-disciplinary significance testing.

---

## Future work

The dissertation identifies four concrete extensions:

1. **Fine-tuning on scholarly data** - a margin-based triplet loss on millions of citing–cited pairs could tailor embeddings to academic discourse and bridge the synonym-drift failures observed in P2/P3.
2. **Multilingual support** - initial tests on German abstracts revealed tokenisation artefacts; sub-word vocabularies and multilingual checkpoints warrant systematic study.
3. **Web interface with caching** - replacing the CLI with a Flask/React dashboard backed by a Redis vector cache would cut perceived latency below 500 ms.
4. **Full-text ingestion** - as more publishers open XML archives, extending the embedding to include introduction and methods sections could further sharpen ranking fidelity.

---

## About this project

This was my MSci final-year project at the **University of Birmingham, School of Computer Science** (2024–25), supervised by Ben McCanna. The full 8,352-word dissertation is included in [`docs/dissertation.pdf`](docs/dissertation.pdf) and covers the literature review, methodology evolution, implementation decisions, full results, and discussion.

The project demonstrates that semantic citation re-ranking is both technically tractable on commodity hardware and empirically valuable for literature navigation.

---

## Contact

**Gurpreet Pawar**
📧 pawar_gg@outlook.com
💼 [linkedin.com/in/gurpreet811](https://linkedin.com/in/gurpreet811)

For questions, feedback, or research collaboration - feel free to open an issue or reach out directly.

---

## Licence

MIT - see [LICENSE](LICENSE) for details.

The curated 10-pair benchmark dataset in `src/evaluation.py` is released under the same licence.
