"""
ranker.py  –  fetch, embed & rank references for a single paper
 * Removes self‑citations (Paper B should not appear in its own reference list).
 * Safe‑concatenate when title/abstract is None.
"""

import time, requests, torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

from config import BibliographyRankerConfig, OPENALEX_BASE_URL
from data_structures import Paper


# ───────────────────────────── helpers ──────────────────────────────
def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    if not inverted_index:
        return ""
    pairs = [(i, w) for w, idxs in inverted_index.items() for i in idxs]
    pairs.sort()
    return " ".join(w for _, w in pairs)

def normalise_openalex_url(openalex_id: str) -> str:
    if "api.openalex.org/works/" in openalex_id.lower():
        return openalex_id
    if "openalex.org/" in openalex_id.lower():
        w_id = openalex_id.split("/")[-1]
        return f"{OPENALEX_BASE_URL}/{w_id}"
    return f"{OPENALEX_BASE_URL}/{openalex_id}"


# ─────────────────────────── ranker class ───────────────────────────
class BibliographyRanker:
    def __init__(self, cfg: BibliographyRankerConfig):
        self.cfg = cfg
        device_name = "cuda" if cfg.use_gpu and torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)

        if cfg.model_name.startswith("sentence-transformers/"):
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(cfg.model_name, device=str(self.device))
            self.tokenizer = None
            self.use_st = True
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
            self.model = AutoModel.from_pretrained(cfg.model_name).to(self.device)
            self.use_st = False

    # ──────────────────────── fetching helpers ──────────────────────
    def fetch_paper_by_openalex_id(self, openalex_id: str) -> Paper:
        url  = normalise_openalex_url(openalex_id)
        data = requests.get(url).json()
        return Paper(
            title       = data.get("display_name") or "",
            abstract    = reconstruct_abstract(data.get("abstract_inverted_index", {})),
            openalex_id = data.get("id", "")
        )

    def get_reference_ids(self, paper_json: dict) -> List[str]:
        return paper_json.get("referenced_works", [])

    def fetch_references_metadata(self, ids: List[str]) -> List[dict]:
        from tqdm import tqdm
        results = []
        for i in tqdm(range(0, len(ids), self.cfg.chunk_size), desc="Fetching references"):
            subset = ids[i:i+self.cfg.chunk_size]
            filter_param = "openalex_id:" + "|".join(s.split("/")[-1] for s in subset)
            params = {"filter": filter_param, "per-page": len(subset)}
            results.extend(requests.get(OPENALEX_BASE_URL, params=params).json().get("results", []))
            time.sleep(self.cfg.sleep_seconds)
        return results

    # ───────────────────────── embedding ────────────────────────────
    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        if self.use_st:
            from torch import cat
            batches = [texts[i:i+self.cfg.batch_size] for i in range(0, len(texts), self.cfg.batch_size)]
            embs = [self.model.encode(b, convert_to_tensor=True) for b in batches]
            return cat(embs, 0)

        all_emb = []
        for i in range(0, len(texts), self.cfg.batch_size):
            batch = texts[i:i+self.cfg.batch_size]
            inp   = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inp   = {k: v.to(self.device) for k, v in inp.items()}
            with torch.no_grad():
                hid = self.model(**inp).last_hidden_state
            emb = hid[:,0,:] if self.cfg.pooling_method=="cls" else hid.mean(1)
            all_emb.append(emb.cpu())
        return torch.cat(all_emb, 0)

    # ───────────────────────── main pipeline ────────────────────────
    def run_pipeline_by_openalex_id(self, openalex_id: str) -> List[Paper]:
        main_p  = self.fetch_paper_by_openalex_id(openalex_id)
        url     = normalise_openalex_url(openalex_id)
        ref_ids = self.get_reference_ids(requests.get(url).json())
        if not ref_ids:
            print("⚠  No references found.")
            return []

        # 1) download reference metadata
        refs_raw = self.fetch_references_metadata(ref_ids)

        # 2) build Paper objects (deduplicated by ID)
        refs = {}
        for r in refs_raw:
            rid  = r.get("id","").lower()
            refs[rid] = Paper(
                title       = r.get("display_name") or "",
                abstract    = reconstruct_abstract(r.get("abstract_inverted_index", {})),
                openalex_id = r.get("id","")
            )

        # 3) drop self‑citation (if any)
        self_id = main_p.openalex_id.lower()
        refs.pop(self_id, None)

        references = list(refs.values())
        if not references:
            print("⚠  After removing self‑citation, no references remain.")
            return []

        # 4) embed & rank
        def safe_concat(p: Paper) -> str:
            return f"{(p.title or '')} {(p.abstract or '')}"

        main_emb   = self.embed_texts([safe_concat(main_p)])[0].unsqueeze(0).numpy()
        ref_embs   = self.embed_texts([safe_concat(r) for r in references]).numpy()
        sims       = cosine_similarity(main_emb, ref_embs)[0]

        for s, p in zip(sims, references):
            p.similarity = float(s)

        return sorted(references, key=lambda x: x.similarity, reverse=True)
