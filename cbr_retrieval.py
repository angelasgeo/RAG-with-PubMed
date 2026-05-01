from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return matrix / norms

def lexical_overlap_score(query: str, text: str) -> float:
    query_tokens = set(_tokenize(query))
    doc_tokens = set(_tokenize(text))
    if not query_tokens or not doc_tokens:
        return 0.0
    overlap = len(query_tokens & doc_tokens)
    return overlap / math.sqrt(len(query_tokens) * len(doc_tokens))


@dataclass
class PubMedQACase:
    qid: str
    question: str
    final_decision: str
    long_answer: str
    meshes: List[str]
    contexts: List[str]

    @property
    def retrieval_text(self) -> str:
        mesh_text = " ".join(self.meshes[:8])
        ctx_preview = " ".join(self.contexts[:2])[:900]
        return f"{self.question} {mesh_text} {ctx_preview}".strip()


class PubMedQACaseRetriever:
    def __init__(self, emb_model) -> None:
        self.emb_model = emb_model
        self.cases: List[PubMedQACase] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None

    def fit(
        self,
        questions_data: Dict,
        gt_data: Optional[Dict] = None,
        excluded_ids: Optional[Iterable[str]] = None,
        max_cases: Optional[int] = None,
    ) -> "PubMedQACaseRetriever":
        excluded = {str(x) for x in (excluded_ids or [])}
        cases: List[PubMedQACase] = []

        for qid, entry in questions_data.items():
            qid = str(qid)
            if qid in excluded or not isinstance(entry, dict):
                continue

            question = str(entry.get("QUESTION", "")).strip()
            if not question:
                continue

            label = _resolve_label(qid, entry, gt_data)
            if label not in {"yes", "no", "maybe"}:
                continue

            contexts = _extract_contexts(entry)
            long_answer = str(entry.get("LONG_ANSWER", "") or "").strip()
            meshes = [str(x).strip() for x in entry.get("MESHES", []) if str(x).strip()]

            cases.append(
                PubMedQACase(
                    qid=qid,
                    question=question,
                    final_decision=label,
                    long_answer=long_answer,
                    meshes=meshes,
                    contexts=contexts,
                )
            )

            if max_cases is not None and len(cases) >= max_cases:
                break

        if not cases:
            raise ValueError("No valid PubMedQA cases were found for the case base.")

        texts = [case.retrieval_text for case in cases]
        embeddings = self.emb_model.encode(texts, convert_to_numpy=True).astype("float32")
        self.embeddings = _normalize_rows(embeddings)
        self.cases = cases
        self.index = _build_index(self.embeddings)
        return self

    def retrieve(self, query: str, k: int = 4) -> List[Dict]:
        if self.index is None or self.embeddings is None:
            raise ValueError("Case retriever is not fitted yet.")

        query_emb = self.emb_model.encode([query], convert_to_numpy=True).astype("float32")
        query_emb = _normalize_rows(query_emb)

        scores, indices = self.index.search(query_emb, min(k, len(self.cases)))
        results: List[Dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            case = self.cases[int(idx)]
            lexical = lexical_overlap_score(query, case.retrieval_text)
            final_score = 0.85 * float(score) + 0.15 * lexical
            results.append(
                {
                    "qid": case.qid,
                    "question": case.question,
                    "final_decision": case.final_decision,
                    "long_answer": case.long_answer,
                    "meshes": case.meshes,
                    "contexts": case.contexts,
                    "score": final_score,
                    "semantic_score": float(score),
                    "lexical_score": lexical,
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:k]


def _build_index(embeddings: np.ndarray):
    if faiss is not None:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index

    class NumpyIndex:
        def __init__(self, matrix: np.ndarray):
            self.matrix = matrix

        def search(self, queries: np.ndarray, top_k: int):
            sims = queries @ self.matrix.T
            order = np.argsort(-sims, axis=1)[:, :top_k]
            sorted_scores = np.take_along_axis(sims, order, axis=1)
            return sorted_scores, order

    return NumpyIndex(embeddings)


def _resolve_label(qid: str, entry: Dict, gt_data: Optional[Dict]) -> str:
    direct = str(entry.get("final_decision", "") or entry.get("FINAL_DECISION", "")).strip().lower()
    if direct in {"yes", "no", "maybe"}:
        return direct
    if gt_data is None or qid not in gt_data:
        return ""

    gt_entry = gt_data[qid]
    if isinstance(gt_entry, dict):
        return str(gt_entry.get("final_decision", "")).strip().lower()
    return str(gt_entry).strip().lower()


def _extract_contexts(entry: Dict) -> List[str]:
    context_payload = entry.get("CONTEXTS", [])
    if isinstance(context_payload, dict):
        values = context_payload.get("contexts") or context_payload.get("abstracts") or []
    else:
        values = context_payload or []

    if not isinstance(values, Sequence) or isinstance(values, str):
        values = [values]

    return [str(x).strip() for x in values if str(x).strip()]
