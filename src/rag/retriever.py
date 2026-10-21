"""Paper retrieval for the RAG parser.

Loads the bundled polymer-paper corpus (`papers.jsonl`), embeds each paper, and
returns the top-k most relevant papers for a query. Uses sentence-transformer
embeddings when available (semantic match) and falls back to TF-IDF so the
retriever always works, even offline without the embedding model.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CORPUS_FILE = Path(__file__).resolve().parent / "papers.jsonl"
_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_corpus(path=CORPUS_FILE):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _text(paper):
    return f"{paper['title']}. {paper['abstract']}"


class Retriever:
    def __init__(self, corpus=None):
        self.corpus = corpus or load_corpus()
        self.texts = [_text(p) for p in self.corpus]
        self.backend, self._model, self._emb, self._vec = self._build()

    def _build(self):
        # preferred: semantic sentence-transformer embeddings
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(_EMBED_MODEL)
            emb = model.encode(self.texts, normalize_embeddings=True,
                               show_progress_bar=False)
            return "sentence-transformers", model, np.asarray(emb), None
        except Exception:
            pass
        # fallback: TF-IDF (no download, always available)
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        emb = vec.fit_transform(self.texts)
        return "tfidf", None, emb, vec

    def _query_vec(self, query):
        if self.backend == "sentence-transformers":
            return np.asarray(self._model.encode([query], normalize_embeddings=True))
        return self._vec.transform([query])

    def search(self, query, k=5):
        q = self._query_vec(query)
        if self.backend == "sentence-transformers":
            sims = (self._emb @ q[0])
        else:
            from sklearn.metrics.pairwise import linear_kernel
            sims = linear_kernel(q, self._emb)[0]
        order = np.argsort(sims)[::-1][:k]
        out = []
        for i in order:
            p = dict(self.corpus[int(i)])
            p["score"] = round(float(sims[int(i)]), 4)
            out.append(p)
        return out


_RETRIEVER = None


def retrieve(query, k=5):
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever()
    return _RETRIEVER.search(query, k)


if __name__ == "__main__":
    import sys as _s
    q = " ".join(_s.argv[1:]) or "high Tg, low density"
    r = Retriever()
    print(f"backend: {r.backend}  |  corpus: {len(r.corpus)} papers\n")
    for p in r.search(q, 5):
        print(f"[{p['score']:+.3f}] ({p['prop']}) {p['title']} — {p['authors']} {p['year']}")
