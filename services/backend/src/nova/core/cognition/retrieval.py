"""NOVA Layer 4 Local Hybrid RAG & Retrieval Engine (Phase 9 / ADR-001 / ADR-006 / Phase 15 / Section 16.2).

Real-Life Analogy:
Think of `HybridRetrievalEngine` as the **Office Librarian** of NOVA.
When you have hundreds of PDFs, notes, and code files on your laptop, the `Research` agent
cannot read every single file from start to finish on every question. Instead, it asks the
Librarian (`rag.retrieve`), who does 4 jobs:

1. **Chops Long Documents into Numbered Index Cards (`chunk_document` -> `DocumentChunk`)**:
   - Splits a long document into bite-sized chunks (preserving paragraph boundaries where possible)
     and stamps every card with a unique citation ID (`chunk_id`, like `"doc_oct_inv#chunk_0"`).
   - Blocks `CREDENTIALS_SECRETS` (passwords/API keys) from ever being indexed (`Section 16.2`)
     and escapes any `</untrusted_data>` prompt-injection tags (`Phase 15 Layer 1`).

2. **Searches Using TWO Brains at Once ("Hybrid Search" — Phase 9, p. 40)**:
   - **Brain #1: Exact Keyword / Serial-Number Search (`BM25`)**:
     Finds exact matches like invoice number `"#INV-2026-10"`, error codes, or function names.
   - **Brain #2: Meaning / Concept Search (`EmbeddingPort` + Vector Cosine Similarity)**:
     Finds chunks that mean the same thing even if they use different words (e.g., matching
     *"automobile maintenance"* to *"car repair bill"*).

3. **Double-Checks the Top Candidates (`RerankerPort` — Cross-Encoder Reranking)**:
   - Takes the top candidates from Hybrid Search and carefully re-grades each one against your
     exact question so the #1 result is placed at the very top.

4. **Hands Back Safe, Citable Evidence (`RetrievalResponse.to_untrusted_data_items()`)**:
   - Wraps each winning chunk into a safe `UntrustedDataItem` whose `item_id` is the exact
     `chunk_id`, so the `Research` agent can cite it in `FinalAnswerProposal.cited_source_ids`.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    PYDANTIC_V2 = False

from nova.core.domain.context import (
    DataGovernanceError,
    UntrustedDataItem,
    UntrustedSourceType,
    sanitize_untrusted_text,
)
from nova.core.domain.primitives import (
    CURRENT_SCHEMA_VERSION,
    DataClassification,
    validate_schema_version,
)

# Keeps alphanumeric words, invoice/issue codes (#inv-2026-10), and identifiers together
_RAG_TOKEN_RE = re.compile(r"[#a-z0-9_.-]+")


def tokenize_for_bm25(text: str) -> Tuple[str, ...]:
    """Tokenize text into normalized lowercase terms for BM25 lexical indexing."""
    raw_tokens = _RAG_TOKEN_RE.findall(text.lower())
    cleaned = [t.strip(".-_") for t in raw_tokens if t.strip(".-_")]
    return tuple(cleaned)


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity in [0.0, 1.0] between two dense vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


class DocumentChunk(BaseModel):
    """Immutable indexed chunk of a local document with first-class citation metadata (Phase 9)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION)
    chunk_id: str = Field(
        ...,
        min_length=1,
        description="Unique citation ID (e.g., 'doc_inv_oct#chunk_0') used in cited_source_ids",
    )
    document_id: str = Field(..., min_length=1)
    source_uri: str = Field(..., min_length=1, description="Local file URI or path")
    source_type: UntrustedSourceType = Field(default="local_file")
    chunk_index: int = Field(..., ge=0)
    content: str = Field(..., min_length=1, description="Delimiter-sanitized chunk text")
    data_classification: DataClassification = Field(
        default=DataClassification.USER_CONTENT_LOCAL,
    )
    embedding: Tuple[float, ...] = Field(
        default_factory=tuple,
        description="Local dense vector embedding (Section 12.1 BGE/E5 adapter)",
    )

    def __init__(self, **data: object) -> None:
        version = data.get("schema_version", CURRENT_SCHEMA_VERSION)
        if isinstance(version, str):
            validate_schema_version(version)

        raw_cls = data.get("data_classification", DataClassification.USER_CONTENT_LOCAL)
        if (
            raw_cls == DataClassification.CREDENTIALS_SECRETS
            or raw_cls == "credentials_secrets"
        ):
            raise DataGovernanceError(
                "Section 16.2 violation: CREDENTIALS_SECRETS must never be indexed in RAG."
            )

        raw_content = data.get("content")
        if isinstance(raw_content, str):
            data = dict(data)
            data["content"] = sanitize_untrusted_text(raw_content)

        super().__init__(**data)

    def to_untrusted_data_item(self) -> UntrustedDataItem:
        """Convert this chunk into an isolated `<untrusted_data>` item for an AgentContext."""
        return UntrustedDataItem(
            item_id=self.chunk_id,
            source_type=self.source_type,
            source_uri=self.source_uri,
            data_classification=self.data_classification,
            content=self.content,
        )


class ScoredChunkHit(BaseModel):
    """Single ranked chunk returned by NOVA's Hybrid + Rerank RAG pipeline (Phase 9)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    chunk: DocumentChunk = Field(...)
    final_score: float = Field(..., ge=0.0, le=1.0)
    bm25_score: float = Field(..., ge=0.0)
    vector_score: float = Field(..., ge=0.0, le=1.0)
    hybrid_score: float = Field(..., ge=0.0, le=1.0)
    rerank_score: float = Field(..., ge=0.0, le=1.0)


class RetrievalResponse(BaseModel):
    """Complete result of a `rag.retrieve` query with ready-to-cite chunks (Phase 9)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    query: str = Field(...)
    hits: Tuple[ScoredChunkHit, ...] = Field(default_factory=tuple)

    @property
    def citation_ids(self) -> Tuple[str, ...]:
        """Return the ordered `chunk_id`s of all retrieved hits for citation verification."""
        return tuple(h.chunk.chunk_id for h in self.hits)

    def to_untrusted_data_items(self) -> Tuple[UntrustedDataItem, ...]:
        """Convert all retrieved hits into delimited `UntrustedDataItem`s for `AgentContext`."""
        return tuple(h.chunk.to_untrusted_data_item() for h in self.hits)

    def format_observation_summary(self) -> str:
        """Render a human- and agent-readable tool observation listing each chunk and its citation ID."""
        if not self.hits:
            return f"No matching document chunks found for query: {self.query!r}."
        lines: List[str] = []
        for idx, hit in enumerate(self.hits, start=1):
            lines.append(
                f"[{idx}] Citation ID: {hit.chunk.chunk_id} "
                f"(source={hit.chunk.source_uri}, score={hit.final_score:.4f}):\n"
                f"{hit.chunk.content}"
            )
        return "\n\n".join(lines)


@runtime_checkable
class EmbeddingPort(Protocol):
    """Abstract interface for local sentence embeddings (Section 12.1 / ADR-001 / ADR-006)."""

    def embed_text(self, text: str) -> Tuple[float, ...]:
        """Compute a dense vector embedding for `text` locally."""


@runtime_checkable
class RerankerPort(Protocol):
    """Abstract interface for local cross-encoder reranking (Section 12.1 / Phase 9 / ADR-006)."""

    def score_pair(self, query: str, chunk_text: str) -> float:
        """Return a relevance score in [0.0, 1.0] for `(query, chunk_text)`."""


class DeterministicLocalEmbeddingAdapter:
    """Offline, zero-dependency local embedding adapter with optional concept-synonym mapping.

    In production (Phase 10), a BGE/E5 local ONNX/GGUF model implements `EmbeddingPort`.
    For fast, 100% deterministic offline unit & integration tests, this adapter maps words
    (and optional synonym groups, such as `'car' <-> 'automobile'`) into a fixed-dimension
    L2-normalized vector.
    """

    def __init__(
        self,
        dimensions: int = 32,
        synonym_groups: Sequence[Sequence[str]] = (),
    ) -> None:
        if dimensions < 4:
            raise ValueError("dimensions must be >= 4.")
        self._dims = dimensions
        self._canonical_map: Dict[str, str] = {}
        for group in synonym_groups:
            if group:
                canonical = group[0].lower()
                for word in group:
                    self._canonical_map[word.lower()] = canonical

    def embed_text(self, text: str) -> Tuple[float, ...]:
        tokens = tokenize_for_bm25(text)
        vec = [0.0] * self._dims
        if not tokens:
            return tuple(vec)

        for tok in tokens:
            canonical = self._canonical_map.get(tok, tok)
            # Deterministic hash bucket projection
            h = sum((i + 1) * ord(ch) for i, ch in enumerate(canonical))
            bucket = h % self._dims
            vec[bucket] += 1.0

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [round(v / norm, 6) for v in vec]
        return tuple(vec)


class DefaultLocalCrossEncoderReranker:
    """Deterministic cross-encoder-style reranker scoring exact phrase, token coverage, and density."""

    def score_pair(self, query: str, chunk_text: str) -> float:
        q_clean = query.strip().lower()
        c_clean = chunk_text.strip().lower()
        if not q_clean or not c_clean:
            return 0.0

        q_tokens = tokenize_for_bm25(q_clean)
        c_tokens = tokenize_for_bm25(c_clean)
        if not q_tokens or not c_tokens:
            return 0.0

        q_set = set(q_tokens)
        c_set = set(c_tokens)
        coverage = len(q_set & c_set) / len(q_set)

        # Boost when exact multi-word phrase or exact code/number appears verbatim in chunk
        exact_substring_boost = 0.25 if q_clean in c_clean else 0.0
        # Boost when high-precision tokens (containing digits or '#') match
        code_tokens = {t for t in q_set if any(ch.isdigit() or ch == "#" for ch in t)}
        code_boost = (
            0.20 * (len(code_tokens & c_set) / len(code_tokens)) if code_tokens else 0.0
        )

        raw = (0.65 * coverage) + exact_substring_boost + code_boost
        return round(min(1.0, max(0.0, raw)), 4)


def chunk_document(
    *,
    document_id: str,
    source_uri: str,
    content: str,
    max_chars: int = 400,
    overlap_chars: int = 60,
    source_type: UntrustedSourceType = "local_file",
    data_classification: DataClassification = DataClassification.USER_CONTENT_LOCAL,
    embedder: Optional[EmbeddingPort] = None,
) -> Tuple[DocumentChunk, ...]:
    """Split a document into structure-aware overlapping `DocumentChunk`s (Phase 9, p. 39).

    Preserves paragraph boundaries (`\\n\\n`) whenever paragraphs fit within `max_chars`,
    and falls back to sliding character windows with `overlap_chars` for long paragraphs.
    """
    if max_chars < 50:
        raise ValueError("max_chars must be >= 50.")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be in [0, max_chars).")

    cleaned = sanitize_untrusted_text(content).strip()
    if not cleaned:
        raise ValueError("Cannot chunk an empty document.")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    raw_chunks: List[str] = []
    current_buf = ""

    for para in paragraphs:
        if len(para) > max_chars:
            if current_buf:
                raw_chunks.append(current_buf)
                current_buf = ""
            step = max(1, max_chars - overlap_chars)
            for start in range(0, len(para), step):
                piece = para[start : start + max_chars].strip()
                if piece:
                    raw_chunks.append(piece)
                if start + max_chars >= len(para):
                    break
        else:
            candidate = f"{current_buf}\n\n{para}" if current_buf else para
            if len(candidate) <= max_chars:
                current_buf = candidate
            else:
                raw_chunks.append(current_buf)
                current_buf = para

    if current_buf:
        raw_chunks.append(current_buf)

    chunks: List[DocumentChunk] = []
    for idx, text_piece in enumerate(raw_chunks):
        emb = embedder.embed_text(text_piece) if embedder is not None else ()
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document_id}#chunk_{idx}",
                document_id=document_id,
                source_uri=source_uri,
                source_type=source_type,
                chunk_index=idx,
                content=text_piece,
                data_classification=data_classification,
                embedding=emb,
            )
        )
    return tuple(chunks)


class SqliteHybridChunkIndex:
    """Durable SQLite store for indexed `DocumentChunk`s and their local vector embeddings."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        if str(self._db_path) != ":memory:":
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    data_classification TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    schema_version TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_rag_chunks_doc
                ON rag_document_chunks (document_id, chunk_index ASC);
                """
            )

    def upsert_chunks(self, chunks: Sequence[DocumentChunk]) -> int:
        if not chunks:
            return 0
        with self._connect() as conn:
            for ch in chunks:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO rag_document_chunks (
                        chunk_id,
                        document_id,
                        source_uri,
                        source_type,
                        chunk_index,
                        content,
                        data_classification,
                        embedding_json,
                        schema_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        ch.chunk_id,
                        ch.document_id,
                        ch.source_uri,
                        ch.source_type,
                        ch.chunk_index,
                        ch.content,
                        ch.data_classification.value,
                        json.dumps(list(ch.embedding)),
                        ch.schema_version,
                    ),
                )
        return len(chunks)

    def load_all_chunks(self) -> Tuple[DocumentChunk, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, document_id, source_uri, source_type, chunk_index,
                       content, data_classification, embedding_json, schema_version
                FROM rag_document_chunks
                ORDER BY document_id ASC, chunk_index ASC;
                """
            ).fetchall()
        return tuple(
            DocumentChunk(
                chunk_id=str(r[0]),
                document_id=str(r[1]),
                source_uri=str(r[2]),
                source_type=str(r[3]),  # type: ignore[arg-type]
                chunk_index=int(r[4]),
                content=str(r[5]),
                data_classification=DataClassification(str(r[6])),
                embedding=tuple(float(x) for x in json.loads(str(r[7]))),
                schema_version=str(r[8]),
            )
            for r in rows
        )


class HybridRetrievalEngine:
    """Layer 4 Hybrid (BM25 + Dense Vector) + Cross-Encoder Reranking RAG Engine (Phase 9)."""

    def __init__(
        self,
        index: SqliteHybridChunkIndex,
        embedder: Optional[EmbeddingPort] = None,
        reranker: Optional[RerankerPort] = None,
        vector_weight: float = 0.5,
        rerank_weight: float = 0.5,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        if not (0.0 <= vector_weight <= 1.0):
            raise ValueError("vector_weight must be in [0.0, 1.0].")
        if not (0.0 <= rerank_weight <= 1.0):
            raise ValueError("rerank_weight must be in [0.0, 1.0].")
        self._index = index
        self._embedder = embedder or DeterministicLocalEmbeddingAdapter()
        self._reranker = reranker or DefaultLocalCrossEncoderReranker()
        self._vector_weight = vector_weight
        self._rerank_weight = rerank_weight
        self._k1 = bm25_k1
        self._b = bm25_b

    def ingest_document(
        self,
        *,
        document_id: str,
        source_uri: str,
        content: str,
        max_chars: int = 400,
        overlap_chars: int = 60,
        source_type: UntrustedSourceType = "local_file",
        data_classification: DataClassification = DataClassification.USER_CONTENT_LOCAL,
    ) -> Tuple[DocumentChunk, ...]:
        """Chunk, embed locally, and persist a document into the hybrid index."""
        chunks = chunk_document(
            document_id=document_id,
            source_uri=source_uri,
            content=content,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            source_type=source_type,
            data_classification=data_classification,
            embedder=self._embedder,
        )
        self._index.upsert_chunks(chunks)
        return chunks

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        candidate_pool_size: int = 10,
    ) -> RetrievalResponse:
        """Execute Hybrid (BM25 + Vector) search followed by Cross-Encoder Reranking (Phase 9)."""
        all_chunks = self._index.load_all_chunks()
        if not all_chunks or not query.strip():
            return RetrievalResponse(query=query, hits=())

        # 1. Compute Okapi BM25 scores across the corpus
        bm25_raw = self._compute_bm25_scores(query, all_chunks)
        max_bm25 = max(bm25_raw) if bm25_raw else 0.0
        bm25_norm = [
            (score / max_bm25) if max_bm25 > 0.0 else 0.0 for score in bm25_raw
        ]

        # 2. Compute dense vector cosine similarity scores
        query_emb = self._embedder.embed_text(query)
        vec_scores = [
            _cosine_similarity(query_emb, ch.embedding) if ch.embedding else 0.0
            for ch in all_chunks
        ]

        # 3. Combine into initial Hybrid score and select top `candidate_pool_size`
        candidates: List[Tuple[int, float, float, float]] = []
        for idx, ch in enumerate(all_chunks):
            b_norm = bm25_norm[idx]
            v_score = vec_scores[idx]
            if b_norm == 0.0 and v_score == 0.0:
                continue
            hybrid = (
                (1.0 - self._vector_weight) * b_norm
                + self._vector_weight * v_score
            )
            candidates.append((idx, bm25_raw[idx], v_score, hybrid))

        candidates.sort(key=lambda item: item[3], reverse=True)
        shortlist = candidates[: max(top_k, candidate_pool_size)]

        # 4. Cross-Encoder Reranking pass over the shortlist (Phase 9, p. 40)
        hits: List[ScoredChunkHit] = []
        for idx, b_raw, v_score, hybrid in shortlist:
            ch = all_chunks[idx]
            r_score = self._reranker.score_pair(query, ch.content)
            final = (
                (1.0 - self._rerank_weight) * hybrid
                + self._rerank_weight * r_score
            )
            hits.append(
                ScoredChunkHit(
                    chunk=ch,
                    final_score=round(min(1.0, max(0.0, final)), 4),
                    bm25_score=round(b_raw, 4),
                    vector_score=round(v_score, 4),
                    hybrid_score=round(min(1.0, max(0.0, hybrid)), 4),
                    rerank_score=round(r_score, 4),
                )
            )

        hits.sort(key=lambda h: (h.final_score, h.bm25_score), reverse=True)
        return RetrievalResponse(query=query, hits=tuple(hits[: max(1, top_k)]))

    def _compute_bm25_scores(
        self,
        query: str,
        chunks: Sequence[DocumentChunk],
    ) -> List[float]:
        """Compute standard Okapi BM25 lexical relevance scores over `chunks`."""
        q_terms = tokenize_for_bm25(query)
        if not q_terms or not chunks:
            return [0.0] * len(chunks)

        doc_tokens = [tokenize_for_bm25(ch.content) for ch in chunks]
        doc_lens = [len(toks) for toks in doc_tokens]
        avg_dl = sum(doc_lens) / len(chunks) if chunks else 1.0
        if avg_dl == 0.0:
            avg_dl = 1.0

        n_docs = len(chunks)
        # Document frequency for each query term
        df: Dict[str, int] = {}
        for term in set(q_terms):
            df[term] = sum(1 for toks in doc_tokens if term in toks)

        scores: List[float] = []
        for toks, dl in zip(doc_tokens, doc_lens):
            tf_counter = Counter(toks)
            score = 0.0
            for term in q_terms:
                tf = tf_counter.get(term, 0)
                if tf == 0:
                    continue
                term_df = df.get(term, 0)
                # Standard Robertson-Spark Jones IDF with +1 smoothing so small corpora never go negative
                idf = math.log(1.0 + ((n_docs - term_df + 0.5) / (term_df + 0.5)))
                denom = tf + self._k1 * (1.0 - self._b + self._b * (dl / avg_dl))
                score += idf * ((tf * (self._k1 + 1.0)) / denom)
            scores.append(score)

        return scores