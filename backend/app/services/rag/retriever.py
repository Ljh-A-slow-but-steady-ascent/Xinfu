from __future__ import annotations

import atexit
import hashlib
import math
import os
import re
import select
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from backend.app.config import AppConfig, get_config


ROOT = Path(__file__).resolve().parents[4]
DOCUMENT_DIR = Path(__file__).resolve().parents[2] / "knowledge" / "rag_documents"
RAG_WORKER = Path(__file__).resolve().parent / "model_worker.py"
RAG_PYTHON = Path(os.getenv("DERMAGUARD_RAG_PYTHON") or Path(sys.prefix).parent / "dermfm-zero/bin/python")

QUERY_EXPANSIONS = {
    "bcc": "basal cell carcinoma 基底细胞癌", "基底细胞癌": "basal cell carcinoma bcc",
    "scc": "squamous cell carcinoma 鳞状细胞癌", "鳞状细胞癌": "squamous cell carcinoma scc",
    "黑色素瘤": "melanoma malignant melanoma", "日光性角化病": "actinic keratosis ak",
    "色素痣": "nevus melanocytic nevus mole", "脂溢性角化病": "seborrheic keratosis benign keratosis",
    "良性角化": "benign keratosis seborrheic keratosis solar lentigo lichenoid keratosis",
    "皮肤纤维瘤": "dermatofibroma benign fibrous lesion",
    "血管性病变": "vascular lesion angioma angiokeratoma pyogenic granuloma hemorrhage",
    "带状疱疹": "herpes zoster shingles", "体癣": "tinea corporis ringworm",
    "足癣": "tinea pedis athlete foot", "疥疮": "scabies",
    "特应性皮炎": "atopic dermatitis eczema",
}


def _tokens(text: str) -> list[str]:
    lowered = text.lower().replace("_", " ")
    result = re.findall(r"[a-z0-9]+", lowered)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", lowered):
        if len(sequence) <= 4:
            result.append(sequence)
        result.extend(sequence[index:index + 2] for index in range(len(sequence) - 1))
        result.extend(sequence[index:index + 3] for index in range(len(sequence) - 2))
    return result


def _expand_query(text: str) -> str:
    lowered = text.lower()
    return " ".join([
        text,
        *(value for key, value in QUERY_EXPANSIONS.items()
          if (re.search(rf"\b{re.escape(key)}\b", lowered) if key.isascii() else key in lowered)),
    ])


def build_dynamic_query(
    user_query: str = "", visual_candidates: list[str] | None = None,
    positive_symptoms: list[str] | None = None, risk_reasons: list[str] | None = None,
    duration: str | None = None, topic: str = "鉴别依据 检查依据 风险分级 转诊指南",
) -> str:
    sections = [
        user_query.strip(), "视觉候选 " + " ".join(visual_candidates or []),
        "阳性症状 " + " ".join(positive_symptoms or []),
        "风险原因 " + " ".join(risk_reasons or []),
        f"病程变化 {duration}" if duration else "", f"检索主题 {topic}" if topic else "",
    ]
    return "；".join(item for item in sections if item and item.strip())


def _parse_frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text.strip()
    _, metadata_block, content = text.split("---", 2)
    metadata: dict[str, Any] = {}
    for line in metadata_block.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    for key in ("symptoms", "conditions", "aliases", "tags"):
        metadata[key] = [item.strip() for item in str(metadata.get(key, "")).split(",") if item.strip()]
    return metadata, content.strip()


def _structured_chunks(path: Path) -> list[dict[str, Any]]:
    metadata, content = _parse_frontmatter(path)
    title = str(metadata.get("title", path.stem))
    current_heading = str(metadata.get("section", title))
    blocks: list[tuple[str, str]] = []
    buffer: list[str] = []
    for line in content.splitlines():
        heading = re.match(r"^#{1,4}\s+(.+)$", line.strip())
        if heading:
            if buffer:
                blocks.append((current_heading, "\n".join(buffer).strip())); buffer = []
            current_heading = heading.group(1).strip()
        elif not line.strip():
            if buffer:
                blocks.append((current_heading, "\n".join(buffer).strip())); buffer = []
        else:
            buffer.append(line)
    if buffer:
        blocks.append((current_heading, "\n".join(buffer).strip()))
    blocks = [(heading, text) for heading, text in blocks if text] or [(current_heading, content)]
    chunks = []
    for index, (heading, text) in enumerate(blocks, start=1):
        document_id = str(metadata.get("document_id", path.stem))
        chunks.append({
            **metadata, "id": document_id if len(blocks) == 1 else f"{document_id}#C{index:02d}", "document_id": document_id,
            "title": title, "document_name": str(metadata.get("document_name", title)),
            "organization": str(metadata.get("organization", metadata.get("source", "未知"))),
            "year": str(metadata.get("year", "未标注")), "section": heading,
            "page": str(metadata.get("page", "网页资料")), "version": str(metadata.get("version", "未标注")),
            "content": text, "chunk_index": index - 1,
            "file_key": path.name,
        })
    return chunks


class BM25Index:
    """Small in-memory BM25 index; no additional service or model is required."""

    def __init__(self, corpus: list[list[str]]) -> None:
        self.frequencies = [Counter(tokens) for tokens in corpus]
        self.lengths = [len(tokens) for tokens in corpus]
        self.average_length = sum(self.lengths) / max(1, len(corpus))
        document_frequency = Counter(token for tokens in self.frequencies for token in tokens)
        self.idf = {
            token: math.log(1 + (len(corpus) - count + 0.5) / (count + 0.5))
            for token, count in document_frequency.items()
        }

    def score(self, query: str) -> list[float]:
        query_tokens = set(_tokens(query))
        scores = []
        for frequencies, length in zip(self.frequencies, self.lengths, strict=True):
            norm = 1.2 * (0.25 + 0.75 * length / max(1, self.average_length))
            scores.append(sum(
                self.idf.get(token, 0) * frequencies[token] * 2.2 / (frequencies[token] + norm)
                for token in query_tokens if frequencies[token]
            ))
        return scores


def _canonical_source(source: str, fallback: str) -> str:
    parts = urlsplit(source)
    if parts.scheme in {"http", "https"} and parts.netloc:
        # Keep meaningful query parameters, removing only tracking and fragments.
        parameters = sorted((key, value) for key, value in parse_qsl(parts.query)
                            if not key.lower().startswith("utm_") and key.lower() not in {"fbclid", "gclid"})
        return urlunsplit(("https", parts.netloc.lower().removeprefix("www."), parts.path.rstrip("/"), urlencode(parameters), ""))
    return source.strip() or fallback


def _source_kind(source: str) -> str:
    if source.startswith("local_"):
        return "local_rule"
    hostname = (urlsplit(source).hostname or "").lower()
    trusted = ("aad.org", "nice.org.uk", "cdc.gov", "dermnetnz.org")
    if any(hostname == domain or hostname.endswith("." + domain) for domain in trusted):
        return "guideline_summary"
    return "local_document"


def _evidence_category(text: str) -> str:
    lowered = text.lower()
    groups = [
        ("冲突证据", ("冲突", "不能替代", "不支持", "局限")),
        ("转诊依据", ("转诊", "急诊", "优先", "就医")),
        ("检查依据", ("活检", "检查", "病理", "评估")),
        ("鉴别依据", ("鉴别", "边界", "颜色", "表现")),
        ("风险依据", ("风险", "恶性", "红旗")),
    ]
    return next((name for name, terms in groups if any(term in lowered for term in terms)), "信息缺口")


def _evidence_types(text: str) -> list[str]:
    groups = {
        "冲突证据": ("冲突", "不能替代", "不支持", "局限", "不能确诊", "不能判断", "不能排除"),
        "转诊依据": ("转诊", "急诊", "优先", "就医", "皮肤科"),
        "检查依据": ("活检", "检查", "病理", "评估", "皮肤镜"),
        "鉴别依据": ("鉴别", "边界", "颜色", "表现", "形状", "ABCDE"),
        "风险依据": ("风险", "恶性", "红旗", "出血", "变化"),
    }
    return [name for name, words in groups.items() if any(word.lower() in text.lower() for word in words)] or ["信息缺口"]


class RagModelClient:
    """One local embedding worker with bounded waits and a failure cooldown."""

    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.lock = threading.Lock()
        self.unavailable_until = 0.0

    def close(self) -> None:
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=3)
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()

    def _start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        self.process = subprocess.Popen(
            [str(RAG_PYTHON), str(RAG_WORKER)], cwd=str(ROOT), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        assert self.process.stdin and self.process.stdout
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)

    def request(self, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        import json
        maximum_wait = min(60.0, max(1.0, float(os.getenv("DERMAGUARD_RAG_WORKER_TIMEOUT", "20"))))
        wait = maximum_wait if timeout is None else min(maximum_wait, max(0.0, timeout))
        if wait <= 0:
            raise TimeoutError("本地嵌入工具预算已耗尽")
        deadline = time.monotonic() + wait
        # The lock covers the entire request/response pair, including the cooldown.
        # It prevents concurrent callers from starting duplicate failed workers.
        if not self.lock.acquire(timeout=wait):
            raise TimeoutError("本地嵌入队列繁忙，使用 BM25")
        try:
            if time.monotonic() < self.unavailable_until:
                raise RuntimeError("本地嵌入暂不可用，使用 BM25")
            try:
                self._start()
                assert self.process and self.process.stdin and self.process.stdout
                message = (json.dumps(payload, ensure_ascii=False) + "\n").encode()
                sent = 0
                while sent < len(message):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("本地医学嵌入模型响应超时")
                    _, writable, _ = select.select([], [self.process.stdin], [], remaining)
                    if not writable:
                        raise TimeoutError("本地医学嵌入模型响应超时")
                    try:
                        sent += os.write(self.process.stdin.fileno(), message[sent:sent + 65536])
                    except BlockingIOError:
                        continue
                response = bytearray()
                while b"\n" not in response:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("本地医学嵌入模型响应超时")
                    readable, _, _ = select.select([self.process.stdout], [], [], remaining)
                    if not readable:
                        raise TimeoutError("本地医学嵌入模型响应超时")
                    try:
                        block = os.read(self.process.stdout.fileno(), 65536)
                    except BlockingIOError:
                        continue
                    if not block:
                        raise RuntimeError("本地 RAG 模型进程未返回结果")
                    response.extend(block)
                    if len(response) > 8_000_000:
                        raise RuntimeError("本地 RAG 响应超出限制")
                result = json.loads(bytes(response).split(b"\n", 1)[0])
                if result.get("error"):
                    raise RuntimeError(str(result["error"]))
                return result
            except Exception:
                self.unavailable_until = time.monotonic() + 60.0
                self.close()
                raise
        finally:
            self.lock.release()


# Several application services use the same local corpus. Share its model process.
_MODEL_CLIENT = RagModelClient()
atexit.register(_MODEL_CLIENT.close)


class LocalRagRetriever:
    """Multi-query BM25 + medical Dense retrieval, fused with reciprocal ranks.

    Search scores describe relevance only. A caller must read and assess evidence
    before deciding whether it supports an answer.
    """

    def __init__(self, config: AppConfig | None = None, document_dir: Path = DOCUMENT_DIR) -> None:
        self.config = config or get_config()
        self.documents = [chunk for path in sorted(document_dir.glob("*.md")) for chunk in _structured_chunks(path)]
        # Preserve the first legacy document id, disambiguating reused ids across
        # distinct source pages. Chunk ids are globally unique and deterministic.
        document_sources: dict[str, str] = {}
        used_ids: set[str] = set()
        for doc in self.documents:
            original_id = str(doc["document_id"])
            source = _canonical_source(str(doc.get("source", "")), str(doc["file_key"]))
            suffix = hashlib.sha256(source.encode()).hexdigest()[:8]
            previous_source = document_sources.setdefault(original_id, source)
            if previous_source != source:
                doc["document_id"] = f"{original_id}~{suffix}"
                doc["id"] = str(doc["id"]).replace(original_id, str(doc["document_id"]), 1)
            if str(doc["id"]) in used_ids:
                file_suffix = hashlib.sha256(str(doc["file_key"]).encode()).hexdigest()[:8]
                doc["id"] = f"{doc['id']}~{file_suffix}"
            used_ids.add(str(doc["id"]))
            doc["canonical_source"] = source
            doc["source_kind"] = _source_kind(str(doc.get("source", "")))
            doc["authoritative"] = doc["source_kind"] == "guideline_summary"
        self.by_id = {str(doc["id"]): doc for doc in self.documents}
        self.searchable_documents = [self._searchable(doc) for doc in self.documents]
        self.corpus_tokens = [_tokens(text) for text in self.searchable_documents]
        self.bm25 = BM25Index(self.corpus_tokens)
        self.model_client = _MODEL_CLIENT
        # Lazy loading keeps application construction free of unbudgeted model
        # waits. The first budgeted search encodes and caches the corpus.
        self.last_backend = "BM25 + IlhaEmbed Dense / RRF（待检索）" if self.advanced_enabled else "BM25（Dense 未开启）"

    @property
    def advanced_enabled(self) -> bool:
        return os.getenv("DERMAGUARD_RAG_ADVANCED", "0").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _searchable(doc: dict[str, Any]) -> str:
        important = [str(doc.get("title", "")), str(doc.get("section", "")), *doc.get("conditions", []),
                     *doc.get("aliases", []), *doc.get("symptoms", []), *doc.get("tags", [])]
        return " ".join([*important, *important, str(doc["content"])])

    def _result(self, doc: dict[str, Any], backend: str = "local_read") -> dict[str, Any]:
        return {
            "id": str(doc["id"]), "document_id": str(doc["document_id"]), "title": str(doc["title"]),
            "content": str(doc["content"]), "source": str(doc.get("source", "")),
            "score": 0.0, "evidence_type": _evidence_category(str(doc["content"])),
            "evidence_types": _evidence_types(str(doc["content"])), "evidence_status": "待核验",
            "citation": {"document": str(doc["document_name"]), "organization": str(doc["organization"]),
                "year": str(doc["year"]), "section": str(doc["section"]), "page": str(doc["page"]),
                "version": str(doc["version"])},
            "retrieval_backend": backend, "query": "",
            "source_kind": str(doc["source_kind"]), "authoritative": bool(doc["authoritative"]),
            "canonical_source": str(doc["canonical_source"]),
            "stored_in_database": True, "source_scope": "本地知识库",
        }

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return self.search_many([query], top_k=top_k or self.config.rag.top_k)

    def search_many(self, queries: list[str], top_k: int = 12, timeout: float | None = None) -> list[dict[str, Any]]:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        queries = list(dict.fromkeys(query.strip()[:1200] for query in queries if query.strip()))[:8]
        if not queries or not self.documents:
            self.last_backend = "本地知识库（空查询或空语料）"
            return []
        limit = max(1, min(40, top_k))
        expanded = [_expand_query(query) for query in queries]
        lexical = [self.bm25.score(query) for query in expanded]
        dense: list[list[float]] | None = None
        if self.advanced_enabled:
            try:
                rows = self.model_client.request({
                    "action": "dense_queries", "queries": expanded, "documents": self.searchable_documents,
                }, timeout=None if deadline is None else max(0.0, deadline - time.monotonic()))["scores"]
                if len(rows) != len(expanded) or any(len(row) != len(self.documents) for row in rows):
                    raise ValueError("本地嵌入分数形状无效")
                dense = [[float(value) for value in row] for row in rows]
                if any(not math.isfinite(value) for row in dense for value in row):
                    raise ValueError("本地嵌入分数包含非有限值")
                backend = "BM25 + IlhaEmbed Dense / RRF"
            except Exception:
                dense = None
                backend = "BM25（Dense 不可用，已降级）"
        else:
            backend = "BM25（Dense 未开启）"
        # Compatibility only. Request results carry their own local backend;
        # another concurrent search may update this shared diagnostic field.
        self.last_backend = backend

        rrf = [0.0] * len(self.documents)
        # Bound per-channel candidate depth; do not reward zero lexical matches.
        for channel in [lexical, *([dense] if dense is not None else [])]:
            for row in channel:
                ranked = sorted(range(len(row)), key=lambda i: (-row[i], str(self.documents[i]["id"])))
                for rank, index in enumerate((i for i in ranked if row[i] > 0), start=1):
                    if rank > 24:
                        break
                    rrf[index] += 1.0 / (60 + rank)
        order = sorted(range(len(self.documents)), key=lambda i: (-rrf[i], str(self.documents[i]["id"])))
        # Return independent source pages. Reading neighbors supplies adjacent
        # sections without presenting one page as several independent sources.
        selected: list[int] = []
        seen_sources: set[str] = set()
        for index in order:
            if rrf[index] <= 0:
                continue
            source = str(self.documents[index]["canonical_source"])
            if source not in seen_sources:
                selected.append(index)
                seen_sources.add(source)
            if len(selected) >= limit:
                break
        query_tokens = set(_tokens(" ".join(expanded)))
        results = []
        for index in selected[:limit]:
            bm25 = max(row[index] for row in lexical)
            semantic = max((row[index] for row in dense), default=0.0) if dense is not None else None
            matched = sorted(query_tokens.intersection(self.corpus_tokens[index]))
            coverage = len(matched) / max(1, len(query_tokens))
            # Absolute bounded transform, never normalized by this query's best
            # result. This is a ranking hint, not a calibrated truth probability.
            lexical_relevance = (1.0 - math.exp(-bm25 / 8.0)) * math.sqrt(coverage)
            score = lexical_relevance if semantic is None else 0.35 * lexical_relevance + 0.65 * max(0, semantic)
            item = self._result(self.documents[index], backend=backend)
            item.update({
                "score": round(max(0.0, min(1.0, score)), 6),
                "retrieval_scores": {"bm25": round(bm25, 6), "dense": semantic, "rrf": round(rrf[index], 8)},
                "lexical_coverage": round(coverage, 6), "matched_terms": matched,
                "query": "；".join(expanded), "queries": queries,
            })
            results.append(item)
        return results

    def read(self, evidence_ids: list[str], include_neighbors: bool = True) -> list[dict[str, Any]]:
        requested = list(dict.fromkeys(str(item) for item in evidence_ids))[:12]
        primary_ids = set(requested)
        selected: dict[str, tuple[dict[str, Any], str]] = {}
        for evidence_id in requested:
            doc = self.by_id.get(evidence_id)
            if doc is None:
                continue
            selected[evidence_id] = (doc, evidence_id)
            if include_neighbors:
                for neighbor in self.documents:
                    if neighbor["file_key"] == doc["file_key"] and abs(neighbor["chunk_index"] - doc["chunk_index"]) == 1:
                        selected.setdefault(str(neighbor["id"]), (neighbor, evidence_id))
        results = []
        for evidence_id, (doc, requested_id) in selected.items():
            item = self._result(doc)
            item.update({"is_neighbor": evidence_id not in primary_ids, "requested_evidence_id": requested_id})
            results.append(item)
        return results
