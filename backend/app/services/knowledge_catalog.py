from __future__ import annotations

import re
import json
from pathlib import Path
from uuid import uuid4
from backend.app.config import get_config
from backend.app.services.rag.retriever import LocalRagRetriever


DOCUMENT_DIR = Path(__file__).resolve().parents[1] / "knowledge" / "rag_documents"
DOMAIN_CATALOG = DOCUMENT_DIR.parent / "domain_catalog.json"
UPLOAD_ROOT = Path(__file__).resolve().parents[3] / "data" / "knowledge_uploads"
REFRESHED_AT = "2026-11-02T09:00:00+08:00"


def _source_name(value: str) -> tuple[str, str]:
    lowered = value.lower()
    if "aad.org" in lowered:
        return "American Academy of Dermatology", "AAD"
    if "dermnetnz" in lowered:
        return "DermNet New Zealand Trust", "DermNet"
    if "cdc.gov" in lowered:
        return "Centers for Disease Control and Prevention", "CDC"
    if "nice.org" in lowered:
        return "NICE", "NICE"
    return "Xinfu Clinical Rules", "项目资料"


def _parse_document(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    metadata: dict[str, object] = {}
    content = raw
    if raw.startswith("---"):
        _, frontmatter, content = raw.split("---", 2)
        for line in frontmatter.strip().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip()
            if key.strip() in {"symptoms", "conditions", "aliases", "tags"}:
                metadata[key.strip()] = [item.strip() for item in value.split(",") if item.strip()]
            else:
                metadata[key.strip()] = value
    source, short_source = _source_name(str(metadata.get("source", "")))
    title = str(metadata.get("title", path.stem))
    blocks: list[dict[str, str]] = []
    current = str(metadata.get("section", title))
    buffer: list[str] = []
    for line in content.strip().splitlines():
        heading = re.match(r"^#{1,4}\s+(.+)$", line.strip())
        if heading:
            if buffer:
                blocks.append({"section": current, "content": "\n".join(buffer).strip()})
                buffer = []
            current = heading.group(1).strip()
        elif not line.strip():
            if buffer:
                blocks.append({"section": current, "content": "\n".join(buffer).strip()})
                buffer = []
        else:
            buffer.append(line)
    if buffer:
        blocks.append({"section": current, "content": "\n".join(buffer).strip()})
    if not blocks:
        blocks = [{"section": current, "content": content.strip()}]
    document_id = str(metadata.get("document_id", path.stem))
    return {
        "document_id": document_id,
        "title": title,
        "source": short_source,
        "source_name": source,
        "source_url": str(metadata.get("source", "")),
        "year": str(metadata.get("year", "未标注")),
        "risk": str(metadata.get("risk", "未标注")),
        "chunk_count": len(blocks),
        "tags": list(metadata.get("tags", [])),
        "conditions": list(metadata.get("conditions", [])),
        "updated_at": REFRESHED_AT,
        "filename": path.name,
        "front_matter": metadata,
        "sections": [block["section"] for block in blocks],
        "chunks": [
            {"index": index, "id": f"{document_id}#C{index:02d}" if len(blocks) > 1 else document_id,
             "section": block["section"], "content": block["content"]}
            for index, block in enumerate(blocks, start=1)
        ],
    }


def _all_items() -> list[dict]:
    retriever = LocalRagRetriever(document_dir=DOCUMENT_DIR)
    result = []
    for path in sorted(DOCUMENT_DIR.glob("*.md")):
        item = _parse_document(path)
        chunks = [chunk for chunk in retriever.documents if chunk["file_key"] == path.name]
        if chunks:
            item["document_id"] = chunks[0]["document_id"]
            item["chunks"] = [{"index": index, "id": chunk["id"], "section": chunk["section"],
                               "content": chunk["content"]} for index, chunk in enumerate(chunks, 1)]
        result.append(item)
    return result


def overview() -> dict:
    items = _all_items()
    domains = json.loads(DOMAIN_CATALOG.read_text(encoding="utf-8"))
    pending_uploads = list(UPLOAD_ROOT.iterdir()) if UPLOAD_ROOT.exists() else []
    sources: dict[str, dict[str, object]] = {}
    risks: dict[str, int] = {}
    for item in items:
        source = str(item["source"])
        bucket = sources.setdefault(source, {
            "name": item["source_name"], "short_name": source, "documents": 0, "chunks": 0,
        })
        bucket["documents"] = int(bucket["documents"]) + 1
        bucket["chunks"] = int(bucket["chunks"]) + int(item["chunk_count"])
        risk = str(item["risk"])
        risks[risk] = risks.get(risk, 0) + 1
    return {
        "data_source": "filesystem_snapshot",
        "version": "xinfu-knowledge-v1",
        "document_count": len(items),
        "chunk_count": sum(int(item["chunk_count"]) for item in items),
        "domain_distribution": domains,
        "domain_total_chunks": sum(item["chunks"] for item in domains),
        "domain_count": len(domains),
        "source_count": len(sources),
        "sources": list(sources.values()),
        "risk_distribution": risks,
        "pending_upload_count": len([path for path in pending_uploads if path.is_file()]),
        "pending_upload_bytes": sum(path.stat().st_size for path in pending_uploads if path.is_file()),
        "retrieval": {"backend": "Agentic RAG · BM25 + IlhaEmbed / RRF", "top_k": get_config().rag.candidate_k,
                      "assessment": "LLM评估及引用校验", "max_tool_calls": get_config().rag.max_tool_calls,
                      "scope": "configured_capability_not_live_health"},
        "last_refresh": REFRESHED_AT,
    }


def items(source: str = "", risk: str = "", query: str = "") -> list[dict]:
    needle = query.strip().lower()
    result = []
    for item in _all_items():
        haystack = " ".join([str(item["title"]), *item["tags"], *item["conditions"], *[chunk["content"] for chunk in item["chunks"]]]).lower()
        if source and item["source"] != source:
            continue
        if risk and item["risk"] != risk:
            continue
        if needle and needle not in haystack:
            continue
        result.append({key: value for key, value in item.items() if key not in {"front_matter", "chunks", "sections", "conditions"}})
    return result


def item(document_id: str) -> dict | None:
    return next((value for value in _all_items() if value["document_id"] == document_id), None)


def save_upload(filename: str, content: bytes) -> dict:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    upload_id = f"KB_UPLOAD_{uuid4().hex[:10]}"
    safe_name = Path(filename or "upload.bin").name
    (UPLOAD_ROOT / f"{upload_id}_{safe_name}").write_bytes(content)
    return {"upload_id": upload_id, "filename": safe_name, "size_bytes": len(content),
            "status": "PENDING_REVIEW", "data_source": "upload_queue"}
