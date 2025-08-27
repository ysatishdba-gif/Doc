# ==============================================
# CODE 2 (genai-only, no batches): Context Pyramid + CUI Graph
# ==============================================
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import re
from datetime import datetime
import os

# ----------------------------
# CONFIG
# ----------------------------
PROJECT_ID = "YOUR_GCP_PROJECT"
LOCATION = "global"
BQ_CUI_TABLE = "your_project.your_dataset.cui_embeddings"  # cui STRING, embedding ARRAY<FLOAT64>

# ----------------------------
# Vertex AI Embeddings via google-genai (Vertex routing)
# ----------------------------
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", PROJECT_ID)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", LOCATION)
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

from google import genai
from google.genai.types import EmbedContentConfig
_client = genai.Client()

def gemini_embed_single(text: str, *, task_type: str = "RETRIEVAL_DOCUMENT", output_dim: Optional[int] = None) -> np.ndarray:
    """Embed ONE text using google-genai (no batch). Returns 1D np.ndarray."""
    cfg = EmbedContentConfig(task_type=task_type)
    if output_dim:
        cfg.output_dimensionality = int(output_dim)
    resp = _client.models.embed_content(
        model="gemini-embedding-001",
        contents=[text],
        config=cfg,
    )
    return np.array(resp.embeddings[0].values, dtype=np.float32)

# ----------------------------
# BigQuery CUI vectors
# ----------------------------
import pandas_gbq
def load_cui_vectors_bq(table_fqn: str = BQ_CUI_TABLE) -> Dict[str, np.ndarray]:
    df = pandas_gbq.read_gbq(f"SELECT cui, embedding FROM `{table_fqn}`", project_id=PROJECT_ID)
    return {row["cui"]: np.array(row["embedding"], dtype=np.float32).ravel() for _, row in df.iterrows()}

CUI_VECTORS = load_cui_vectors_bq(BQ_CUI_TABLE)

# UMLS linker stub
def umls_link(text: str) -> List[Dict[str, Any]]:
    return []

# ----------------------------
# DocAI parsing + context pyramid (no LLM)
# ----------------------------
DATE_REGEX = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})(?:[ ,;]+(\d{1,2}[:.]\d{2}))?\b")

def norm_date(s: str) -> Optional[str]:
    s = (s or "").strip().replace('.', ':').replace(',', ' ')
    try:
        tok0 = s.split()[0] if s.split() else ""
        for fmt in ("%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%m-%d-%y"):
            try:
                return datetime.strptime(tok0, fmt).strftime("%Y-%m-%d")
            except Exception:
                pass
        m = DATE_REGEX.search(s)
        if m:
            raw = m.group(1)
            for fmt in ("%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%m-%d-%y"):
                try:
                    return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
                except Exception:
                    pass
    except Exception:
        pass
    return None

@dataclass
class Section:
    section_id: str
    page: int
    title: str
    text_snippet: str
    emb: Optional[np.ndarray] = None
    cui: Optional[str] = None

@dataclass
class Entity:
    entity_id: str
    page: int
    text: str
    kind: str
    section_title: Optional[str]
    value_norm_date: Optional[str]
    emb: Optional[np.ndarray] = None
    cui: Optional[str] = None

def parse_docai_to_pyramid(docai: Dict):
    pages = (docai or {}).get("document", {}).get("pages", []) or []
    sections: List[Section] = []
    entities: List[Entity] = []
    headers_by_page: Dict[int, List[str]] = {}

    for p_idx, page in enumerate(pages):
        headers_by_page[p_idx] = []
        # KVs
        for ff in page.get("formFields", []) or []:
            name = ff.get("fieldName", {}).get("textAnchor", {}).get("content", "") or ff.get("fieldName", {}).get("content", "")
            value = ff.get("fieldValue", {}).get("textAnchor", {}).get("content", "") or ff.get("fieldValue", {}).get("content", "")
            kv_text = f"{name.strip()}: {value.strip()}".strip(": ")
            entities.append(Entity(
                entity_id=f"kv:{p_idx}:{len(entities)}",
                page=p_idx, text=kv_text, kind="kv",
                section_title=None, value_norm_date=norm_date(kv_text)
            ))
            if DATE_REGEX.search(kv_text or ""):
                entities.append(Entity(
                    entity_id=f"date:{p_idx}:{len(entities)}",
                    page=p_idx, text=kv_text, kind="date",
                    section_title=None, value_norm_date=norm_date(kv_text)
                ))
        # Lines → headings/sections
        current_title = None
        buf: List[str] = []
        def flush():
            nonlocal buf, current_title, sections
            if current_title and buf:
                text = " ".join(buf)[:1000]
                sections.append(Section(
                    section_id=f"sec:{len(sections)}",
                    page=p_idx, title=current_title.upper(),
                    text_snippet=text
                ))
                buf = []
        for ln in page.get("lines", []) or []:
            t = ln.get("layout", {}).get("textAnchor", {}).get("content", "") or ln.get("layout", {}).get("content", "")
            if not t:
                continue
            if t.endswith(":") or (t.isupper() and len(t) <= 40):
                flush()
                current_title = t.strip().strip(":")
                headers_by_page[p_idx].append(current_title)
            else:
                if current_title:
                    buf.append(t)
                    for m in DATE_REGEX.finditer(t or ""):
                        entities.append(Entity(
                            entity_id=f"datei:{p_idx}:{len(entities)}",
                            page=p_idx, text=t, kind="date",
                            section_title=current_title.upper(),
                            value_norm_date=norm_date(m.group(0))
                        ))
        flush()

    # Deterministic context strings (no LLM)
    doc_headers = list(dict.fromkeys([h for lst in headers_by_page.values() for h in lst]))
    doc_ctx = "[DOC] " + " | ".join(doc_headers)[:800]
    page_ctx = {p: "[PAGE] " + " | ".join(headers_by_page.get(p, []))[:400] for p in headers_by_page}
    return sections, entities, doc_ctx, page_ctx

# ----------------------------
# CUIs + context-aware embeddings (genai-only, single-call per text)
# ----------------------------
def attach_cuis_and_context_embeddings(sections: List[Section], entities: List[Entity],
                                       doc_ctx: str, page_ctx: Dict[int, str]) -> None:
    # Sections
    for s in sections:
        cand = umls_link(f"{s.title}: {s.text_snippet}")
        if cand:
            best = max(cand, key=lambda x: x.get("score", 0.0))
            s.cui = best.get("cui")
        if s.cui and s.cui in CUI_VECTORS:
            s.emb = CUI_VECTORS[s.cui]
        else:
            t = f"{doc_ctx}\n{page_ctx.get(s.page,'')}\n[SECTION] {s.title}\n[TEXT] {s.text_snippet[:600]}"
            s.emb = gemini_embed_single(t)

    # Entities
    for e in entities:
        cand = umls_link(e.text)
        if cand:
            best = max(cand, key=lambda x: x.get("score", 0.0))
            e.cui = best.get("cui")
        if e.cui and e.cui in CUI_VECTORS:
            e.emb = CUI_VECTORS[e.cui]
        else:
            t = f"{doc_ctx}\n{page_ctx.get(e.page,'')}\n[SECTION] {e.section_title or 'NA'}\n[ENTITY:{e.kind}] {e.text[:400]}"
            e.emb = gemini_embed_single(t)

# ----------------------------
# Scoring + retrieval
# ----------------------------
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    da = float(np.linalg.norm(a) + 1e-8); db = float(np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / (da * db))

def page_has_init_same_day(entities: List[Entity], day_iso: str, page: int) -> bool:
    for e in entities:
        if e.page == page and e.kind == "date" and e.value_norm_date == day_iso:
            if "initialization" in (e.text or "").lower() or "init" in (e.text or "").lower():
                return True
    return False

def same_day_score(day_iso: Optional[str], section: Section, entities_all: List[Entity]) -> float:
    if not day_iso:
        return 0.0
    for m in DATE_REGEX.finditer(section.text_snippet or ""):
        if norm_date(m.group(0)) == day_iso:
            return 1.0
    return 0.6 if page_has_init_same_day(entities_all, day_iso, section.page) else 0.0

def header_semantics(title: str) -> float:
    t = (title or "").lower()
    if "procedure" in t or "operation" in t:
        return 1.0
    if "brief history" in t or "clinical summary" in t or "history" in t:
        return 0.8
    return 0.3

def association_score(date_ent: Entity, section: Section, day_iso: str, entities_all: List[Entity]) -> Tuple[float, Dict[str,float]]:
    S_local = cosine(date_ent.emb, section.emb) if (date_ent.emb is not None and section.emb is not None) else 0.0
    T_same = same_day_score(day_iso, section, entities_all)
    H_sem  = header_semantics(section.title)
    P_page = 0.2 if date_ent.page == section.page else 0.0
    score = (0.35*S_local) + (0.30*T_same) + (0.25*H_sem) + (0.10*P_page)
    return float(score), {"S_local": S_local, "T_same": T_same, "H_sem": H_sem, "P_page": P_page}

# ----------------------------
# Build model and queries
# ----------------------------
def build_model_context(docai: Dict):
    sections, entities, doc_ctx, page_ctx = parse_docai_to_pyramid(docai)
    attach_cuis_and_context_embeddings(sections, entities, doc_ctx, page_ctx)
    # precompute section-section similarity
    sec_with_emb = [s for s in sections if s.emb is not None]
    S = {}
    for i in range(len(sec_with_emb)):
        for j in range(i+1, len(sec_with_emb)):
            sim = cosine(sec_with_emb[i].emb, sec_with_emb[j].emb)
            S[(sec_with_emb[i].section_id, sec_with_emb[j].section_id)] = sim
            S[(sec_with_emb[j].section_id, sec_with_emb[i].section_id)] = sim
    return sections, entities, doc_ctx, page_ctx, S

def query_context_cui(docai: Dict, query_date_str: str,
                      min_direct: float=0.60, min_indirect: float=0.45, decay: float=0.8):
    sections, entities, doc_ctx, page_ctx, S = build_model_context(docai)

    q_iso = norm_date(query_date_str) or query_date_str
    date_nodes = [e for e in entities if e.kind == "date" and e.value_norm_date == q_iso]

    direct = []
    for de in date_nodes:
        for s in sections:
            sc, parts = association_score(de, s, q_iso, entities)
            if sc >= min_direct:
                direct.append({
                    "section_id": s.section_id,
                    "title": s.title,
                    "score": round(sc,3),
                    "signals": {k: round(v,3) for k,v in parts.items()},
                    "cui": s.cui,
                    "evidence": s.text_snippet[:240]
                })

    sec_with_emb = [s for s in sections if s.emb is not None]
    indirect = []
    direct_by_id = {d["section_id"]: d for d in direct}
    for sid, drec in direct_by_id.items():
        for other in sec_with_emb:
            if other.section_id == sid:
                continue
            sim = S.get((sid, other.section_id), 0.0)
            ind = drec["score"] * sim * decay
            if ind >= min_indirect:
                indirect.append({
                    "via_section_id": sid,
                    "target_section_id": other.section_id,
                    "target_title": other.title,
                    "score": round(ind,3),
                    "path": [
                        {"from": "DATE", "to": sid, "score": drec["score"]},
                        {"from": sid, "to": other.section_id, "score": round(sim,3)}
                    ],
                    "evidence": other.text_snippet[:240]
                })

    return {
        "query_date": q_iso,
        "direct_associations": sorted(direct, key=lambda x: -x["score"]),
        "indirect_associations": sorted(indirect, key=lambda x: -x["score"]),
    }

# General query (concept/topic/date) — genai-only, no batches
def _embed_query_vector_context(query_text: str, doc_ctx: str, page_ctx_guess: str = "") -> np.ndarray:
    cand = umls_link(query_text)
    if cand:
        best = max(cand, key=lambda x: x.get("score", 0.0))
        cui = best.get("cui")
        if cui and cui in CUI_VECTORS:
            return CUI_VECTORS[cui]
    qtxt = f"{doc_ctx}\n{page_ctx_guess}\n[QUERY] {query_text[:800]}"
    return gemini_embed_single(qtxt)

def _is_date_like(q: str) -> Optional[str]:
    return norm_date(q)

def _nearest_sections(vec: np.ndarray, sections: List[Section], topk: int = 10) -> List[Tuple[Section, float]]:
    cands = [(s, cosine(vec, s.emb)) for s in sections if s.emb is not None]
    cands.sort(key=lambda x: -x[1])
    return cands[:topk]

def _nearest_entities(vec: np.ndarray, entities: List[Entity], topk: int = 10) -> List[Tuple[Entity, float]]:
    cands = [(e, cosine(vec, e.emb)) for e in entities if e.emb is not None]
    cands.sort(key=lambda x: -x[1])
    return cands[:topk]

def query_any_context(docai: Dict, query_text: str,
                      min_direct: float = 0.60, min_indirect: float = 0.45, decay: float = 0.8,
                      topk: int = 8):
    sections, entities, doc_ctx, page_ctx, S = build_model_context(docai)

    q_iso = _is_date_like(query_text)
    page_ctx_guess = page_ctx.get(0, "")
    q_vec = _embed_query_vector_context(query_text, doc_ctx, page_ctx_guess)

    sec_nn = _nearest_sections(q_vec, sections, topk=topk)
    ent_nn = _nearest_entities(q_vec, entities, topk=max(3, topk//2))

    direct = []
    for s, score in sec_nn:
        if score < min_direct:
            continue
        direct.append({
            "type": "section",
            "section_id": s.section_id,
            "title": s.title,
            "score": round(float(score), 3),
            "cui": s.cui,
            "evidence": s.text_snippet[:240]
        })

    for e, score in ent_nn:
        if score < min_direct:
            continue
        direct.append({
            "type": "entity",
            "entity_id": e.entity_id,
            "kind": e.kind,
            "section_title": e.section_title,
            "cui": e.cui,
            "score": round(float(score), 3),
            "evidence": e.text[:240]
        })

    if q_iso:
        date_nodes = [e for e in entities if e.kind == "date" and e.value_norm_date == q_iso]
        for de in date_nodes:
            for s in sections:
                sc, parts = association_score(de, s, q_iso, entities)
                if sc >= min_direct:
                    direct.append({
                        "type": "section",
                        "section_id": s.section_id,
                        "title": s.title,
                        "score": round(sc,3),
                        "signals": {k: round(v,3) for k,v in parts.items()},
                        "cui": s.cui,
                        "evidence": s.text_snippet[:240]
                    })

    indirect = []
    start_sections = [d["section_id"] for d in direct if d.get("type")=="section"][:3]
    for sid in start_sections:
        for other in sections:
            if other.section_id == sid:
                continue
            sim = S.get((sid, other.section_id), 0.0)
            src_score = next((d["score"] for d in direct if d.get("section_id")==sid), 0.0)
            ind = src_score * sim * decay
            if ind >= min_indirect:
                indirect.append({
                    "target_section_id": other.section_id,
                    "target_title": other.title,
                    "score": round(float(ind), 3),
                    "path": [
                        {"from": "QUERY", "to": sid, "score": round(float(src_score), 3)},
                        {"from": sid, "to": other.section_id, "score": round(float(sim), 3)}
                    ],
                    "evidence": other.text_snippet[:240]
                })

    direct.sort(key=lambda x: -x["score"])
    indirect.sort(key=lambda x: -x["score"])
    return {
        "query": query_text,
        "query_iso_date": q_iso,
        "direct": direct,
        "indirect": indirect
    }
