# ============================
# CODE 1 (genai-only, no batches): CUI Graph (Local Context Only)
# ============================
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
LOCATION = "global"  # Vertex routing location; keep 'global' unless required otherwise
BQ_CUI_TABLE = "your_project.your_dataset.cui_embeddings"  # columns: cui STRING, embedding ARRAY<FLOAT64>

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
# BigQuery: load CUI embeddings (two columns only)
# ----------------------------
import pandas_gbq

def load_cui_vectors_bq(table_fqn: str = BQ_CUI_TABLE) -> Dict[str, np.ndarray]:
    df = pandas_gbq.read_gbq(f"SELECT cui, embedding FROM `{table_fqn}`", project_id=PROJECT_ID)
    out: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        out[row["cui"]] = np.array(row["embedding"], dtype=np.float32).ravel()
    return out

CUI_VECTORS = load_cui_vectors_bq(BQ_CUI_TABLE)  # {cui: vector}

# ----------------------------
# UMLS linker (stub — replace with QuickUMLS/MedCAT/scispaCy, etc.)
# ----------------------------
def umls_link(text: str) -> List[Dict[str, Any]]:
    # Example expected return: [{"cui": "C0001449", "score": 0.97}]
    return []

# ----------------------------
# DocAI parsing (lightweight: KVs, section lines, dates)
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
class Entity:
    entity_id: str
    page: int
    text: str
    kind: str                  # "date" | "kv" | "section" | "other"
    section_title: Optional[str]
    value_norm_date: Optional[str]
    cui: Optional[str] = None
    cui_score: Optional[float] = None
    emb: Optional[np.ndarray] = None

def parse_docai_entities(docai: Dict) -> List[Entity]:
    entities: List[Entity] = []
    pages = (docai or {}).get("document", {}).get("pages", []) or []
    for p_idx, page in enumerate(pages):
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
        # headings + paragraphs (simple)
        heading = None
        for line in page.get("lines", []) or []:
            t = line.get("layout", {}).get("textAnchor", {}).get("content", "") or line.get("layout", {}).get("content", "")
            if not t:
                continue
            if t.endswith(":") or t.isupper():
                heading = t.strip().strip(":")
            else:
                entities.append(Entity(
                    entity_id=f"sec:{p_idx}:{len(entities)}",
                    page=p_idx,
                    text=f"{(heading or '').upper()}: {t}",
                    kind="section",
                    section_title=(heading or "").upper(),
                    value_norm_date=norm_date(t)
                ))
                for m in DATE_REGEX.finditer(t or ""):
                    entities.append(Entity(
                        entity_id=f"datei:{p_idx}:{len(entities)}",
                        page=p_idx, text=t, kind="date",
                        section_title=(heading or "").upper(),
                        value_norm_date=norm_date(m.group(0))
                    ))
    return entities

# ----------------------------
# CUI mapping + embeddings (genai-only, single-call per text)
# ----------------------------
def attach_cuis_and_embeddings_local(entities: List[Entity]) -> None:
    for e in entities:
        cand = umls_link(e.text)
        if cand:
            best = max(cand, key=lambda x: x.get("score", 0.0))
            e.cui, e.cui_score = best.get("cui"), best.get("score")
        if e.cui and e.cui in CUI_VECTORS:
            e.emb = CUI_VECTORS[e.cui]
        else:
            e.emb = gemini_embed_single(f"[ENTITY:{e.kind}] {e.text[:500]}")

# ----------------------------
# Graph + query
# ----------------------------
def _cos(a: np.ndarray, b: np.ndarray) -> float:
    da = float(np.linalg.norm(a) + 1e-8); db = float(np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / (da * db))

def build_edges(entities: List[Entity], sim_threshold: float=0.65) -> Dict[str, List[Tuple[str, float]]]:
    ids = [e.entity_id for e in entities if e.emb is not None]
    by_id = {e.entity_id: e for e in entities}
    adj: Dict[str, List[Tuple[str, float]]] = {i: [] for i in ids}
    for i in range(len(ids)):
        ei = by_id[ids[i]]
        for j in range(i+1, len(ids)):
            ej = by_id[ids[j]]
            s = _cos(ei.emb, ej.emb)
            if s >= sim_threshold:
                adj[ei.entity_id].append((ej.entity_id, s))
                adj[ej.entity_id].append((ei.entity_id, s))
    return adj

def query_date_local(docai: Dict, query_date_str: str,
                     min_direct: float=0.60, min_indirect: float=0.45, decay: float=0.8):
    ents = parse_docai_entities(docai)
    attach_cuis_and_embeddings_local(ents)
    adj = build_edges(ents, sim_threshold=min_direct)

    q_iso = norm_date(query_date_str) or query_date_str
    by_id = {e.entity_id: e for e in ents}
    dates = [e for e in ents if e.kind == "date" and e.value_norm_date == q_iso]

    out = {"query_date": q_iso, "direct_associations": [], "indirect_associations": []}

    seen = set()
    for d in dates:
        for nbr, score in sorted(adj.get(d.entity_id, []), key=lambda t: -t[1]):
            if nbr in seen: 
                continue
            tgt = by_id[nbr]
            out["direct_associations"].append({
                "target_entity_id": nbr,
                "kind": tgt.kind,
                "section_title": tgt.section_title,
                "cui": tgt.cui,
                "score": round(score, 3),
                "evidence": tgt.text[:220]
            })
            seen.add(nbr)

    # indirect (1 middle hop)
    for d in dates:
        for mid, s1 in adj.get(d.entity_id, []):
            if s1 < min_indirect: 
                continue
            for tgt, s2 in adj.get(mid, []):
                if tgt in seen: 
                    continue
                path_score = s1 * s2 * decay
                if path_score >= min_indirect:
                    e_tgt = by_id[tgt]
                    out["indirect_associations"].append({
                        "target_entity_id": tgt,
                        "kind": e_tgt.kind,
                        "section_title": e_tgt.section_title,
                        "score": round(path_score,3),
                        "path": [
                            {"from": d.entity_id, "to": mid, "score": round(s1,3)},
                            {"from": mid, "to": tgt, "score": round(s2,3)},
                        ],
                        "evidence": e_tgt.text[:220]
                    })
    return out

# ============================
# General query support (genai-only, no batches)
# ============================
def _embed_query_vector(query_text: str) -> np.ndarray:
    cand = umls_link(query_text)
    if cand:
        best = max(cand, key=lambda x: x.get("score", 0.0))
        cui = best.get("cui")
        if cui and cui in CUI_VECTORS:
            return CUI_VECTORS[cui]
    return gemini_embed_single(f"[QUERY] {query_text[:800]}")

def _is_date_like(q: str) -> Optional[str]:
    return norm_date(q)

def _nearest_neighbors(vec: np.ndarray, entities: List[Entity], topk: int = 10) -> List[Tuple[Entity, float]]:
    cands = [(e, _cos(vec, e.emb)) for e in entities if e.emb is not None]
    cands.sort(key=lambda x: -x[1])
    return cands[:topk]

def build_model_local(docai: Dict, min_edge: float = 0.60):
    ents = parse_docai_entities(docai)
    attach_cuis_and_embeddings_local(ents)
    adj = build_edges(ents, sim_threshold=min_edge)
    by_id = {e.entity_id: e for e in ents}
    return ents, adj, by_id

def query_any_local(docai: Dict, query_text: str,
                    min_direct: float = 0.60, min_indirect: float = 0.45, decay: float = 0.8,
                    topk: int = 8):
    ents, adj, by_id = build_model_local(docai, min_edge=min_direct)

    q_iso = _is_date_like(query_text)
    q_vec = _embed_query_vector(query_text)

    nn = _nearest_neighbors(q_vec, ents, topk=topk)

    date_anchors = []
    if q_iso:
        date_anchors = [e for e in ents if e.kind == "date" and e.value_norm_date == q_iso]
        for e in date_anchors:
            nn.insert(0, (e, 1.0))
        seen = set()
        nn = [(e, s) for e, s in nn if not (e.entity_id in seen or seen.add(e.entity_id))]

    direct = []
    kept = set()
    for e, score in nn:
        if score < min_direct:
            continue
        direct.append({
            "entity_id": e.entity_id,
            "kind": e.kind,
            "section_title": e.section_title,
            "cui": e.cui,
            "score": round(float(score), 3),
            "evidence": e.text[:240]
        })
        kept.add(e.entity_id)
        if len(direct) >= topk:
            break

    indirect = []
    start_nodes = date_anchors if date_anchors else [by_id[d["entity_id"]] for d in direct[:3]]
    for src in start_nodes:
        for mid, s1 in adj.get(src.entity_id, []):
            if s1 < min_indirect:
                continue
            for tgt, s2 in adj.get(mid, []):
                if tgt in kept:
                    continue
                ind_score = s1 * s2 * decay
                if ind_score >= min_indirect:
                    tEnt = by_id[tgt]
                    indirect.append({
                        "target_entity_id": tgt,
                        "kind": tEnt.kind,
                        "section_title": tEnt.section_title,
                        "score": round(float(ind_score), 3),
                        "path": [
                            {"from": src.entity_id, "to": mid, "score": round(float(s1), 3)},
                            {"from": mid, "to": tgt, "score": round(float(s2), 3)}
                        ],
                        "evidence": tEnt.text[:240]
                    })
                    kept.add(tgt)

    direct.sort(key=lambda x: -x["score"])
    indirect.sort(key=lambda x: -x["score"])
    return {
        "query": query_text,
        "query_iso_date": q_iso,
        "direct": direct,
        "indirect": indirect
    }
