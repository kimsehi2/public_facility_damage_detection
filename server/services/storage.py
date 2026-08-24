"""Live 신고 원장 저장소 — JSON 파일 + Chroma 업서트."""
import json
import logging
import statistics
from functools import lru_cache
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

from config import LIVE_REPORTS, CHROMA_DIR, COLLECTION, EMBED_MODEL, ALLOWED_ITEMS

log = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _embedder():
    log.info(f"임베딩 모델 로드: {EMBED_MODEL}")
    return SentenceTransformer(EMBED_MODEL)

@lru_cache(maxsize=1)
def collection():
    log.info(f"Chroma 연결: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

def list_reports() -> List[dict]:
    if not LIVE_REPORTS.exists():
        return []
    return json.loads(LIVE_REPORTS.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def item_repair_stats() -> dict:
    """OBB 인식 11 시설 전체 + 손상단계별 평균 수리비/소요일 (Estimate_DB CSV 직접 읽음).

    Chroma 백엔드 schema 이슈 회피 위해 CSV 직접 파싱. 운영 8 (ALLOWED) + 비운영 3 (보호펜스/무단횡단/가로수)
    = 11 시설 + 단계 (1/2/3) 별로 stats 제공. 챗봇이 모든 OBB 인식 시설 답변 가능.

    반환: {품목명: {n, avg_days, median_days, avg_cost, median_cost, by_stage: {1:{...},2:{...},3:{...}}}}
    """
    import csv
    from collections import defaultdict
    from config import EST_CSV, ITEM_NORMALIZE

    # 통계 대상: 운영 8 시설 (ALLOWED_ITEMS) 만 — legacy(보호펜스/가로수보호/무단횡단방지) 제외
    target_items = set(ALLOWED_ITEMS)

    rows_by_item_stage = defaultdict(list)   # (item, stage) → [{cost, days}]
    rows_by_item       = defaultdict(list)   # item → [{cost, days}]

    if not EST_CSV.exists():
        return {}

    with open(EST_CSV, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            raw = (row.get("Item") or "").strip()
            item = ITEM_NORMALIZE.get(raw, raw)
            if item not in target_items:
                continue
            try:
                stage = int(row.get("Damage_Rate") or 0)
                cost  = int(row.get("Cost") or 0)
                days  = int(row.get("Repair_Days") or 0)
            except ValueError:
                continue
            entry = {"cost": cost, "days": days}
            rows_by_item[item].append(entry)
            if stage in (1, 2, 3):
                rows_by_item_stage[(item, stage)].append(entry)

    def _agg(entries):
        if not entries: return None
        days  = [e["days"]  for e in entries if e["days"]]
        costs = [e["cost"]  for e in entries if e["cost"]]
        if not (days or costs): return None
        return {
            "n": max(len(days), len(costs)),
            "avg_days":    round(sum(days) / len(days), 1)    if days  else None,
            "median_days": int(statistics.median(days))        if days  else None,
            "avg_cost":    int(sum(costs) / len(costs))        if costs else None,
            "median_cost": int(statistics.median(costs))       if costs else None,
        }

    out = {}
    for item, entries in rows_by_item.items():
        agg = _agg(entries)
        if not agg: continue
        agg["by_stage"] = {}
        for stage in (1, 2, 3):
            s = _agg(rows_by_item_stage.get((item, stage), []))
            if s: agg["by_stage"][stage] = s
        out[item] = agg
    return out

def save_report(data: dict):
    reports = list_reports()
    reports.append(data)
    LIVE_REPORTS.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    n_child = int(data.get("n_child", 0))
    n_elder = int(data.get("n_elder", 0))
    group_id = data.get("group_id", "")
    duplicate_of = data.get("duplicate_of") or ""
    nz = data.get("nearest_zone") or {}
    nz_name = nz.get("name", "")
    nz_dist = float(nz.get("distance_m") or 0.0)
    nz_group = nz.get("group", "")
    doc = (
        f"[live] {data['item']} 손상도 {data['damage_rate']}단계, "
        f"그룹 {group_id}, "
        f"위치 {data['location']['lat']:.5f},{data['location']['lon']:.5f}, "
        f"보호구역 child {n_child}개·elder {n_elder}개 (300m내), "
        f"최근접 {nz_group}/{nz_name} ({int(nz_dist)}m), "
        f"위험도 {data['risk_score']}, 접수 {data['created_at']}"
    )
    try:
        col = collection()
        emb = _embedder().encode([doc]).tolist()
        col.upsert(
            ids=[data["id"]],
            documents=[doc],
            embeddings=emb,
            metadatas=[{
                "source": "live",
                "user_id": data.get("user_id", ""),
                "Item": data["item"],
                "Damage_Rate": int(data["damage_rate"]),
                "Group_ID": group_id,
                "duplicate_of": duplicate_of,
                "match_score": int(data.get("match_score") or 0),
                "lat": float(data["location"]["lat"]),
                "lon": float(data["location"]["lon"]),
                "risk_score": float(data["risk_score"]),
                "n_child": n_child,
                "n_elder": n_elder,
                "nearest_zone_group": nz_group,
                "nearest_zone_name": nz_name,
                "nearest_zone_distance_m": nz_dist,
                "status": data["status"],
                "created_at": data["created_at"],
                "image_path": data["image_path"],
            }],
        )
    except Exception as e:
        log.warning(f"Chroma 업서트 실패(JSON은 저장됨): {e}")
