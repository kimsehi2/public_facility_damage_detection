"""Chroma 메타 + 문서 정규화 (원본 CSV 보존, Chroma만 업데이트).

ITEM_NORMALIZE + STATUS_NORMALIZE 딕셔너리 기준:
  1) rep/est 소스의 Item 메타 정규화
  2) rep 소스의 Status 메타 정규화
  3) rep 소스의 doc text 도 교체 (embedding은 유지 — 재임베딩 비용 절감,
     semantic 차이 미미: 한글 "신고접수" ↔ 영문 "pending")

rebuild_chroma.py 에도 동일 로직 반영됨 → 다음 전체 재빌드 시 일관성 보장.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ITEM_NORMALIZE, STATUS_NORMALIZE
from services import storage

col = storage.collection()

# ─────────────────────────────────────────────────────────────
# 1) Item 정규화 (est + rep 모두)
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Item 정규화")
print("=" * 60)
item_updated = 0
for original, canonical in ITEM_NORMALIZE.items():
    got = col.get(where={"Item": original}, include=["metadatas"])
    ids = got["ids"]
    if not ids:
        print(f"  '{original}' → 매칭 0건 (이미 정규화됐거나 원본에 없음)")
        continue
    new_metas = [{**m, "Item": canonical} for m in got["metadatas"]]
    col.update(ids=ids, metadatas=new_metas)
    print(f"  ✓ '{original}' → '{canonical}' : {len(ids)}건")
    item_updated += len(ids)

# ─────────────────────────────────────────────────────────────
# 2) Status 정규화 (rep 만) — 메타 + doc text
# ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Status 정규화 (rep 메타 + doc text)")
print("=" * 60)
status_updated = 0
for original, canonical in STATUS_NORMALIZE.items():
    got = col.get(
        where={"$and": [{"source": "rep"}, {"Status": original}]},
        include=["metadatas", "documents"],
    )
    ids = got["ids"]
    if not ids:
        print(f"  '{original}' → 매칭 0건")
        continue

    new_metas = [{**m, "Status": canonical} for m in got["metadatas"]]
    # NOTE: docs 텍스트는 그대로 (Chroma가 doc 교체 시 기본 임베더로 재임베딩 시도 → 차원 불일치 에러).
    # LLM 프롬프트에 한글 상태 매핑 힌트를 제공하는 편으로 처리.
    col.update(ids=ids, metadatas=new_metas)
    print(f"  ✓ '{original}' → '{canonical}' : {len(ids)}건 (메타만, doc 는 한글 유지)")
    status_updated += len(ids)

print()
print(f"완료. Item {item_updated}건, Status {status_updated}건 업데이트.")
print("서버 재요청 시 갱신된 값 반영됨. lru_cache 는 uvicorn reload 또는 수동 clear 로 갱신.")
