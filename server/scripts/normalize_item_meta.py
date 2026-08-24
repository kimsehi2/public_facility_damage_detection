"""Chroma 메타 Item 정규화 (원본 CSV 보존, Chroma 만 업데이트).

ITEM_NORMALIZE 딕셔너리에 정의된 매핑에 따라 est/rep 소스의 Item 메타를
ALLOWED_ITEMS 표기로 교체. 1회 실행 + rebuild_chroma.py 에도 동일 로직 반영.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ITEM_NORMALIZE
from services import storage

col = storage.collection()

total_updated = 0
for original, canonical in ITEM_NORMALIZE.items():
    got = col.get(where={"Item": original}, include=["metadatas"])
    ids = got["ids"]
    if not ids:
        print(f"  '{original}' → 매칭 0건 (이미 정규화됐거나 원본에 없음)")
        continue

    new_metas = [{**m, "Item": canonical} for m in got["metadatas"]]
    col.update(ids=ids, metadatas=new_metas)
    print(f"  ✓ '{original}' → '{canonical}' : {len(ids)}건 업데이트")
    total_updated += len(ids)

    # 검증
    after = col.get(where={"Item": canonical}, include=["metadatas"])
    print(f"    확인: '{canonical}' 이름 총 {len(after['ids'])}건")

print(f"\n완료. 총 {total_updated}건 메타 업데이트.")
print("서버(uvicorn) 가 lru_cache 잡고 있을 수 있음 — 다음 요청은 갱신되지만 item_repair_stats 같은 캐시는 재시작 또는 수동 clear 필요.")
