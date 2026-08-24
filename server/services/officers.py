"""담당 부서·담당자 조회 (Officer_DB.csv 기반).

신고 좌표 → reverse-geocoding(sido/sigungu) → Officer_DB 조회 → {dept, name, phone}

데이터 출처:
- 포항권 (sigungu_prefix=포항시): Report_DB.csv historical 1만건 통계 기반 **실데이터**
  · 도로과 이영후 / 공원녹지과 최두식 / 시설관리과 정우성
- 서울/부산/제주 (sido): **데모용 가상 데이터** (실제 공무원 정보 아님)
- 매칭 row 없는 region: 부서명(ITEM_TO_DEPT 자동 결정) 만 반환, 담당자 빈칸
"""
import csv
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from config import BASE_DIR, ITEM_TO_DEPT

log = logging.getLogger(__name__)

OFFICER_CSV = BASE_DIR / "data" / "Officer_DB.csv"


@lru_cache(maxsize=1)
def _table() -> list[dict]:
    if not OFFICER_CSV.exists():
        log.warning(f"Officer_DB 없음: {OFFICER_CSV}")
        return []
    with OFFICER_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _match_row(rows: list[dict], sido: str, sigungu: str, dept: str) -> Optional[dict]:
    """가장 구체적인(=좁은) 매칭 우선: sigungu_prefix > sido."""
    sigungu = sigungu or ""
    sido = sido or ""
    # 1) sigungu_prefix 우선
    for r in rows:
        if r.get("Region_Type") == "sigungu_prefix" and r.get("Dept") == dept:
            m = r.get("Region_Match") or ""
            if m and sigungu.startswith(m):
                return r
    # 2) sido 매칭
    for r in rows:
        if r.get("Region_Type") == "sido" and r.get("Dept") == dept:
            if (r.get("Region_Match") or "") == sido:
                return r
    return None


def lookup(item: str = "", sido: str = "", sigungu: str = "") -> dict:
    """item + region → 담당 부서·담당자 정보.

    반환:
      {
        "dept": "도로과",
        "dept_full": "포항시남구 도로과",   # 표시용 (sigungu + dept)
        "name": "이영후" 또는 "",
        "phone": "010-..." 또는 "",
        "matched": True/False,             # Officer_DB 매칭 여부
      }
    """
    dept = ITEM_TO_DEPT.get(item, "시설관리과")  # 모르는 품목은 시설관리과 default
    full = (sigungu + " " if sigungu else "") + dept

    row = _match_row(_table(), sido, sigungu, dept)
    if not row:
        return {"dept": dept, "dept_full": full, "name": "", "phone": "",
                "matched": False}

    name = (row.get("Name") or "").strip()
    phone = (row.get("Phone") or "").strip()
    return {"dept": dept, "dept_full": full,
            "name": name, "phone": phone,
            "matched": bool(name)}
