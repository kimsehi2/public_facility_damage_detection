"""긴급도 집계 — Group_ID 단위로 live_reports 합쳐서 최종 점수 산출.

- status == 'completed' / 'rejected' 멤버는 매칭/통계에서 제외 (Group_DB 통합 정책)
- 한 그룹이라도 active(pending/in_progress) 가 있으면 해당 그룹은 긴급도 뷰에 노출

긴급도 식 (각 항 1/2/3 이산 점수):
  긴급도 = 객체위험계수(ITEM_BASE_URGENCY)
         × [위치위험도 × 0.3 + 민원빈도 × 0.2 + 파손도 × 0.3 + 방치기간 × 0.2]

  · 위치위험도 = 보호구역 종류 수 → 1/2/3
                  - 1: 보호구역 없음
                  - 2: 어린이 또는 노인 한 종류만
                  - 3: 어린이 + 노인 둘 다
  · 민원빈도   = active_count → 1/2/3
                  - 1: <5건
                  - 2: 5~30건
                  - 3: 30건 이상
  · 파손도     = damage_rate (1/2/3 단계, 이미 시설별 임계로 구간화됨)
  · 방치기간   = elapsed_days → 1/2/3
                  - 1: <7일
                  - 2: 7~30일
                  - 3: 30일 이상

반환 범위: 약 1.0 ~ 3.9 (객체위험계수 1.0~1.3 × 가중합 1~3)
"""
from datetime import datetime
from typing import List, Optional

from config import emergency_tier, ITEM_BASE_URGENCY, ITEM_DAMAGE_THRESHOLDS, DAMAGE_RATE_THRESHOLDS_DEFAULT


def _parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00").split("+")[0])
    except Exception:
        return None


def _elapsed_days(first_iso: str, now: Optional[datetime] = None) -> int:
    first = _parse_iso(first_iso)
    if not first:
        return 0
    now = now or datetime.now()
    return max((now - first).days, 0)


def compute_final_score(*, max_n_child: int, max_n_elder: int,
                        frequency: int, max_damage_rate: int,
                        max_damage_ratio: Optional[float],
                        elapsed_days: int, item: str = "") -> float:
    """긴급도 = 객체위험계수 × [위치위험도×0.4 + 민원빈도×0.3 + 파손도×0.2 + 방치기간×0.1].
    각 항 1/2/3 이산 점수. 객체위험계수 1.0~1.3.
    반환 범위: 약 1.0 ~ 3.9.
    (max_damage_ratio 인자는 호환용 — 현재 식에선 사용 안 함, damage_rate 만 사용)
    """
    # 1) 위치위험도 — 보호구역 종류 수
    has_child = max_n_child > 0
    has_elder = max_n_elder > 0
    if has_child and has_elder: location_score = 3
    elif has_child or has_elder: location_score = 2
    else: location_score = 1

    # 2) 민원빈도 — 5건/30건 임계
    if frequency >= 30:  freq_score = 3
    elif frequency >= 5: freq_score = 2
    else:                freq_score = 1

    # 3) 파손도 — damage_rate (이미 시설별 임계로 구간화 1/2/3)
    damage_score = max(min(int(max_damage_rate or 1), 3), 1)

    # 4) 방치기간 — 7일/30일 임계
    if elapsed_days >= 30:  elapsed_score = 3
    elif elapsed_days >= 7: elapsed_score = 2
    else:                   elapsed_score = 1

    weighted = (location_score * 0.3 + freq_score * 0.2
                + damage_score * 0.3 + elapsed_score * 0.2)
    item_w = ITEM_BASE_URGENCY.get(item, 1.0)
    return round(item_w * weighted, 3)


def aggregate(reports: List[dict],
              include_statuses=("pending", "in_progress")) -> List[dict]:
    """live_reports → Group_ID 단위 집계 리스트.

    긴급도 뷰는 **활성(pending/in_progress) 신고만** 반영:
      - completed / rejected 멤버는 `member_ids` 에서 제외
      - frequency / first·last / max_risk / final_score 도 활성 기준
      - 그룹 전체가 활성 0건 이면 뷰에서 숨김
      - `completed_count` 는 참고용으로 유지 (프론트 "+ 완료 N건" 뱃지 가능)

    각 그룹 dict:
      group_id, item, frequency, max_damage_rate, max_risk_score,
      first_report_date, last_report_date, elapsed_days,
      final_score, tier, representative_id, location, address,
      member_ids (활성만), active_count, completed_count
    """
    groups: dict[str, dict] = {}
    for r in reports:
        gid = r.get("group_id")
        if not gid:
            # 구 버전 데이터 (group_id 없음) — id 자체를 단일 그룹으로 취급
            gid = f"LEGACY_{r['id']}"
        g = groups.setdefault(gid, {
            "group_id": gid,
            "members": [],
            "active_members": [],
            "completed_members": [],
        })
        g["members"].append(r)
        # rejected 도 '처리됨'으로 보고 긴급도 뷰에서 제외
        if r.get("status") in include_statuses:
            g["active_members"].append(r)
        else:
            g["completed_members"].append(r)

    out = []
    now = datetime.now()
    for gid, g in groups.items():
        active = g["active_members"]
        # 활성 멤버 0 → 긴급도 뷰에서 제외 (모든 건이 completed/rejected)
        if not active:
            continue

        # 모든 통계·대표·멤버 목록은 **활성만** 기준
        rep = max(active, key=lambda r: r.get("risk_score", 0))

        dates = [r.get("created_at", "") for r in active if r.get("created_at")]
        first = min(dates) if dates else ""
        last = max(dates) if dates else ""
        elapsed = _elapsed_days(first, now=now)

        max_risk = max((r.get("risk_score") or 0) for r in active)
        max_damage = max((r.get("damage_rate") or 0) for r in active)
        # damage_ratio 는 신규 필드. 옛 데이터엔 없을 수 있음 → None 처리
        ratios = [r.get("damage_ratio") for r in active if r.get("damage_ratio") is not None]
        max_ratio = max(ratios) if ratios else None
        max_nc = max((r.get("n_child", 0) or 0) for r in active)
        max_ne = max((r.get("n_elder", 0) or 0) for r in active)
        freq = len(active)
        # 그룹 item 은 모든 active 멤버 동일 (feature matching 이 같은 품목 한정)
        item = rep.get("item", "")
        final = compute_final_score(
            max_n_child=max_nc, max_n_elder=max_ne,
            frequency=freq, max_damage_rate=int(max_damage),
            max_damage_ratio=max_ratio,
            elapsed_days=elapsed, item=item,
        )
        emoji, color, label = emergency_tier(final)

        addr = (rep.get("address") or {})
        out.append({
            "group_id": gid,
            "item": rep.get("item"),
            "frequency": freq,
            "active_count": len(active),
            "completed_count": len(g["completed_members"]),
            "max_damage_rate": int(max_damage),
            "max_risk_score": round(float(max_risk), 2),
            "first_report_date": first,
            "last_report_date": last,
            "elapsed_days": elapsed,
            "final_score": final,
            "tier": {"emoji": emoji, "color": color, "label": label},
            "representative_id": rep["id"],
            "location": rep.get("location") or {},
            "address": {
                "address": addr.get("address", ""),
                "sido": addr.get("sido", ""),
                "sigungu": addr.get("sigungu", ""),
            },
            "member_ids": [r["id"] for r in sorted(active, key=lambda x: x.get("created_at", ""))],
        })

    out.sort(key=lambda g: -g["final_score"])
    return out
