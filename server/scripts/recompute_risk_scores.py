"""기존 live_reports.json 의 risk_score 를 새 식으로 재계산.

옛 식: damage_rate × (5·n_child + 2·n_elder)        → 0~150 범위
새 식: 객체위험계수 × [loc·0.3 + 빈도·0.2 + 파손·0.3 + 방치·0.2]   → 1.0~3.4 범위

사용:
  python3 scripts/recompute_risk_scores.py        # dry-run
  python3 scripts/recompute_risk_scores.py --go   # 실행
"""
import json, sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LIVE_REPORTS, risk_score, risk_tier

DRY = "--go" not in sys.argv
print(f"=== Recompute risk_scores {'(DRY)' if DRY else '(실행)'} ===\n")

reports = json.load(open(LIVE_REPORTS, encoding="utf-8"))
now = datetime.now()
changes = 0

for r in reports:
    old = r.get("risk_score", 0)
    # elapsed_days 계산 (created_at → 지금)
    try:
        c = datetime.fromisoformat(r.get("created_at", "").split("+")[0].replace("Z",""))
        elapsed = max((now - c).days, 0)
    except Exception:
        elapsed = 0
    new = risk_score(
        damage_rate = int(r.get("damage_rate") or 1),
        n_child     = int(r.get("n_child") or 0),
        n_elder     = int(r.get("n_elder") or 0),
        item        = r.get("item", ""),
        elapsed_days= elapsed,
    )
    if abs(old - new) > 0.01:
        changes += 1
        if changes <= 10:
            print(f"  {r['id']} | {r.get('item','-'):15s} | {old:>6.2f} → {new:>5.2f}")

print(f"\n총 {len(reports)}건 중 {changes}건 변경 예정.")

if not DRY and changes > 0:
    bak = Path(str(LIVE_REPORTS) + f".bak_pre_recompute_{now.strftime('%Y%m%d_%H%M%S')}")
    bak.write_text(open(LIVE_REPORTS, encoding="utf-8").read(), encoding="utf-8")
    print(f"\n백업: {bak.name}")
    # 다시 계산하면서 저장
    for r in reports:
        try:
            c = datetime.fromisoformat(r.get("created_at", "").split("+")[0].replace("Z",""))
            elapsed = max((now - c).days, 0)
        except Exception:
            elapsed = 0
        r["risk_score"] = risk_score(
            damage_rate = int(r.get("damage_rate") or 1),
            n_child     = int(r.get("n_child") or 0),
            n_elder     = int(r.get("n_elder") or 0),
            item        = r.get("item", ""),
            elapsed_days= elapsed,
        )
    LIVE_REPORTS.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"✓ {LIVE_REPORTS} 갱신 완료")
elif DRY:
    print("\n(--go 붙여 다시 실행)")
