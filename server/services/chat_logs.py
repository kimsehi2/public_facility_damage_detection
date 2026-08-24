"""민원인 챗봇 대화 로그 저장·조회·피드백.

단순 JSON append-only 파일. 실서비스는 SQLite/DB로 승격 권장.
"""
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config import BASE_DIR

log = logging.getLogger(__name__)
LOGS_FILE = BASE_DIR / "chat_logs.json"


def _read_all() -> List[dict]:
    if not LOGS_FILE.exists():
        return []
    try:
        return json.loads(LOGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        log.exception("chat_logs.json 로드 실패")
        return []


def _write_all(logs: List[dict]):
    LOGS_FILE.write_text(
        json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_turn(user_id: str, user_name: str, message: str, reply: str) -> str:
    """대화 1턴 저장. 생성된 turn_id 반환 (클라이언트 피드백용)."""
    turn_id = uuid.uuid4().hex[:12]
    logs = _read_all()
    logs.append({
        "turn_id": turn_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user_id": user_id,
        "user_name": user_name,
        "message": message,
        "reply": reply,
        "feedback": None,      # None | "good" | "bad"
        "feedback_at": None,
        "feedback_note": None,
    })
    _write_all(logs)
    return turn_id


def set_feedback(turn_id: str, rating: str, note: Optional[str] = None) -> bool:
    """rating: 'good' | 'bad'. 성공 여부 반환."""
    if rating not in ("good", "bad"):
        return False
    logs = _read_all()
    hit = False
    for r in logs:
        if r["turn_id"] == turn_id:
            r["feedback"] = rating
            r["feedback_at"] = datetime.now().isoformat(timespec="seconds")
            if note:
                r["feedback_note"] = note
            hit = True
            break
    if hit:
        _write_all(logs)
    return hit


def list_logs(user_id: Optional[str] = None, rating: Optional[str] = None,
              date: Optional[str] = None, limit: int = 200, offset: int = 0) -> dict:
    """관리자 조회. date: 'YYYY-MM-DD' 로 해당 날 필터."""
    logs = _read_all()
    if user_id:
        logs = [r for r in logs if r.get("user_id") == user_id]
    if rating == "good":
        logs = [r for r in logs if r.get("feedback") == "good"]
    elif rating == "bad":
        logs = [r for r in logs if r.get("feedback") == "bad"]
    elif rating == "none":
        logs = [r for r in logs if r.get("feedback") is None]
    if date:
        logs = [r for r in logs if r.get("timestamp", "")[:10] == date]
    logs.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    total = len(logs)
    page = logs[offset:offset + limit]
    return {"total": total, "items": page, "offset": offset, "limit": limit}


def stats() -> dict:
    """간단 요약 통계."""
    logs = _read_all()
    total = len(logs)
    good = sum(1 for r in logs if r.get("feedback") == "good")
    bad = sum(1 for r in logs if r.get("feedback") == "bad")
    users = len({r.get("user_id") for r in logs})
    return {"total": total, "good": good, "bad": bad, "unique_users": users}
