"""관리자 계정 파일 기반 로그인 + 파일 영속 세션 토큰 관리.

- 계정 파일: server/admin_users.json (없으면 기본 admin/admin 자동 생성)
- 세션 파일: server/data/admin_sessions.json — uvicorn reload/재시작 시 토큰 유지
  (예전엔 인메모리 dict 여서 파일 수정마다 전 관리자 강제 로그아웃되던 이슈 해결)
- 실서비스: bcrypt 해시 + Redis 세션 저장 권장.
"""
import json
import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional

from config import BASE_DIR

log = logging.getLogger(__name__)

ACCOUNTS_FILE = BASE_DIR / "admin_users.json"
SESSIONS_FILE = BASE_DIR / "data" / "admin_sessions.json"
SESSION_TTL = timedelta(hours=8)

# 인메모리 캐시 + 파일 미러
_sessions: dict[str, tuple[str, datetime]] = {}


def _load_sessions_from_disk():
    """파일에서 세션 로드 (프로세스 시작/리로드 시 1회)."""
    global _sessions
    if not SESSIONS_FILE.exists():
        _sessions = {}
        return
    try:
        raw = json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
        now = datetime.now()
        loaded = {}
        for token, (username, exp_iso) in raw.items():
            exp = datetime.fromisoformat(exp_iso)
            if exp > now:
                loaded[token] = (username, exp)
        _sessions = loaded
        if loaded:
            log.info(f"admin 세션 복원: {len(loaded)}개")
    except Exception:
        log.exception("admin_sessions.json 로드 실패 — 빈 세션으로 시작")
        _sessions = {}


def _flush_sessions_to_disk():
    """세션 변경 후 파일 덮어쓰기."""
    try:
        SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {token: (username, exp.isoformat())
                for token, (username, exp) in _sessions.items()}
        SESSIONS_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        log.exception("admin_sessions.json 저장 실패 (세션은 메모리에만 존재)")


# 모듈 로드 시 한 번 복원
_load_sessions_from_disk()


def _ensure_file():
    """최초 기동 시 기본 계정 생성."""
    if ACCOUNTS_FILE.exists():
        return
    default = [
        {"username": "admin", "password": "admin", "name": "시스템관리자"},
    ]
    ACCOUNTS_FILE.write_text(
        json.dumps(default, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.warning("admin_users.json 생성 — 기본 계정 admin/admin. 배포 전 반드시 변경!")


def _load_users() -> list[dict]:
    _ensure_file()
    try:
        return json.loads(ACCOUNTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        log.exception("admin_users.json 로드 실패")
        return []


def verify_login(username: str, password: str) -> Optional[dict]:
    for u in _load_users():
        if u.get("username") == username and u.get("password") == password:
            out = {"username": u["username"], "name": u.get("name", u["username"])}
            if u.get("region"):
                out["region"] = u["region"]
            return out
    return None


def create_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    _sessions[token] = (username, datetime.now() + SESSION_TTL)
    _flush_sessions_to_disk()
    return token


def verify_token(token: str) -> Optional[str]:
    info = _sessions.get(token)
    # 인메모리에 없으면 파일에서 한 번 더 복원 시도 (다른 프로세스 로그인 케이스)
    if not info and SESSIONS_FILE.exists():
        _load_sessions_from_disk()
        info = _sessions.get(token)
    if not info:
        return None
    username, exp = info
    if datetime.now() > exp:
        _sessions.pop(token, None)
        _flush_sessions_to_disk()
        return None
    return username


def get_user_by_token(token: str) -> Optional[dict]:
    """토큰으로 사용자 전체 정보(region 포함) 조회. 유효 token 아니면 None."""
    username = verify_token(token)
    if not username:
        return None
    for u in _load_users():
        if u.get("username") == username:
            return u
    return None


def logout(token: str):
    if _sessions.pop(token, None) is not None:
        _flush_sessions_to_disk()
