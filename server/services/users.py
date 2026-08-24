"""User_DB.csv 기반 민원인 조회·로그인·회원가입."""
import csv
import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

from config import PARENT_DIR

log = logging.getLogger(__name__)
USER_CSV = PARENT_DIR / "User_DB.csv"


def _clean_contact(s) -> str:
    return str(s).replace("-", "").replace(" ", "").strip()


@lru_cache(maxsize=1)
def _df():
    log.info(f"User_DB 로드: {USER_CSV}")
    df = pd.read_csv(USER_CSV, encoding="utf-8-sig")
    # 연락처에서 하이픈 제거 후 마지막 4자리
    df["Contact4"] = df["Contact"].astype(str).str.replace("-", "", regex=False).str[-4:]
    return df


def verify(user_id: str, contact4: str):
    """로그인 검증: User_ID + 연락처 뒤 4자리. 성공 시 유저 정보 dict."""
    df = _df()
    hit = df[(df["User_ID"] == user_id) & (df["Contact4"] == str(contact4).strip())]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return {
        "user_id": str(row["User_ID"]),
        "name":    str(row["Name"]),
        "contact": str(row["Contact"]),
        "job":     str(row.get("Job") or ""),
    }


def get(user_id: str):
    """user_id로 유저 정보 조회. 없으면 None."""
    df = _df()
    hit = df[df["User_ID"] == user_id]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return {"user_id": str(row["User_ID"]), "name": str(row["Name"])}


def signup(name: str, contact: str, job: str = "", age: int = 0) -> dict:
    """새 사용자 등록. 중복 연락처면 {ok:False, error:'already_registered', user_id:...}.
    성공 시 {ok:True, user_id, name, contact, contact4}. User_DB.csv 에 append + lru 캐시 무효화.
    """
    df = _df()
    c = _clean_contact(contact)
    if len(c) < 4 or not c.isdigit():
        return {"ok": False, "error": "invalid_contact"}

    existing = df[df["Contact"].astype(str).apply(_clean_contact) == c]
    if not existing.empty:
        return {"ok": False, "error": "already_registered",
                "user_id": str(existing.iloc[0]["User_ID"])}

    def _num(uid):
        try: return int(str(uid).replace("USER_", ""))
        except Exception: return 0
    next_num = max((_num(x) for x in df["User_ID"]), default=1000) + 1
    new_id = f"USER_{next_num}"

    with open(USER_CSV, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([new_id, name, contact, job or "-", int(age or 0)])
    _df.cache_clear()
    log.info(f"signup: {new_id} ({name}, {contact})")
    return {"ok": True, "user_id": new_id, "name": name, "contact": contact,
            "contact4": c[-4:]}
