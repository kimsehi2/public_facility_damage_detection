"""시연용 깨끗한 상태 만들기.

지우는 것:
  - uploads/*.jpg/jpeg/png         (사용자 업로드 사진)
  - uploads/crops/, features/, matches/  (산출물)
  - live_reports.json               (백업 후 [])
  - chat_logs.json                  (백업 후 [])
  - Chroma 'estimates' 컬렉션 중 source='live'

유지:
  - admin_users.json, User_DB.csv  (로그인 필요)
  - chroma_est/  est/rep/emg 데이터 (RAG 컨텍스트)
  - features_orb_bak/              (옛 백업)

사용법:
  cd /home/piai/다운로드/llm및\\ data/server
  python scripts/cleanup_for_demo.py        # dry-run (뭐 지울지 보기만)
  python scripts/cleanup_for_demo.py --go   # 실행
"""
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

DRY = "--go" not in sys.argv
TS = datetime.now().strftime("%Y%m%d_%H%M%S")

BASE = Path(__file__).resolve().parent.parent  # .../server
UPLOADS = BASE / "uploads"
LIVE_REPORTS = BASE / "live_reports.json"
CHAT_LOGS = BASE / "chat_logs.json"
CHROMA_DIR = BASE.parent / "chroma_est"
COLLECTION = "estimates"

print(f"=== Cleanup for demo {'(DRY RUN — --go 붙이면 실행)' if DRY else '(실행 모드)'} ===\n")


def log(msg):
    prefix = "[DRY] " if DRY else "[실행] "
    print(prefix + msg)


# ── 1. uploads/*.jpg|jpeg|png 삭제 ──
photos = [p for p in UPLOADS.iterdir()
          if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
log(f"uploads/ 사진 {len(photos)}장 삭제")
if not DRY:
    for p in photos:
        p.unlink()

# ── 2. crops/, features/, matches/ 비우기 (디렉토리 유지) ──
for sub in ["crops", "features", "matches"]:
    d = UPLOADS / sub
    if d.exists():
        cnt = sum(1 for _ in d.iterdir())
        log(f"uploads/{sub}/ 내용 {cnt}건 삭제 (폴더 유지)")
        if not DRY:
            for p in d.iterdir():
                if p.is_file():
                    p.unlink()
                else:
                    shutil.rmtree(p)

# ── 3. live_reports.json 백업 후 [] ──
if LIVE_REPORTS.exists():
    n = len(json.load(open(LIVE_REPORTS)))
    bak = LIVE_REPORTS.with_suffix(f".json.bak_pre_demo_{TS}")
    log(f"live_reports.json {n}건 → 백업({bak.name}) + []")
    if not DRY:
        shutil.copy(LIVE_REPORTS, bak)
        LIVE_REPORTS.write_text("[]")

# ── 4. chat_logs.json 백업 후 [] ──
if CHAT_LOGS.exists():
    n = len(json.load(open(CHAT_LOGS)))
    bak = CHAT_LOGS.with_suffix(f".json.bak_pre_demo_{TS}")
    log(f"chat_logs.json {n}건 → 백업({bak.name}) + []")
    if not DRY:
        shutil.copy(CHAT_LOGS, bak)
        CHAT_LOGS.write_text("[]")

# ── 5. Chroma estimates 컬렉션 중 source='live' 만 삭제 ──
try:
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_or_create_collection(COLLECTION)
    # source='live' 만 필터링
    res = col.get(where={"source": "live"}, include=[])
    ids = res.get("ids") or []
    log(f"Chroma {COLLECTION} 의 source=live 항목 {len(ids)}개 삭제")
    if not DRY and ids:
        col.delete(ids=ids)
except ImportError:
    log("chromadb 미설치 — Chroma 정리 skip")
except Exception as e:
    log(f"Chroma 처리 실패 (무시): {e}")

print("\n완료." + (" (--go 붙여서 다시 실행)" if DRY else ""))
