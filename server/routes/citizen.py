"""민원인 API — 로그인, 사진 업로드, 신고 등록, LLM 챗봇."""
import re
import uuid
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from config import (
    UPLOADS_DIR, risk_score, risk_tier, LLM_MODEL_CITIZEN,
    ZONE_RADIUS_M, ALLOWED_ITEMS, damage_ratio_to_rate,
)
from services import yolo_service, zones, storage, llm, users, chat_logs, events, geocoding, feature_service

log = logging.getLogger(__name__)
router = APIRouter()


class LoginIn(BaseModel):
    user_id: str
    contact4: str   # 연락처 뒤 4자리


class SignupIn(BaseModel):
    name: str
    contact: str                # 예: 010-1234-5678 또는 01012345678
    job: Optional[str] = ""
    age: Optional[int] = 0


class ReportIn(BaseModel):
    id: str
    item: str
    lat: float
    lon: float
    description: Optional[str] = ""
    user_id: str
    # damage_rate 는 더 이상 클라이언트 입력 X — 서버가 seg 모델로 자동 산정.


class ChatTurn(BaseModel):
    role: str
    text: str


class ChatIn(BaseModel):
    message: str
    history: List[ChatTurn] = []
    user_id: str


class FeedbackIn(BaseModel):
    turn_id: str
    rating: str                        # 'good' | 'bad'
    note: Optional[str] = None


@router.get("/ping")
def ping():
    return {"ok": True, "scope": "citizen"}


@router.get("/items")
def items():
    """신고 가능 품목 화이트리스트 — config.ALLOWED_ITEMS 기준."""
    return {"items": sorted(ALLOWED_ITEMS)}


@router.get("/notifications")
def notifications(user_id: str):
    """본인 신고 중 상태 변경이 발생한 건 목록 (최신순). 민원인 알림 패널용.

    반환: 상태가 pending 이 아닌(=처리 시작/완료/반려된) 신고.
    """
    mine = [r for r in storage.list_reports() if r.get("user_id") == user_id]
    out = []
    for r in mine:
        st = r.get("status", "pending")
        if st == "pending":
            continue
        out.append({
            "id":           r["id"],
            "item":         r.get("item", ""),
            "status":       st,
            "created_at":   r.get("created_at"),
            "updated_at":   r.get("updated_at") or r.get("completed_at") or r.get("created_at"),
            "completed_at": r.get("completed_at"),
        })
    out.sort(key=lambda x: x["updated_at"] or "", reverse=True)
    return {"user_id": user_id, "notifications": out}


@router.post("/login")
def login(body: LoginIn):
    """User_DB.csv 기반 로그인 (User_ID + 연락처 뒤 4자리)."""
    info = users.verify(body.user_id.strip(), body.contact4.strip())
    if not info:
        raise HTTPException(401, "아이디 또는 연락처 뒤 4자리가 일치하지 않습니다.")
    return info


@router.post("/signup")
def signup(body: SignupIn):
    """회원가입 — 이름 + 연락처. User_ID 자동 발급, PIN 은 연락처 뒤 4자리."""
    name = body.name.strip()
    contact = body.contact.strip()
    if not name or not contact:
        raise HTTPException(400, "이름과 연락처는 필수입니다.")
    result = users.signup(name, contact, body.job or "", body.age or 0)
    if result.get("ok"):
        return result
    err = result.get("error")
    if err == "already_registered":
        raise HTTPException(409, f"이미 가입된 연락처입니다. 본인 아이디: {result.get('user_id')}")
    if err == "invalid_contact":
        raise HTTPException(400, "연락처는 숫자만 포함해야 하며 4자리 이상이어야 합니다.")
    raise HTTPException(500, "회원가입 처리 실패")


@router.post("/upload")
async def upload(image: UploadFile = File(...)):
    """사진 업로드 → YOLO 분석 + EXIF GPS. 서버는 이미지를 LIVE_XXXX 로 즉시 저장."""
    content = await image.read()
    if not content:
        raise HTTPException(400, "빈 파일")

    rid = f"LIVE_{uuid.uuid4().hex[:8].upper()}"
    ext = (Path(image.filename or "upload.jpg").suffix or ".jpg").lower()
    if ext not in {".jpg", ".jpeg", ".png"}:
        ext = ".jpg"
    path = UPLOADS_DIR / f"{rid}{ext}"
    path.write_bytes(content)

    try:
        # OBB 탐지 + Stage 2 damage seg → 각 detection 에 damage_ratio 포함
        dets = yolo_service.detect_with_damage(content)
    except Exception as e:
        log.exception("YOLO 실패")
        dets = []
    gps = yolo_service.extract_gps(content)
    # 신고 대상: '파손(Damaged)' + 운영 8 시설(ALLOWED_ITEMS) 만 추천.
    # legacy(보호펜스/무단횡단방지/가로수보호) 는 OBB 가 인식해도 추천 X — 정책 일관성.
    damaged_dets = [d for d in dets if d.get("damaged", True)]
    allowed_dets = [d for d in damaged_dets if d.get("item_ko") in ALLOWED_ITEMS]
    suggested = allowed_dets[0]["item_ko"] if allowed_dets else None
    # 자동 산정 damage_rate (top-1 detection 기준; 실제 산정은 /report 에서 다시)
    suggested_rate = None
    suggested_ratio = None
    if damaged_dets:
        suggested_ratio = float(damaged_dets[0].get("damage_ratio", 0.0))
        suggested_rate = damage_ratio_to_rate(suggested_ratio,
                                              damaged_dets[0].get("item_ko", ""))

    return {
        "id": rid,
        "image_url": f"/uploads/{path.name}",
        "detections": dets,
        "suggested_item": suggested,
        "suggested_damage_rate": suggested_rate,
        "suggested_damage_ratio": suggested_ratio,
        "gps": {"lat": gps[0], "lon": gps[1]} if gps else None,
    }


@router.post("/report")
def report(body: ReportIn):
    """신고 확정 — 학교 거리·위험도 계산 + JSON/Chroma 저장 (user_id 포함)."""
    if not users.get(body.user_id):
        raise HTTPException(401, f"알 수 없는 user_id: {body.user_id}")

    matches = list(UPLOADS_DIR.glob(f"{body.id}.*"))
    # feature 폴더는 제외 (uploads/features/*.npz 가 glob 결과에 들어올 수 있음)
    matches = [p for p in matches if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not matches:
        raise HTTPException(404, f"업로드된 이미지 없음: {body.id}")
    img_path = matches[0]

    # ── damage_rate 서버 자동 산정 (seg 모델) ──
    # 사용자가 확정한 item 과 매칭되는 detection 의 damage_ratio 사용.
    # 매칭 안 되면 top-1 사용. 모두 실패 시 1단계(경미) fallback.
    damage_ratio = 0.0
    damage_rate = 1
    try:
        with open(img_path, "rb") as f:
            content = f.read()
        dets = yolo_service.detect_with_damage(content)
        match = next((d for d in dets
                      if d.get("damaged", True) and d.get("item_ko") == body.item), None)
        if match is None and dets:
            match = next((d for d in dets if d.get("damaged", True)), None)
        if match is not None:
            damage_ratio = float(match.get("damage_ratio", 0.0))
            damage_rate = damage_ratio_to_rate(damage_ratio, body.item)
    except Exception as e:
        log.warning(f"damage 자동 산정 실패 → 1단계 fallback: {e}")

    z = zones.score(body.lat, body.lon)
    n_child = z["n_child"]
    n_elder = z["n_elder"]
    # 새 신고는 elapsed_days=0 (방금 접수). item 도 가중치 적용.
    risk = risk_score(damage_rate, n_child=n_child, n_elder=n_elder,
                      item=body.item, elapsed_days=0)
    emoji, color, label = risk_tier(risk)
    # 가까운 보호구역 합본 (child 우선) — 표시·LLM 컨텍스트용
    nearby_protection = sorted(
        z["nearby_child"] + z["nearby_elder"],
        key=lambda x: x["distance_m"],
    )[:5]
    for it in nearby_protection:
        # group 표기 보강: child / elder
        it.setdefault("group", "child" if it in z["nearby_child"] else "elder")

    # reverse-geocoding (실패 시 None — 치명적 오류 아님)
    addr_info = geocoding.reverse(body.lat, body.lon)

    # ── Feature matching: 5m 이내 + 같은 품목 후보와 ORB 매칭 → Group_ID 할당 ──
    feat = feature_service.extract(img_path)
    feature_path = None
    duplicate_of = None
    match_score = 0
    match_candidates = []
    match_viz = None
    group_id = None

    if feat is not None:
        feature_service.save(body.id, feat)
        feature_path = f"uploads/features/{body.id}.npz"
        existing = storage.list_reports()
        candidates = feature_service.find_nearby_candidates(
            existing, float(body.lat), float(body.lon), body.item
        )
        result = feature_service.resolve_group_id(feat, candidates)
        duplicate_of = result["duplicate_of"]
        match_score = result["match_score"]
        match_candidates = result["match_candidates"]
        if duplicate_of:
            # 기존 후보의 group_id 승계 (None 이면 — 후보가 아직 group_id 없으면 duplicate_of 기준 fallback 은 아래서 처리)
            group_id = result.get("group_id")
            # 매칭 시각화 저장
            best = next((r for r in existing if r["id"] == duplicate_of), None)
            if best:
                viz = feature_service.draw_match(img_path, body.id, best["image_path"], duplicate_of)
                if viz is not None:
                    match_viz = f"uploads/matches/{viz.name}"

    # Group_ID 미정이면 새로 발급 (LIVE_GRP_ prefix 로 기존 Report_DB.csv 와 네임스페이스 분리)
    if group_id is None:
        group_id = f"LIVE_GRP_{uuid.uuid4().hex[:8].upper()}"

    data = {
        "id": body.id,
        "user_id": body.user_id,
        "image_path": str(img_path),
        "item": body.item,
        "damage_rate": int(damage_rate),
        "damage_ratio": float(damage_ratio),
        "description": body.description or "",
        "location": {"lat": float(body.lat), "lon": float(body.lon)},
        "address": addr_info,
        "nearest_zone": z["nearest"],
        "n_child": int(n_child),
        "n_elder": int(n_elder),
        "nearby_protection": nearby_protection,
        "zone_radius_m": int(ZONE_RADIUS_M),
        "risk_score": float(risk),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "pending",
        # feature matching 결과
        "group_id": group_id,
        "feature_path": feature_path,
        "duplicate_of": duplicate_of,
        "match_score": int(match_score),
        "match_candidates": match_candidates,
        "match_viz": match_viz,
    }
    storage.save_report(data)

    # 관리자 SSE 브로드캐스트
    nz = data.get("nearest_zone") or {}
    events.broadcast("report.new", {
        "id": data["id"],
        "item": data["item"],
        "damage_rate": data["damage_rate"],
        "risk_score": data["risk_score"],
        "tier": {"emoji": emoji, "color": color, "label": label},
        "lat": data["location"]["lat"],
        "lon": data["location"]["lon"],
        "address": (addr_info or {}).get("address", ""),
        "n_child": data["n_child"],
        "n_elder": data["n_elder"],
        "nearest_zone_name": nz.get("name", ""),
        "nearest_zone_group": nz.get("group", ""),
        "nearest_zone_distance_m": float(nz.get("distance_m") or 0.0),
        "status": data["status"],
        "created_at": data["created_at"],
        "user_id": data["user_id"],
        "group_id": data["group_id"],
        "duplicate_of": data["duplicate_of"],
        "match_score": data["match_score"],
    })

    # 평균 수리 소요일 (Estimate_DB) — 품목 + 단계별 우선
    item_stats = storage.item_repair_stats().get(body.item) or {}
    by_stage = (item_stats.get("by_stage") or {}).get(int(damage_rate)) or {}
    avg_days = by_stage.get("avg_days") or item_stats.get("avg_days")

    return {
        **data,
        "image_url": f"/uploads/{img_path.name}",
        "tier": {"emoji": emoji, "color": color, "label": label},
        "avg_repair_days": avg_days,
    }


@router.get("/my-reports")
def my_reports(user_id: str):
    """본인 신고 목록 (최신순)."""
    mine = [r for r in storage.list_reports() if r.get("user_id") == user_id]
    mine.sort(key=lambda r: r["created_at"], reverse=True)
    # 이미지 URL 편의 추가
    for r in mine:
        img = Path(r["image_path"])
        r["image_url"] = f"/uploads/{img.name}"
        _, color, label = risk_tier(r["risk_score"])
        r["tier_label"] = label
    return {"user_id": user_id, "count": len(mine), "reports": mine}


@router.get("/status/{report_id}")
def status(report_id: str, user_id: Optional[str] = None):
    """특정 신고 상태 조회 — 본인 신고만. user_id 쿼리 필수."""
    for r in storage.list_reports():
        if r["id"] == report_id:
            if user_id and r.get("user_id") != user_id:
                raise HTTPException(403, "본인 신고만 조회 가능합니다.")
            return r
    raise HTTPException(404, f"신고 없음: {report_id}")


_ID_RE = re.compile(r"LIVE_[A-F0-9]{8}", re.I)


@router.post("/chat")
def chat(body: ChatIn):
    """민원인 LLM 챗봇 — 본인 신고 자료만 컨텍스트로 제공."""
    if not users.get(body.user_id):
        raise HTTPException(401, f"알 수 없는 user_id: {body.user_id}")

    # 본인 신고만
    mine = [r for r in storage.list_reports() if r.get("user_id") == body.user_id]

    # 메시지에서 LIVE_XXXXXXXX 직접 언급 시 해당 신고 우선 (본인 것만)
    id_match = _ID_RE.search(body.message.upper())
    target = []
    if id_match:
        rid = id_match.group(0).upper()
        target = [r for r in mine if r["id"] == rid]

    # 없으면 본인 최근 전체
    if not target:
        target = sorted(mine, key=lambda r: r["created_at"], reverse=True)[:30]

    user_info = users.get(body.user_id)
    user_name = user_info["name"] if user_info else body.user_id

    # 처리 시간 컨텍스트 — 📌 평균 지연일(실측 30일) + 각 신고별 예상 완료 시각
    def _parse_iso(s):
        if not s: return None
        try: return datetime.fromisoformat(s.split("+")[0].replace("Z", ""))
        except Exception: return None

    now = datetime.now()
    all_reports = storage.list_reports()
    deltas = []
    for r in all_reports:
        if r.get("status") != "completed": continue
        c = _parse_iso(r.get("created_at"))
        u = _parse_iso(r.get("completed_at") or r.get("updated_at"))
        if not c or not u: continue
        if (now - u).days > 30: continue
        deltas.append((u - c).total_seconds() / 86400.0)
    avg_delay_30d = round(sum(deltas) / len(deltas), 2) if deltas else None

    if target:
        # 품목별 + 단계별 평균 소요일 (Estimate_DB) 기반 예상 완료
        item_stats_all = storage.item_repair_stats()
        STATUS_KO = {"pending": "접수됨", "in_progress": "처리중",
                     "completed": "처리완료", "rejected": "반려"}
        def _expected(r):
            c = _parse_iso(r.get("created_at"))
            if not c: return "미정"
            s = item_stats_all.get(r.get("item")) or {}
            by_stage = (s.get("by_stage") or {}).get(int(r.get("damage_rate") or 0)) or {}
            avg = by_stage.get("avg_days") or s.get("avg_days")
            if not avg: return "산출 불가 (견적 데이터 없음)"
            return (c + timedelta(days=float(avg))).strftime("%Y-%m-%d") + f" (평균 {avg}일)"
        def _zone_desc(r):
            nz = r.get("nearest_zone") or {}
            nm = nz.get("name") or "-"
            dist = int(nz.get("distance_m") or 0)
            nc = r.get("n_child", 0); ne = r.get("n_elder", 0)
            return f"보호구역 어린이{nc}·노인{ne}, 최근접 {nm} {dist}m"
        ctx = "\n".join(
            f"- {r['id']}: {r['item']} 손상도 {r['damage_rate']} "
            f"({_zone_desc(r)} · 위험도 {r['risk_score']}), "
            f"상태 **{STATUS_KO.get(r['status'], r['status'])}**, 접수 {r['created_at']}, "
            f"예상 완료 {_expected(r)}"
            for r in target
        )
    else:
        ctx = "(아직 본인 신고 내역이 없습니다)"

    delay_line = (
        f"최근 30일 완료건 평균 처리 기간: {avg_delay_30d}일 (실측, 표본 {len(deltas)}건)"
        if avg_delay_30d is not None else
        "최근 30일 완료 데이터 부족 — 시스템 기본 룰로 안내하세요."
    )

    # 품목별 평균 수리 소요 일수 (Estimate_DB 기준)
    item_stats = storage.item_repair_stats()
    if item_stats:
        item_lines = "\n".join(
            f"- {it}: 평균 {s['avg_days']}일 (중앙값 {s['median_days']}일, 표본 {s['n']}건)"
            for it, s in sorted(item_stats.items())
        )
        # 데이터 없는 품목 명시 (볼라드 등)
        missing = sorted(set(__import__('config').ALLOWED_ITEMS) - set(item_stats.keys()))
        if missing:
            item_lines += f"\n- (데이터 부족) {', '.join(missing)} — 일반적으로 2~4주 안내"
    else:
        item_lines = "(품목별 통계 데이터 없음)"

    system = (
        f"당신은 전국 공공시설 파손 신고 안내 챗봇입니다. 한국어로 친근하고 간결하게 답하세요.\n"
        f"로그인한 사용자: {user_name} ({body.user_id}).\n\n"
        "규칙:\n"
        "1) [신고 자료]는 이 사용자 본인 신고만 포함합니다. 타인 신고는 알 수 없습니다.\n"
        "2) [신고 자료]에 없는 내용은 추측하지 말고 '확인이 어렵습니다'라고 답하세요.\n"
        "3) 어떤 신고인지 특정 안 되면 신고번호(LIVE_XXXXXXXX)를 물어보세요.\n"
        "4) 새 신고는 화면의 📷 버튼으로 사진을 올리도록 안내하세요.\n"
        "5) 상태는 [신고 자료]에 한글로 적힌 그대로 사용 (접수됨/처리중/처리완료/반려). 영문 코드(pending 등) 노출 금지.\n"
        "6) '목록', '전체', '다른 신고', '모든 신고' 요청 시 [신고 자료]의 각 LIVE_XXXXXXXX · 품목 · 상태를 번호 매겨 전부 나열하세요. 회피 금지.\n\n"
        "[처리 시간 안내 가이드라인]\n"
        "- '언제 완료', '얼마나 걸려', '평균', '소요', '처리 기간' 류 질문엔 반드시 아래 정보로 답변하세요.\n"
        "- '알 수 없다' / '담당 부서에 문의' 로 회피하지 마세요. 아래 수치를 근거로 제시하세요.\n"
        "- 시스템 기본 처리 룰 (자동 상태 진행): 접수→검토중 1시간, →처리중 4시간, →처리완료 **접수 후 약 8시간(≈0.33일)**.\n"
        f"- {delay_line}\n"
        "- 개별 신고가 특정되면 자료의 '예상 완료' 필드를 그대로 안내하세요 (예: 'OOOO-MM-DD HH:MM 경 완료 예정').\n"
        "- 이미 completed 상태면 '처리 완료되었습니다'로 답하세요.\n\n"
        "[품목별 평균 수리 소요 일수 — 견적 DB 통계]\n"
        f"{item_lines}\n"
        "- '볼라드 보통 얼마나 걸려', '맨홀 수리 며칠?' 같은 **품목별 처리 시간 질문은 위 통계를 그대로 인용해** 답하세요.\n"
        "- 형식: '평균 약 X일 소요됩니다 (과거 견적 N건 기준)'. 가능하면 중앙값도 함께 안내.\n"
        "- 위 표에 없는 품목(데이터 부족)이면 '정확한 통계는 부족하나 일반적으로 2~4주 정도 소요됩니다' 라고 안내.\n\n"
        f"[{user_name} 님의 신고 자료]\n{ctx}"
    )

    msgs = [{"role": "system", "content": system}]
    for h in body.history[-10:]:
        if h.role in ("user", "assistant"):
            msgs.append({"role": h.role, "content": h.text})
    msgs.append({"role": "user", "content": body.message})

    reply = llm.chat(msgs, model=LLM_MODEL_CITIZEN)
    turn_id = chat_logs.save_turn(
        user_id=body.user_id,
        user_name=user_name,
        message=body.message,
        reply=reply,
    )
    return {"reply": reply, "turn_id": turn_id}


@router.post("/feedback")
def feedback(body: FeedbackIn):
    """챗봇 답변 👍/👎 — note 는 '왜 잘못됐나' 자유 입력."""
    ok = chat_logs.set_feedback(body.turn_id, body.rating, body.note)
    if not ok:
        raise HTTPException(404, "해당 turn_id를 찾을 수 없습니다 (rating 오입력 가능).")
    return {"ok": True}
