"""관리자 API — 지도·목록·검색·견적·보고서·상태 업데이트."""
import json
import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

from datetime import date, datetime
from typing import List

from config import risk_tier, LIVE_REPORTS, LLM_MODEL_ADMIN, ALLOWED_ITEMS
from services import storage, rag, llm, chat_logs, admin_auth, events, emergency

log = logging.getLogger(__name__)
router = APIRouter()


class SearchIn(BaseModel):
    query: str
    k: int = 8
    source: Optional[str] = None   # est / rep / emg / live
    include_rejected: bool = False


class EstimateIn(BaseModel):
    query: str
    k: int = 8
    item: Optional[str] = None
    damage_rate: Optional[int] = None


class BriefIn(BaseModel):
    group_id: str
    k: int = 6


class ReportIn(BaseModel):
    report_id: str   # REP_XXXXX (historical) 또는 LIVE_XXXXXXXX (live)
    model: Optional[str] = None


class StatusIn(BaseModel):
    status: str
    cascade: bool = False   # True: 같은 group_id 의 active 멤버 일괄 변경


class AdminChatTurn(BaseModel):
    role: str
    text: str


class AdminChatIn(BaseModel):
    message: str
    history: List[AdminChatTurn] = []


class AdminLoginIn(BaseModel):
    username: str
    password: str


@router.get("/ping")
def ping():
    return {"ok": True, "scope": "admin"}


@router.post("/login")
def admin_login(body: AdminLoginIn):
    """관리자 로그인 — admin_users.json 대조. 성공 시 세션 토큰 반환."""
    user = admin_auth.verify_login(body.username.strip(), body.password)
    if not user:
        raise HTTPException(401, "아이디 또는 비밀번호가 일치하지 않습니다.")
    token = admin_auth.create_session(user["username"])
    return {"token": token, **user}


@router.post("/logout")
def admin_logout(request: Request):
    """현재 토큰 무효화."""
    admin_auth.logout(request.headers.get("x-admin-token", ""))
    return {"ok": True}


@router.get("/me")
def admin_me(request: Request):
    """현재 로그인 상태 확인."""
    token = request.headers.get("x-admin-token", "")
    username = admin_auth.verify_token(token)
    return {"authenticated": bool(username), "username": username}


@router.get("/geocoding-status")
def geocoding_status():
    """GeoJSON 로드 상태 (디버그용)."""
    from services import geocoding
    return geocoding.status()


@router.get("/stream")
async def stream(request: Request, token: str = ""):
    """관리자 SSE 스트림. EventSource는 헤더 불가 → 쿼리 토큰으로 인증."""
    if not admin_auth.verify_token(token):
        return Response("관리자 인증 필요", status_code=401)

    async def gen():
        async for msg in events.subscribe():
            if await request.is_disconnected():
                break
            yield msg

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # nginx/proxy 버퍼링 방지
            "Connection": "keep-alive",
        },
    )


@router.get("/risk-map")
def risk_map(request: Request):
    """Live 신고들을 지도 마커용 + 통계로 반환 — 관리자 region 기준 필터."""
    _user = admin_auth.get_user_by_token(request.headers.get("x-admin-token", ""))
    _region = (_user or {}).get("region") or {}
    reports = _filter_by_region(storage.list_reports(), _region)
    out = []
    for r in reports:
        emoji, color, label = risk_tier(r["risk_score"])
        addr = r.get("address") or {}
        nz = r.get("nearest_zone") or {}
        out.append({
            "id": r["id"],
            "item": r["item"],
            "damage_rate": r["damage_rate"],
            "lat": r["location"]["lat"],
            "lon": r["location"]["lon"],
            "address": addr.get("address", ""),
            "sido": addr.get("sido", ""),
            "sigungu": addr.get("sigungu", ""),
            "emd": addr.get("emd", ""),
            "n_child": int(r.get("n_child", 0)),
            "n_elder": int(r.get("n_elder", 0)),
            "nearest_zone_name":     nz.get("name", ""),
            "nearest_zone_group":    nz.get("group", ""),
            "nearest_zone_kind":     nz.get("kind", ""),
            "nearest_zone_distance_m": float(nz.get("distance_m") or 0.0),
            "nearby_protection":     r.get("nearby_protection") or [],
            "zone_radius_m":         int(r.get("zone_radius_m", 300)),
            "risk_score": r["risk_score"],
            "tier": {"emoji": emoji, "color": color, "label": label},
            "status": r["status"],
            "created_at": r["created_at"],
            "image_url": f"/uploads/{Path(r['image_path']).name}",
            "group_id": r.get("group_id"),
            "duplicate_of": r.get("duplicate_of"),
            "match_score": r.get("match_score") or 0,
            "match_viz": r.get("match_viz"),
        })
    # 통계
    n = len(out)
    def count_tier(lo, hi=None):
        return sum(1 for r in out if r["risk_score"] >= lo and (hi is None or r["risk_score"] < hi))

    # 오늘 신규 민원
    today_str = date.today().isoformat()
    today_new = sum(1 for r in out if (r["created_at"] or "")[:10] == today_str)

    # 평균 처리 지연일 — completed + updated_at 있는 건만 실측
    def _parse(ts):
        if not ts: return None
        try: return datetime.fromisoformat(ts.replace("Z", "+00:00").split("+")[0])
        except Exception: return None

    now = datetime.now()
    def avg_delay_days(window_days: int):
        deltas = []
        for r in reports:  # 원본 reports (updated_at 포함)
            if r.get("status") != "completed": continue
            c = _parse(r.get("created_at"))
            u = _parse(r.get("completed_at") or r.get("updated_at"))
            if not c or not u: continue
            if (now - u).days > window_days: continue
            deltas.append((u - c).total_seconds() / 86400.0)
        if not deltas: return None
        return round(sum(deltas) / len(deltas), 2)

    stats = {
        "total": n,
        # 새 1/2/3 식 (객체위험계수 0.7~1.0, 분포 0.7~3.0) — risk_tier 와 동일
        "critical": count_tier(2.4),
        "high":     count_tier(1.9, 2.4),
        "medium":   count_tier(1.3, 1.9),
        "low":      count_tier(0, 1.3),
        "pending":  sum(1 for r in out if r["status"] == "pending"),
        "today_new": today_new,
        "avg_delay_30d":  avg_delay_days(30),
        "avg_delay_365d": avg_delay_days(365),
    }
    return {"stats": stats, "reports": out}


@router.get("/reports")
def reports_list(
    request: Request,
    page: int = 1, limit: int = 20,
    item: Optional[str] = None,
    status: Optional[str] = None,
    min_risk: Optional[float] = None,
):
    """Live 신고 목록 — 위험도 내림차순, 필터·페이지네이션. 관리자 region 기준으로 선필터."""
    _user = admin_auth.get_user_by_token(request.headers.get("x-admin-token", ""))
    _region = (_user or {}).get("region") or {}
    reports = _filter_by_region(list(storage.list_reports()), _region)
    reports.sort(key=lambda r: -r["risk_score"])
    if item:   reports = [r for r in reports if r["item"] == item]
    # 상태 필터: "active" = pending + in_progress (기본). 그 외엔 정확 매칭.
    if status == "active":
        reports = [r for r in reports if r.get("status") in ("pending", "in_progress")]
    elif status:
        reports = [r for r in reports if r["status"] == status]
    if min_risk is not None:
        reports = [r for r in reports if r["risk_score"] >= float(min_risk)]
    total = len(reports)
    start = (max(page, 1) - 1) * limit
    items = reports[start:start + limit]
    for r in items:
        emoji, color, label = risk_tier(r["risk_score"])
        r["tier"] = {"emoji": emoji, "color": color, "label": label}
        r["image_url"] = f"/uploads/{Path(r['image_path']).name}"
        # group/matching 정보 (이미 dict 에 있으면 패스; 구 버전 대응)
        r.setdefault("group_id", None)
        r.setdefault("duplicate_of", None)
        r.setdefault("match_score", 0)
        r.setdefault("match_viz", None)
    return {"total": total, "page": page, "limit": limit, "items": items}


@router.get("/emergency")
def emergency_list(request: Request):
    """긴급도 DB 뷰 — Group_ID 단위 집계. status != completed 만 포함.
    같은 물품(feature matching) 으로 묶인 신고들을 그룹으로 Final_Score 내림차순 반환."""
    _user = admin_auth.get_user_by_token(request.headers.get("x-admin-token", ""))
    _region = (_user or {}).get("region") or {}
    reports = _filter_by_region(storage.list_reports(), _region)
    groups = emergency.aggregate(reports)
    # 대표 이미지 URL 편의 필드
    by_id = {r["id"]: r for r in reports}
    for g in groups:
        rep = by_id.get(g["representative_id"], {})
        img = rep.get("image_path", "")
        if img:
            g["representative_image_url"] = f"/uploads/{Path(img).name}"
        # 멤버 요약
        g["members"] = [
            {
                "id": m["id"],
                "damage_rate": m.get("damage_rate"),
                "risk_score": m.get("risk_score"),
                "status": m.get("status"),
                "created_at": m.get("created_at"),
                "duplicate_of": m.get("duplicate_of"),
                "match_score": m.get("match_score"),
                "match_viz": m.get("match_viz"),
                "image_url": f"/uploads/{Path(m['image_path']).name}" if m.get("image_path") else None,
            }
            for m in (by_id.get(mid) for mid in g["member_ids"]) if m
        ]
    return {"total": len(groups), "groups": groups}


@router.get("/catalog")
def catalog(request: Request):
    """드롭다운용 — 품목, Group_ID, 최근 LIVE_ID 목록.

    region scope 적용 — 긴급도DB 와 일치시키기 위해 같은 필터 사용.
    """
    _user = admin_auth.get_user_by_token(request.headers.get("x-admin-token", ""))
    _region = (_user or {}).get("region") or {}
    col = storage.collection()
    # 품목: 운영 중인 8종 화이트리스트 — Chroma est 데이터 유무와 무관하게 노출
    items = sorted(ALLOWED_ITEMS)
    # Group_ID: rep 소스 distinct
    rep = col.get(where={"source": "rep"}, include=["metadatas"])
    group_ids = sorted({m.get("Group_ID", "") for m in (rep.get("metadatas") or []) if m.get("Group_ID")})
    # 최근 LIVE — region 필터 + 활성만 (pending/in_progress) — UI 라벨이 "활성 신고 ID"
    # 긴급도DB도 status != completed 만 → 일치 시킴
    live_filtered = _filter_by_region(storage.list_reports(), _region)
    ACTIVE = {"pending", "in_progress"}
    live_active = [r for r in live_filtered if r.get("status") in ACTIVE]
    live_sorted = sorted(live_active, key=lambda r: r.get("created_at", ""), reverse=True)[:50]
    live_ids = [{
        "id": r["id"],
        "label": f'{r["id"]} — {r.get("item","-")} 손상도 {r.get("damage_rate","-")} · {(r.get("created_at") or "")[:10]}'
    } for r in live_sorted]
    return {"items": items, "group_ids": group_ids, "live_ids": live_ids}


@router.post("/search")
def search(body: SearchIn):
    return {"results": rag.search(body.query, k=body.k, source=body.source,
                                   include_rejected=body.include_rejected)}


@router.post("/estimate")
def estimate(body: EstimateIn):
    return rag.estimate(body.query, k=body.k, item=body.item, damage_rate=body.damage_rate)


@router.post("/brief")
def brief(body: BriefIn):
    return rag.brief(body.group_id, k=body.k)


@router.post("/report")
def report(body: ReportIn):
    """민원 처리계획 보고서 (공문 양식 7섹션). REP_XXXXX(historical) 또는 LIVE_XXXXXXXX 받음."""
    return rag.report(body.report_id, model=body.model)


def _filter_by_region(reports: list, region: Optional[dict]) -> list:
    """관리자 region scope 에 맞춰 live_reports 필터링.
    scope=all → 그대로, sido → address.sido 일치, sigungu_prefix → address.sigungu 가 match 로 시작."""
    if not region or region.get("scope") in (None, "all"):
        return reports
    scope, match = region.get("scope"), region.get("match", "")
    out = []
    for r in reports:
        addr = r.get("address") or {}
        if scope == "sido" and addr.get("sido") == match:
            out.append(r)
        elif scope == "sigungu_prefix" and (addr.get("sigungu") or "").startswith(match):
            out.append(r)
    return out


def _build_admin_context(region: Optional[dict] = None) -> str:
    """현재 live 신고 + 통계를 LLM 컨텍스트 문자열로 변환. region 이 있으면 해당 지역만."""
    reports = _filter_by_region(storage.list_reports(), region)
    today = date.today().isoformat()

    # 통계
    total = len(reports)
    by_status = {}
    by_tier = {"매우 위험": 0, "위험": 0, "주의": 0, "낮음": 0}
    today_cnt = 0
    for r in reports:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        _, _, label = risk_tier(r["risk_score"])
        by_tier[label] = by_tier.get(label, 0) + 1
        if r["created_at"][:10] == today:
            today_cnt += 1

    # 위험도 내림차순
    reports_sorted = sorted(reports, key=lambda r: -r["risk_score"])

    lines = [f"오늘 날짜: {today}", f"총 신고: {total}건", f"오늘 접수: {today_cnt}건"]
    lines.append("상태별: " + ", ".join(f"{k} {v}건" for k, v in by_status.items()))
    lines.append("위험도별: " + ", ".join(f"{k} {v}건" for k, v in by_tier.items()))
    lines.append("")
    lines.append("[신고 목록 — 위험도 내림차순]")
    if not reports_sorted:
        lines.append("(신고 없음)")
    for r in reports_sorted[:30]:
        _, _, tlabel = risk_tier(r["risk_score"])
        nc = int(r.get("n_child", 0)); ne = int(r.get("n_elder", 0))
        nz = r.get("nearest_zone") or {}
        nz_name = nz.get("name") or "-"
        nz_dist = int(nz.get("distance_m") or 0)
        addr = (r.get("address") or {}).get("address", "")
        addr_part = f"[{addr}] " if addr else ""
        lines.append(
            f"- {r['id']}: {addr_part}{r['item']} 손상도{r['damage_rate']} "
            f"위험도 {r['risk_score']}({tlabel}) 상태 {r['status']} · "
            f"보호구역(300m) 어린이{nc}·노인{ne} · "
            f"최근접 {nz_name}({nz_dist}m) · 접수 {r['created_at']}"
        )
    if len(reports_sorted) > 30:
        lines.append(f"... 외 {len(reports_sorted)-30}건")

    # 품목별 + 단계별 과거 견적 통계 (Estimate_DB) — 수리비·소요일 질문 대응
    # OBB 인식 11 시설 전체 (운영 8 + 보호펜스/무단횡단방지봉/가로수보호덮개) 통계 포함
    item_stats = storage.item_repair_stats()
    if item_stats:
        lines.append("")
        lines.append("[품목별 과거 견적 통계 — Estimate_DB]")
        for it in sorted(item_stats.keys()):
            s = item_stats.get(it)
            if s:
                cost_part = f"평균 {s['avg_cost']:,}원 (중앙 {s['median_cost']:,})" if s.get("avg_cost") else "비용 정보 없음"
                days_part = f"평균 {s['avg_days']}일 (중앙 {s['median_days']}일)" if s.get("avg_days") else "소요일 정보 없음"
                lines.append(f"- {it}: {cost_part}, {days_part}, 표본 {s['n']}건")
                # 단계별 분해
                by_stage = s.get("by_stage") or {}
                for stage in (1, 2, 3):
                    ss = by_stage.get(stage)
                    if ss:
                        s_cost = f"평균 {ss['avg_cost']:,}원" if ss.get("avg_cost") else "비용 없음"
                        s_days = f"평균 {ss['avg_days']}일" if ss.get("avg_days") else "소요일 없음"
                        lines.append(f"    └ {stage}단계: {s_cost}, {s_days}, 표본 {ss['n']}건")
            else:
                lines.append(f"- {it}: (견적 DB 데이터 없음)")
    return "\n".join(lines)


@router.post("/chat")
def admin_chat(body: AdminChatIn, request: Request):
    """관리자 챗봇.
    - 로그인한 관리자 region 기준으로 [신고 자료] 필터 (예: pohang → 포항시남/북구만).
    - 'REP_XXXXX + 보고서/브리핑' 요청이면 rag.report() 로 우회 → 공문 양식 보고서 반환(gemma4:e4b).
    """
    # 1) 관리자 region 조회 (token → user → region)
    token = request.headers.get("x-admin-token", "")
    user = admin_auth.get_user_by_token(token)
    region = (user or {}).get("region") or {}
    region_label = region.get("label") or "전체"

    # 2) 일반 챗봇 — region 필터된 컨텍스트 + 담당 지역 명시
    # (자동보고서는 하단 "📑 자동 보고서" 섹션에서 /api/admin/brief 로 별도 처리)
    ctx = _build_admin_context(region)
    system = (
        f"당신은 시설관리 대시보드의 운영자 도우미입니다. 담당 지역: **{region_label}**. "
        f"관리자가 신고 현황·우선순위·처리 판단을 물으면 [신고 자료]만 근거로 답하세요. "
        f"[신고 자료]에는 담당 지역 신고만 들어 있습니다 — 다른 지역 요청이 와도 '담당 지역({region_label})만 조회 가능'이라 안내하세요.\n"
        "규칙:\n"
        "1) 숫자·날짜·ID는 [신고 자료]를 그대로 인용. 추측 금지.\n"
        "1-1) 카운팅(예: '포항시 신고 몇 건', '서울 위험 등급 몇 건', '품목별 분포')은 [신고 자료] 의 줄을 직접 한 줄씩 세어서 답하세요. 어림짐작·반올림 금지. 숫자 답변 끝에 '([신고 자료] 직접 카운트)' 라고 표기하세요.\n"
        "2) 위험도 기준: 2.4↑ 매우 위험, 1.9↑ 위험, 1.3↑ 주의, 미만 낮음. "
        "(긴급도 = 객체위험계수 × [위치×0.3 + 빈도×0.2 + 파손×0.3 + 방치×0.2], 각 항 1/2/3 점수)\n"
        "3) 상태 의미: pending=접수/검토전(=한글 '신고접수'), in_progress=처리중(=한글 '처리진행중'), completed=완료(=한글 '처리완료'), rejected=반려. 한글·영문 상태는 동일하게 해석하고 답변은 영문 코드로 통일하세요.\n"
        "4) 한국어로 간결하게. 목록이 길면 상위 5~10건만 요약해 보여주세요.\n"
        "5) 수리비·소요일 질문은 **[품목별 과거 견적 통계]** 블록의 수치를 그대로 인용해 답하세요. '평균 X원 / Y일 (N건 기반)' 형식. '과거 견적 DB 조회 필요' 라는 회피 답변 금지.\n"
        "5-1) 품목명 표기가 달라도 같은 품목으로 취급: '보도블럭'=보도블록, '점자블럭'=점자블록, '맨홀뚜껑'=맨홀, '벤치'(등받이 언급 없으면)=등받이있는벤치로 우선 매칭.\n"
        "5-2) '유사 신고 사례 + 수리비/소요일' 같은 **복합 질문**은 2부분 모두 답하세요: (a) [신고 자료]의 같은 품목 live 신고 먼저 번호 매겨 나열 (b) 그 다음 [품목별 과거 견적 통계] 인용. 견적 통계가 없는 품목이면 (a)만 답하고 '견적 DB 데이터 없음' 만 별도 안내. (a)를 빠뜨리지 마세요.\n"
        "6) 'REP_XXXXX 보고서' 류 요청은 공문 양식(1.민원개요 / 2.가.파손내용 나.과거 수리 사례 다.예상 수리비)으로 작성.\n\n"
        f"[신고 자료 — {region_label}]\n{ctx}"
    )
    msgs = [{"role": "system", "content": system}]
    for h in body.history[-10:]:
        if h.role in ("user", "assistant"):
            msgs.append({"role": h.role, "content": h.text})
    msgs.append({"role": "user", "content": body.message})
    reply = llm.chat(msgs, model=LLM_MODEL_ADMIN)
    return {"reply": reply}


@router.get("/chat-logs")
def chat_logs_list(
    user_id: Optional[str] = None,
    rating: Optional[str] = None,       # good | bad | none
    date: Optional[str] = None,         # YYYY-MM-DD
    limit: int = 100,
    offset: int = 0,
):
    return chat_logs.list_logs(user_id=user_id, rating=rating, date=date,
                                limit=limit, offset=offset)


@router.get("/chat-logs/stats")
def chat_logs_stats():
    return chat_logs.stats()


@router.post("/reports/{rid}/status")
def update_status(rid: str, body: StatusIn):
    """Live 신고 상태 업데이트 — JSON + Chroma 메타 동기화.

    - completed 는 변경 불가 (종결).
    - rejected 는 변경 가능 (잘못 반려 회복용).
    - cascade=True 면 같은 group_id 의 **active(pending/in_progress) 멤버 일괄** 변경
      (이미 completed 된 멤버는 보존; 같은 status 인 멤버는 no-op skip).
    """
    allowed = {"pending", "in_progress", "completed", "rejected"}
    if body.status not in allowed:
        raise HTTPException(400, f"상태는 {allowed} 중 하나여야 합니다.")
    reports = storage.list_reports()
    target = next((r for r in reports if r["id"] == rid), None)
    if not target:
        raise HTTPException(404, f"신고 없음: {rid}")
    cur = target.get("status")
    if cur == "completed":
        raise HTTPException(409, "처리 완료된 신고는 상태를 변경할 수 없습니다.")

    # 변경 대상 결정 — single 또는 cascade
    targets = [target]
    if body.cascade and target.get("group_id"):
        gid = target["group_id"]
        for r in reports:
            if r is target:
                continue
            if r.get("group_id") != gid:
                continue
            if r.get("status") in ("completed",):  # 종결된 건 보존
                continue
            if r.get("status") == body.status:     # 이미 같은 상태
                continue
            targets.append(r)

    now_iso = datetime.now().isoformat(timespec="seconds")
    changed_ids = []
    for t in targets:
        if t.get("status") == body.status:
            continue
        t["status"] = body.status
        t["updated_at"] = now_iso
        if body.status == "completed":
            t["completed_at"] = now_iso
        changed_ids.append(t["id"])

    if not changed_ids:
        return {"ok": True, "id": rid, "status": cur, "noop": True}

    LIVE_REPORTS.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        col = storage.collection()
        for cid in changed_ids:
            col.update(ids=[cid], metadatas=[{"status": body.status}])
    except Exception as e:
        log.warning(f"Chroma 메타 업데이트 실패: {e}")

    for cid in changed_ids:
        events.broadcast("report.updated", {"id": cid, "status": body.status})
    return {"ok": True, "id": rid, "status": body.status,
            "changed": changed_ids, "cascade_count": len(changed_ids)}
