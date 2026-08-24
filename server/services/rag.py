"""관리자용 RAG 서비스 — 의미 검색·견적·그룹 보고서·민원처리 보고서.

dashboard.py 의 estimate_cost / group_brief 를 그대로 이식.
report() 는 rag_report.py (민원처리 계획 보고서, 공공기관 공문 양식) 포팅.
"""
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import PARENT_DIR, ITEM_NORMALIZE
from services import storage, llm, geocoding

log = logging.getLogger(__name__)


def _cost_stats(costs):
    arr = np.array(costs, dtype=float)
    return {
        "n": len(arr),
        "min": int(arr.min()), "max": int(arr.max()),
        "median": int(np.median(arr)),
        "q25": int(np.percentile(arr, 25)),
        "q75": int(np.percentile(arr, 75)),
        "mean": int(arr.mean()), "std": int(arr.std()),
    }


def _where(conds):
    conds = [c for c in conds if c]
    if not conds: return None
    if len(conds) == 1: return conds[0]
    return {"$and": conds}


def _embed(q: str):
    return storage._embedder().encode([q]).tolist()


def search(query: str, k: int = 8, source: Optional[str] = None,
           include_rejected: bool = False):
    """의미 검색. 기본값은 반려(rejected) live 신고 제외."""
    col = storage.collection()
    conds = []
    if source:
        conds.append({"source": source})
    if not include_rejected:
        # live 소스에만 status가 있으므로, live이면서 rejected인 것만 제외
        conds.append({"$or": [
            {"source": {"$ne": "live"}},
            {"status": {"$ne": "rejected"}},
        ]})
    where = _where(conds)
    hits = col.query(query_embeddings=_embed(query), n_results=k, where=where)
    out = []
    for doc, meta, dist in zip(hits["documents"][0], hits["metadatas"][0], hits["distances"][0]):
        out.append({
            "document": doc,
            "metadata": meta,
            "similarity": round(1 - float(dist), 3),
        })
    return out


def estimate(query: str, k: int = 8, item: Optional[str] = None,
             damage_rate: Optional[int] = None) -> dict:
    col = storage.collection()
    conds = [{"source": "est"}]
    if item: conds.append({"Item": item})
    if damage_rate is not None: conds.append({"Damage_Rate": int(damage_rate)})
    hits = col.query(query_embeddings=_embed(query), n_results=k, where=_where(conds))
    docs = hits["documents"][0]
    metas = hits["metadatas"][0]
    if not docs:
        return {"reply": "유사 견적 사례 없음. 필터를 완화해보세요.",
                "stats": None, "cases": [], "top_companies": {}}
    costs = [m["Cost"] for m in metas]
    companies = [m["Company_Info"] for m in metas]
    stats = _cost_stats(costs)
    top_companies = {k_: int(v) for k_, v in pd.Series(companies).value_counts().head(3).items()}

    ctx = "\n".join(f"- {d}" for d in docs)
    prompt = f"""당신은 공공시설 수리 견적 분석관입니다.
아래 '유사 과거 사례'와 '사전 계산된 통계'만 근거로 견적을 산정하세요. 외부 지식 사용 금지.
수치는 통계 블록의 값을 그대로 인용하세요(재계산 금지).

[유사 사례 {len(docs)}건]
{ctx}

[통계(원)]
- 표본 수: {stats['n']}
- 최저 ~ 최고: {stats['min']:,} ~ {stats['max']:,}
- 25 ~ 75% 범위: {stats['q25']:,} ~ {stats['q75']:,}
- 중앙값: {stats['median']:,}
- 평균 ± 표준편차: {stats['mean']:,} ± {stats['std']:,}

[업체 빈도 상위] {top_companies}

[질문] {query}

[출력 형식]
1) 권장 견적 범위(25~75% 분위)
2) 중앙값 기준 추정
3) 근거 사례 요약(상위 3건)
4) 추천 업체(빈도 상위)
5) 신뢰도 코멘트(n={stats['n']}, 편차 수준)
"""
    reply = llm.chat([{"role": "user", "content": prompt}])
    return {"reply": reply, "stats": stats, "cases": docs, "top_companies": top_companies}


def brief(group_id: str, k: int = 6) -> dict:
    col = storage.collection()
    rep = col.get(where={"$and": [{"source": "rep"}, {"Group_ID": group_id}]})
    emg = col.get(where={"$and": [{"source": "emg"}, {"Group_ID": group_id}]})
    rep_items = [m.get("Item", "") for m in (rep.get("metadatas") or [])]

    # rep 에 group_id 데이터 없으면 (신규 LIVE_GRP_xxx 케이스) live 컬렉션에서 같은 group_id 의 item 가져오기
    if not rep_items:
        live = col.get(where={"$and": [{"source": "live"}, {"Group_ID": group_id}]})
        live_items = [m.get("Item", "") for m in (live.get("metadatas") or [])]
        rep_items = live_items   # 표/통계는 live 데이터로 대체

    unique_items = sorted(set(i for i in rep_items if i))
    query = " ".join(unique_items) or group_id

    # 1) 같은 품목 필터로 검색 (하드 메타 조건)
    where = {"source": "est"}
    if unique_items:
        item_cond = ({"Item": unique_items[0]} if len(unique_items) == 1
                     else {"Item": {"$in": unique_items}})
        where = {"$and": [{"source": "est"}, item_cond]}
    est_hits = col.query(query_embeddings=_embed(query), n_results=max(k*2, k),
                         where=where)
    est_docs  = est_hits["documents"][0] if est_hits["documents"] else []
    est_metas = est_hits["metadatas"][0] if est_hits["metadatas"] else []

    # 2) 품목 필터로 0건이면 품목 제약 풀어서 재검색 (fallback)
    if not est_metas:
        est_hits = col.query(query_embeddings=_embed(query), n_results=k,
                             where={"source": "est"})
        est_docs  = est_hits["documents"][0] if est_hits["documents"] else []
        est_metas = est_hits["metadatas"][0] if est_hits["metadatas"] else []

    # 3) Report_Date 내림차순 재정렬 (최근 사례 우선)
    if est_metas:
        paired = list(zip(est_metas, est_docs))
        paired.sort(key=lambda p: p[0].get("Report_Date", ""), reverse=True)
        paired = paired[:k]
        est_metas = [p[0] for p in paired]
        est_docs  = [p[1] for p in paired]

    costs = [m["Cost"] for m in est_metas] if est_metas else []

    stats = _cost_stats(costs) if costs else None

    # 유사 사례 비교표 (기호 A~): 감정평가서 "동일 유사 물건의 사례" 양식
    symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    similar_cases = []
    for i, m in enumerate(est_metas[:5]):
        similar_cases.append({
            "symbol": symbols[i] if i < len(symbols) else str(i + 1),
            "item": m.get("Item", "-"),
            "damage_rate": m.get("Damage_Rate", "-"),
            "cost": int(m.get("Cost", 0)),
            "company": m.get("Company_Info", "-"),
        })

    # 결정표: 참고이미지 Ⅵ. 감정평가액의 결정 양식
    decision_row = None
    if stats:
        decision_row = {
            "item": ", ".join(sorted(set(rep_items))) or "(품목 미상)",
            "qty": len(rep.get("documents") or []) or 1,
            "median": stats["median"],
            "range_low": stats["q25"],
            "range_high": stats["q75"],
            "note": f"유사 사례 {stats['n']}건 기반, 25~75% 분위",
        }

    # LLM은 "결정 의견" 본문(참고이미지 하단 문단)만 생성
    rep_docs = rep.get("documents") or []
    emg_docs = emg.get("documents") or []
    rep_ctx = "\n".join(f"- {d}" for d in rep_docs) or "- (없음)"
    emg_ctx = "\n".join(f"- {d}" for d in emg_docs) or "- (없음)"
    est_ctx = "\n".join(f"- {d}" for d in est_docs) or "- (없음)"
    stats_line = (
        f"표본 {stats['n']}건, 25~75%: {stats['q25']:,}~{stats['q75']:,}원, "
        f"중앙값 {stats['median']:,}원"
    ) if stats else "유사 견적 사례 없음"

    prompt = f"""당신은 공공시설 수리 감정평가관입니다.
그룹 {group_id}에 대한 **'감정평가액의 결정 의견'** 한 문단을 작성하세요.
감정평가서 양식의 격식 있는 문체(~하였음, ~임)로, 3~5문장 내로 간결하게.
아래 자료만 근거로 하고 수치는 그대로 인용하세요(재계산 금지).

[신고 이력]
{rep_ctx}

[긴급도]
{emg_ctx}

[유사 견적 사례]
{est_ctx}

[견적 통계]
{stats_line}

[작성 지침]
- 유사 사례(기호 A~)의 수리비를 검토하여 본 건 시설물의 특성(손상 유형·긴급도·경과일)을 비교 설명
- 평균·중앙값·편차를 근거로 권장 범위가 합리적임을 서술
- 즉시/단기/관찰 중 조치 시급성을 명시
- 마지막 문장은 "…를 예상 수리비로 결정하였음." 으로 마무리
"""
    reply = llm.chat([{"role": "user", "content": prompt}])
    return {
        "reply": reply, "stats": stats,
        "similar_cases": similar_cases,
        "decision_row": decision_row,
        "reports": rep_docs,
        "emergency": emg_docs,
        "estimates": est_docs,
    }


# ─────────────────────────────────────────────────────────────
# 민원처리 계획 보고서 (공공기관 공문 양식) — rag_report.py 포팅
# ─────────────────────────────────────────────────────────────

REPORT_TEMPLATE = """[보고서 양식 가이드 — 이 구조와 순서, 기호를 따를 것]

{지역} {품목} 보수 민원 처리계획 보고
보고일자: {보고일자}
담당: {부서} {담당자명} (연락처 : {담당자연락처})

1. 민원 개요

* 민원 번호 : {민원번호}
* 민 원 인: {민원인명} (연락처 : {민원인연락처})
* 접수일시: {접수일시}
* 민원내용: {민원내용_한줄요약}

2. 파손 확인 및 검토

가. 파손 내용
* 보호시설물의 {품목} 파손 {파손단계}단계
* 누적 신고 {누적신고횟수}회와 근처 지역의 보호구역으로 위험도 {위험도}로 빠른 조치가 필요함

나. 과거 수리 사례
(표 형태: 수리일 / 품목 / 손상 단계 / 수리비(원) / 소요 일시(일) / 시공 업체)
(마지막 두 줄에 '평균' 행과 '최근값' 행 포함)

다. 예상 수리비 산출
ㅇ 예상 수리비 : 약 {예상수리비}원 (과거 동일조건 보수 실적 평균 적용)
ㅇ 예상 소요기간 : 약 {예상소요일}일
ㅇ 계약방법 : {계약방법}
"""


def _fmt_date(date_str: str) -> str:
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        w = ["월", "화", "수", "목", "금", "토", "일"][dt.weekday()]
        return f"{dt.year}. {dt.month}. {dt.day}.({w})"
    except Exception:
        return str(date_str)


def _fmt_date_simple(date_str: str) -> str:
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return str(date_str)


def _contract_method(cost: int) -> str:
    if cost <= 20_000_000: return "수의계약"
    if cost <= 100_000_000: return "제한경쟁입찰"
    return "일반경쟁입찰"


def _damage_stage(dr) -> int:
    try:
        rate = float(dr)
    except (TypeError, ValueError):
        return 0
    if rate <= 1:
        if rate < 0.34: return 1
        if rate < 0.67: return 2
        return 3
    return int(rate)


@lru_cache(maxsize=1)
def _report_db() -> pd.DataFrame:
    """원본 Report_DB + 합성 region 데이터 (있으면) 통합 로드."""
    df = pd.read_csv(PARENT_DIR / "Report_DB.csv", encoding="utf-8-sig")
    syn = PARENT_DIR / "Report_DB_synthetic.csv"
    if syn.exists():
        try:
            df_syn = pd.read_csv(syn, encoding="utf-8-sig")
            df = pd.concat([df, df_syn], ignore_index=True)
            log.info(f"Report_DB_synthetic 추가 로드: {len(df_syn)}행")
        except Exception as e:
            log.warning(f"Report_DB_synthetic 로드 실패: {e}")
    return df


@lru_cache(maxsize=1)
def _user_db() -> pd.DataFrame:
    return pd.read_csv(PARENT_DIR / "User_DB.csv", encoding="utf-8-sig")


def _live_to_report_row(report_id: str) -> Optional[dict]:
    """LIVE_XXXXXXXX 신고를 report() 가 받는 row dict 형태로 변환.

    담당자(Dept/Manager/Contact) 는 services.officers.lookup() 으로
    품목·region 기반 자동 조회. Officer_DB.csv 에 매칭 row 있으면 실데이터,
    없으면 부서명만 (Manager/Contact 빈 문자열) — placeholder 박힘 X.
    """
    from services import officers
    live = [r for r in storage.list_reports() if r["id"] == report_id]
    if not live:
        return None
    r = live[0]
    addr = r.get("address") or {}
    item = ITEM_NORMALIZE.get(r.get("item", ""), r.get("item", ""))
    o = officers.lookup(item, sido=addr.get("sido", ""), sigungu=addr.get("sigungu", ""))
    return {
        "Report_ID":   r["id"],
        "User_ID":     r.get("user_id", ""),
        "Group_ID":    r.get("group_id") or "",
        "Item":        item,
        "Damage_Rate": int(r.get("damage_rate", 0)),
        "Report_Date": r.get("created_at", ""),
        "Lat":         r.get("location", {}).get("lat"),
        "Lng":         r.get("location", {}).get("lon"),
        "Dept":        o.get("dept_full", "(부서)"),
        "Manager":     o.get("name", ""),
        "Contact":     o.get("phone", ""),
    }


def report(report_id: str, officer_info: Optional[dict] = None,
           model: Optional[str] = None) -> dict:
    """민원 처리계획 보고서 (RAG + LLM). REP_XXXXX(historical) 또는 LIVE_XXXXXXXX 모두 지원.

    Chroma est (의미검색 + 하드필터) + Report_DB.csv / live_reports.json + User_DB.csv / Emergency_DB 조합.
    """
    if report_id.startswith("LIVE_"):
        row = _live_to_report_row(report_id)
        if not row:
            return {"reply": f"LIVE 신고 ID '{report_id}' 를 찾을 수 없습니다.",
                    "data": None, "prompt": None, "model": model}
    else:
        rdb = _report_db()
        rows = rdb[rdb["Report_ID"] == report_id]
        if rows.empty:
            return {"reply": f"신고 ID '{report_id}' 를 찾을 수 없습니다.",
                    "data": None, "prompt": None, "model": model}
        row = rows.iloc[0].to_dict()

    # Report_DB.csv 원본 Item 을 Chroma 정규화 표기로 변환 (예: 보호시설물볼라드 → 볼라드)
    raw_item = str(row.get("Item", ""))
    item = ITEM_NORMALIZE.get(raw_item, raw_item)
    row["Item"] = item   # 하위 프롬프트·사례 표에서도 정규화된 이름 사용
    damage_stage = _damage_stage(row.get("Damage_Rate"))
    report_date = str(row.get("Report_Date", ""))
    user_id = str(row.get("User_ID", ""))
    group_id = str(row.get("Group_ID", ""))

    # 신고 지역: Lat/Lng 기반 reverse-geocoding (시도+시군구 레벨, 동은 생략)
    lat, lng = row.get("Lat"), row.get("Lng")
    location = "지역 미상"
    if lat is not None and lng is not None:
        try:
            addr = geocoding.reverse(float(lat), float(lng))
            if addr:
                location = f"{addr.get('sido','')} {addr.get('sigungu','')}".strip() or "지역 미상"
        except Exception:
            pass

    # 민원인
    udb = _user_db()
    uhit = udb[udb["User_ID"] == user_id]
    if not uhit.empty:
        u = uhit.iloc[0]
        user_info = {"name": str(u["Name"]), "phone": str(u["Contact"]), "matched": True}
    else:
        # User_DB 에 매칭 레코드 없음 — 원본 데이터 결함. User_ID 는 노출하고 나머지는 명시적 부재 표기.
        user_info = {"name": f"(미상 — User_DB 미등록)",
                     "phone": "(연락처 미상)", "matched": False, "user_id": user_id}

    # 긴급도 (Chroma emg)
    col = storage.collection()
    emg_info = {"frequency": 0, "final_score": 0.0, "matched": False}
    if group_id:
        emg = col.get(where={"$and": [{"source": "emg"}, {"Group_ID": group_id}]})
        if emg.get("metadatas"):
            em = emg["metadatas"][0]
            emg_info = {
                "frequency":   em.get("Frequency", 0),
                "final_score": em.get("Final_Score", 0.0),
                "matched": True,
            }

    # 과거 사례 — 의미검색 + 품목·파손단계 필터
    query = f"{location} {item} 파손 {damage_stage}단계 보수"
    est_conds = [{"source": "est"}]
    if item: est_conds.append({"Item": item})
    if damage_stage: est_conds.append({"Damage_Rate": damage_stage})
    est_hits = col.query(query_embeddings=_embed(query), n_results=8,
                         where=_where(est_conds))
    est_docs  = est_hits["documents"][0] if est_hits["documents"] else []
    est_metas = est_hits["metadatas"][0] if est_hits["metadatas"] else []

    # fallback: 품목 필터만
    if not est_metas and item:
        est_hits = col.query(query_embeddings=_embed(query), n_results=8,
                             where={"$and": [{"source": "est"}, {"Item": item}]})
        est_docs  = est_hits["documents"][0] if est_hits["documents"] else []
        est_metas = est_hits["metadatas"][0] if est_hits["metadatas"] else []

    # 최근순 정렬, 최대 4건
    if est_metas:
        paired = sorted(zip(est_metas, est_docs),
                        key=lambda p: p[0].get("Report_Date", ""))
        paired = paired[-4:]
        est_metas = [p[0] for p in paired]
        est_docs  = [p[1] for p in paired]

    # 통계
    costs = [int(m.get("Cost", 0)) for m in est_metas]
    days  = [int(m.get("Repair_Days", 0)) for m in est_metas]  # Chroma 메타 키는 Repair_Days
    no_data = not costs
    if costs:
        avg_cost = int(sum(costs) / len(costs))
        avg_days = round(sum(days) / len(days), 1)
        recent_cost, recent_days = costs[-1], days[-1]
        recent_year = est_metas[-1].get("Report_Date", "")[:4]
    else:
        avg_cost = recent_cost = recent_days = 0
        avg_days = 0.0
        recent_year = ""
    # 데이터 부재 시엔 수의계약 기본 (재량 + 현장 실측 안내)
    contract = _contract_method(avg_cost) if not no_data else "수의계약 (현장 실측 기반)"

    # 사례 표
    case_rows = []
    for m in est_metas:
        case_rows.append(
            f"| {_fmt_date_simple(m.get('Report_Date',''))} "
            f"| {m.get('Item','-')} "
            f"| {_damage_stage(m.get('Damage_Rate'))} "
            f"| {int(m.get('Cost',0)):,} "
            f"| {int(m.get('Repair_Days',0))} "
            f"| {m.get('Company_Info','-')} |"
        )
    case_table = "\n".join(case_rows) if case_rows else "(유사 사례 없음)"

    # 담당자
    officer = officer_info or {
        "department": str(row.get("Dept", "(부서)")),
        "name":       str(row.get("Manager", "(담당자)")),
        "phone":      str(row.get("Contact", "(연락처)")),
    }
    today = datetime.now()
    today_str = f"{today.year}. {today.month}. {today.day}."

    prompt = f"""당신은 {location} 시설관리과 공무원으로, 민원 처리계획 보고서를 작성하는 업무를 수행합니다.
아래 [DB 조회 결과]만을 근거로 보고서를 작성하세요. 외부 지식·추측 금지.
수치는 제공된 값을 그대로 사용하며 재계산하지 마십시오.

[DB 조회 결과]

■ 신고 정보 (Report_DB)
- 민원 번호: {report_id}
- 신고 지역: {location}
- 품목: {item}
- 파손 단계: {damage_stage}단계
- 접수일: {_fmt_date(report_date)}

■ 민원인 정보 (User_DB)
- 성명: {user_info['name']}
- 연락처: {user_info['phone']}
{"- (원본 User_DB 에 해당 User_ID 기록이 없어 성명·연락처 확인 불가. 보고서에는 '(미상)' 그대로 유지.)" if not user_info.get("matched") else ""}

■ 긴급도 (Emergency_DB)
{(f"- 누적 신고 횟수: {emg_info['frequency']}회" + chr(10) + f"- 위험도(Final Score): {emg_info['final_score']}") if emg_info.get("matched") else "- (해당 Group_ID 에 대한 긴급도 데이터 없음. '누적 신고'·'위험도' 수치는 '미집계' 로 표기하고 0 으로 단정하지 말 것.)"}

■ 과거 동일조건 수리 사례 (Estimate_DB, {len(est_metas)}건)
{("수리일 | 품목 | 손상단계 | 수리비(원) | 소요일 | 시공업체" + chr(10) + case_table) if not no_data else "(Estimate_DB 에 해당 품목 견적 데이터 없음 — 신규 품목이거나 신고 사례 부족)"}

■ 통계 (재계산 금지, 그대로 인용)
{(f"- 평균 수리비: {avg_cost:,}원" + chr(10) + f"- 평균 소요일: {avg_days}일" + chr(10) + f"- 최근값({recent_year}년): {recent_cost:,}원 / {recent_days}일") if not no_data else "- (통계 산출 불가 — 표본 0건)"}

■ 계약 정보
- 보고일자: {today_str}
- 담당: {officer['department']} {officer['name']} ({officer['phone']})
- 계약방법: {contract}

---

{REPORT_TEMPLATE}

---

[작성 지침]
1. 위 [보고서 양식 가이드]의 구조·순서·기호(*, ㅇ, 가/나/다)를 따를 것
2. "나. 과거 수리 사례" 표에는 위 {len(est_metas)}건을 모두 포함하고, 마지막에 '평균'과 '최근값' 행을 추가할 것
3. 수치는 [DB 조회 결과]에 제시된 값을 그대로 인용 (반올림·재계산 금지)
4. 빈 항목은 지어내지 말고 "(미상)" 또는 "(없음)"으로 표기
{'5. [데이터 부족 케이스] Estimate_DB 표본이 0건이므로, "나. 과거 수리 사례"는 표 없이 "(과거 수리 사례 데이터 없음 — 해당 품목의 견적 이력 부재)" 한 줄만 적을 것. "다. 예상 수리비 산출"은 "ㅇ 예상 수리비 : 산출 곤란(과거 견적 데이터 부족) — 현장 실측 견적 후 확정 예정 / ㅇ 예상 소요기간 : 현장 실측 필요 / ㅇ 계약방법 : 수의계약(현장 실측 기반)" 으로 작성. 0원/0일 같은 숫자를 임의로 적지 말 것.' if no_data else ''}
"""

    reply = llm.chat([{"role": "user", "content": prompt}], model=model)

    return {
        "reply": reply,
        "data": {
            # 헤더·표 렌더링용 메타
            "report_id":       report_id,
            "item":            item,
            "damage_stage":    damage_stage,
            "location":        location,
            "report_date":     report_date,
            "report_date_fmt": _fmt_date(report_date),
            "today_str":       today_str,
            "description":     str(row.get("description") or ""),
            "officer":         officer,
            # 데이터 블록 (기존)
            "report": row, "user": user_info, "emergency": emg_info,
            "past_cases": est_metas,
            "stats": {"avg_cost": avg_cost, "avg_days": avg_days,
                      "recent_cost": recent_cost, "recent_days": recent_days,
                      "recent_year": recent_year, "n": len(est_metas)},
            "contract_method": contract,
        },
        "prompt": prompt,
        "model": model or "default",
    }
