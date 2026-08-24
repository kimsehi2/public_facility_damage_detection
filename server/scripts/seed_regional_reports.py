"""서울/부산/제주 합성 historical Report + Emergency 시드.

- 원본 Report_DB.csv / Emergency_DB.csv 는 **건드리지 않음** — 별도 파일에 저장
- 출력:
    server/data/Report_DB_synthetic.csv     (합성 신고 행)
    server/data/Emergency_DB_synthetic.csv  (합성 그룹 긴급도)
- Chroma 'estimates' 컬렉션에 append (rep + emg source). --reset 안 함.
- rag._report_db() 는 자동으로 두 CSV concat 해서 로드 (별도 패치).

데모용 fictional 데이터:
- 좌표는 region 박스 안에서 random
- Group_ID prefix: GRP_SEO_xxxx / GRP_BUS_xxxx / GRP_JEJ_xxxx
- 담당자: Officer_DB.csv 의 region 별 매핑 사용
"""
import csv
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (ALLOWED_ITEMS, ITEM_TO_DEPT, ITEM_NORMALIZE, STATUS_NORMALIZE,
                    CHROMA_DIR, COLLECTION, EMBED_MODEL, BASE_DIR, PARENT_DIR)

random.seed(42)

# ─────────────────────────────────────────────────────────────
# region 별 합성 정책
# ─────────────────────────────────────────────────────────────
REGIONS = [
    {
        # 서울 행정구역 안전 박스 (관악~도봉, 강서~송파)
        "code": "SEO", "sido": "서울특별시",
        "lat_range": (37.485, 37.650), "lng_range": (126.860, 127.150),
        "n_reports": 300,
        "manager_pool": {
            "도로과":     [("김도현", "010-3245-8612"),
                          ("강민혁", "010-5566-7788"),
                          ("조성민", "010-2233-4455")],
            "공원녹지과": [("박민지", "010-7821-3456"),
                          ("윤서연", "010-3344-5566")],
            "시설관리과": [("이지훈", "010-9234-5678"),
                          ("정현우", "010-8877-6655")],
        },
    },
    {
        # 부산 도심 안전 박스
        "code": "BUS", "sido": "부산광역시",
        "lat_range": (35.070, 35.220), "lng_range": (128.970, 129.200),
        "n_reports": 200,
        "manager_pool": {
            "도로과":     [("최정우", "010-5678-1234"), ("이상혁", "010-4455-6677")],
            "공원녹지과": [("김현수", "010-2345-6789")],
            "시설관리과": [("정유진", "010-8765-4321")],
        },
    },
    {
        # 제주도 본섬 (해안 + 일부 내륙) 안전 박스
        "code": "JEJ", "sido": "제주특별자치도",
        "lat_range": (33.220, 33.500), "lng_range": (126.250, 126.780),
        "n_reports": 50,
        "manager_pool": {
            "도로과":     [("한지원", "010-4321-6789")],
            "공원녹지과": [("송미경", "010-1234-9876")],
            "시설관리과": [("박상현", "010-9876-5432")],
        },
    },
]

ITEM_POOL = sorted(ALLOWED_ITEMS)
# 원본 Report_DB.csv 표기로 'CSV-side' 저장 (보호시설물볼라드 형태) — Chroma 인덱싱 시 정규화됨
ITEM_TO_CSV = {"볼라드": "보호시설물볼라드"}
def to_csv_item(item: str) -> str:
    return ITEM_TO_CSV.get(item, item)

CATEGORY_BY_DEPT = {
    "도로과":     "통행시설물",
    "공원녹지과": "휴게시설물",
    "시설관리과": "보호시설물",
}

STATUS_DIST = ["처리완료"] * 70 + ["처리진행중"] * 20 + ["신고접수"] * 10


def gen_reports_for_region(region):
    """한 region 의 reports + groups 생성."""
    n_groups = max(1, region["n_reports"] // 6)   # 평균 6 reports / group
    groups = []
    for gi in range(n_groups):
        gid = f"GRP_{region['code']}_{gi+1:04d}"
        item = random.choice(ITEM_POOL)
        center_lat = random.uniform(*region["lat_range"])
        center_lng = random.uniform(*region["lng_range"])
        groups.append({"gid": gid, "item": item,
                       "lat": center_lat, "lng": center_lng})

    rows = []
    for i in range(region["n_reports"]):
        g = random.choice(groups)
        # 그룹 중심 ±10m 이내 (lat 1도 ≈ 111km)
        lat = g["lat"] + random.uniform(-0.0001, 0.0001)
        lng = g["lng"] + random.uniform(-0.0001, 0.0001)
        item = g["item"]
        dept = ITEM_TO_DEPT.get(item, "시설관리과")
        manager_name, manager_phone = random.choice(region["manager_pool"][dept])
        days_ago = random.randint(30, 700)
        report_date = (datetime(2024, 1, 1) + timedelta(days=days_ago)).strftime("%Y-%m-%d")
        damage_rate = random.choices([1, 2, 3], weights=[3, 5, 2])[0]
        user_id = f"USER_{random.randint(10000, 99999)}"
        status = random.choice(STATUS_DIST)
        rid = f"REP_{region['code']}_{i+1:05d}"
        rows.append({
            "Report_ID":   rid,
            "User_ID":     user_id,
            "Group_ID":    g["gid"],
            "Category":    CATEGORY_BY_DEPT[dept],
            "Item":        to_csv_item(item),
            "Damage_Rate": damage_rate,
            "Lat":         round(lat, 6),
            "Lng":         round(lng, 6),
            "Report_Date": report_date,
            "Dept":        dept,
            "Manager":     manager_name,
            "Contact":     manager_phone,
            "Status":      status,
        })
    return rows, groups


def gen_emergency_for_groups(reports):
    """그룹별 frequency, final_score 등 합성 긴급도."""
    by_gid = defaultdict(list)
    for r in reports:
        by_gid[r["Group_ID"]].append(r)
    out = []
    for gid, members in by_gid.items():
        freq = len(members)
        max_dr = max(int(m["Damage_Rate"]) for m in members)
        first_date = min(m["Report_Date"] for m in members)
        last_date = max(m["Report_Date"] for m in members)
        elapsed = (datetime.strptime(last_date, "%Y-%m-%d")
                   - datetime.strptime(first_date, "%Y-%m-%d")).days
        location_risk = round(random.uniform(2.0, 9.5), 2)
        final = round(
            max_dr * location_risk
            * (1 + 0.15 * min(freq - 1, 10))
            * (1 + 0.005 * min(elapsed, 365)), 2)
        out.append({
            "Group_ID":          gid,
            "Frequency":         freq,
            "Damage_Rate":       max_dr,
            "Location_Risk":     location_risk,
            "First_Report_Date": first_date,
            "Last_Report_Date":  last_date,
            "Elapsed_Days":      elapsed,
            "Final_Score":       final,
        })
    return out


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  → {path}: {len(rows)}행")


def append_to_chroma(reports, emg_rows):
    """Chroma 컬렉션에 upsert (rep + emg). 정규화 적용. idempotent."""
    import chromadb
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    print("\n=== Chroma upsert (rep + emg, 합성) ===")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_or_create_collection(COLLECTION)

    # 기존 합성 ID 삭제 (idempotent — 좌표 박스 등 변경 시 stale 방지)
    old_rep = PARENT_DIR / "Report_DB_synthetic.csv"
    old_emg = PARENT_DIR / "Emergency_DB_synthetic.csv"
    if old_rep.exists():
        try:
            df_old = pd.read_csv(old_rep)
            old_ids = [f"rep_{x}" for x in df_old["Report_ID"]]
            if old_ids:
                col.delete(ids=old_ids)
                print(f"  기존 합성 rep {len(old_ids)}건 삭제 (idempotent)")
        except Exception as e:
            print(f"  기존 rep 삭제 실패 (무시): {e}")
    if old_emg.exists():
        try:
            df_old = pd.read_csv(old_emg)
            old_ids = [f"emg_{x}" for x in df_old["Group_ID"]]
            if old_ids:
                col.delete(ids=old_ids)
                print(f"  기존 합성 emg {len(old_ids)}건 삭제")
        except Exception as e:
            print(f"  기존 emg 삭제 실패 (무시): {e}")

    embedder = SentenceTransformer(EMBED_MODEL)

    # rep
    rep_ids = [f"rep_{r['Report_ID']}" for r in reports]
    rep_docs = [
        f"[신고] 분류 {r['Category']}, "
        f"품목 {ITEM_NORMALIZE.get(r['Item'], r['Item'])}, "
        f"파손율 {r['Damage_Rate']}단계, "
        f"그룹 {r['Group_ID']}, 위도 {r['Lat']}, 경도 {r['Lng']}, "
        f"접수 {r['Report_Date']}, 담당 {r['Dept']} {r['Manager']}, "
        f"상태 {STATUS_NORMALIZE.get(r['Status'], r['Status'])}."
        for r in reports
    ]
    rep_metas = [{
        "source":      "rep",
        "Category":    r["Category"],
        "Item":        ITEM_NORMALIZE.get(r["Item"], r["Item"]),
        "Damage_Rate": int(r["Damage_Rate"]),
        "Group_ID":    r["Group_ID"],
        "Status":      STATUS_NORMALIZE.get(r["Status"], r["Status"]),
        "Report_Date": r["Report_Date"],
    } for r in reports]
    embs = embedder.encode(rep_docs, batch_size=64, show_progress_bar=False).tolist()
    col.upsert(ids=rep_ids, documents=rep_docs, embeddings=embs, metadatas=rep_metas)
    print(f"  rep upsert: {len(rep_ids)}건")

    # emg
    emg_ids = [f"emg_{r['Group_ID']}" for r in emg_rows]
    emg_docs = [
        f"[긴급도] 그룹 {r['Group_ID']}, 누적 신고 {int(r['Frequency'])}건, "
        f"위치 위험도 {r['Location_Risk']}, 파손율 {r['Damage_Rate']}, "
        f"최초 {r['First_Report_Date']} ~ 최근 {r['Last_Report_Date']}, "
        f"경과 {int(r['Elapsed_Days'])}일, 종합점수 {r['Final_Score']}."
        for r in emg_rows
    ]
    emg_metas = [{
        "source":        "emg",
        "Group_ID":      r["Group_ID"],
        "Frequency":     int(r["Frequency"]),
        "Damage_Rate":   int(r["Damage_Rate"]),
        "Location_Risk": float(r["Location_Risk"]),
        "Final_Score":   float(r["Final_Score"]),
        "Elapsed_Days":  int(r["Elapsed_Days"]),
    } for r in emg_rows]
    embs2 = embedder.encode(emg_docs, batch_size=64, show_progress_bar=False).tolist()
    col.upsert(ids=emg_ids, documents=emg_docs, embeddings=embs2, metadatas=emg_metas)
    print(f"  emg upsert: {len(emg_ids)}건")


def main():
    all_reports = []
    all_groups = []
    print("=== 합성 region 데이터 생성 ===")
    for region in REGIONS:
        rows, groups = gen_reports_for_region(region)
        all_reports.extend(rows)
        all_groups.extend(groups)
        print(f"  {region['sido']} ({region['code']}): "
              f"신고 {len(rows)}건, 그룹 {len(groups)}개")

    emg_rows = gen_emergency_for_groups(all_reports)
    print(f"\n  Emergency 합성: {len(emg_rows)}그룹")

    print("\n=== CSV 저장 (원본 보존, 별도 파일) ===")
    rep_path = PARENT_DIR / "Report_DB_synthetic.csv"
    emg_path = PARENT_DIR / "Emergency_DB_synthetic.csv"
    write_csv(rep_path, all_reports, list(all_reports[0].keys()))
    write_csv(emg_path, emg_rows, list(emg_rows[0].keys()))

    append_to_chroma(all_reports, emg_rows)

    print(f"\n✓ 완료. 원본 Report_DB.csv 미변경. 합성은 별도 _synthetic.csv 에.")
    print(f"  rag._report_db() 패치 후 REP_SEO_xxxxx 같은 ID 도 보고서 조회 가능.")


if __name__ == "__main__":
    main()
