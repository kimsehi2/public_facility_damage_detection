"""EST + REP + EMG 3개 소스를 단일 Chroma 컬렉션에 재빌드.

실행: python scripts/rebuild_chroma.py [--reset]
  --reset : chroma_est/ 폴더를 지우고 처음부터

노트북 RAG_Estimate.ipynb의 doc-cell/emb-cell 로직을 이식.
"""
import sys
import shutil
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (EST_CSV, REPORT_CSV, EMERG_CSV, CHROMA_DIR, COLLECTION, EMBED_MODEL,
                    ITEM_NORMALIZE, STATUS_NORMALIZE)


def _normalize_item(v):
    """원본 CSV Item → ALLOWED_ITEMS 표기로 정규화."""
    v = _scalar(v)
    return ITEM_NORMALIZE.get(str(v), v)


def _normalize_status(v):
    """원본 CSV Status(한글) → Live 와 동일한 영문 코드로 정규화."""
    v = _scalar(v)
    return STATUS_NORMALIZE.get(str(v), v)


def _clean(df):
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    return df


def _scalar(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if pd.isna(v): return ""
    return v


def build_docs():
    est_df = _clean(pd.read_csv(EST_CSV))
    rep_df = _clean(pd.read_csv(REPORT_CSV))
    emg_df = _clean(pd.read_csv(EMERG_CSV))

    # 합성 region 데이터 (있으면) 같이 인덱싱
    rep_syn = REPORT_CSV.parent / "Report_DB_synthetic.csv"
    emg_syn = EMERG_CSV.parent / "Emergency_DB_synthetic.csv"
    if rep_syn.exists():
        rep_df = pd.concat([rep_df, _clean(pd.read_csv(rep_syn))], ignore_index=True)
        print(f"  + 합성 신고 추가 로드: {rep_syn.name}")
    if emg_syn.exists():
        emg_df = pd.concat([emg_df, _clean(pd.read_csv(emg_syn))], ignore_index=True)
        print(f"  + 합성 긴급도 추가 로드: {emg_syn.name}")

    # EST
    est_ids = [f"est_{x}" for x in est_df["ID"]]
    est_docs = est_df.apply(lambda r: (
        f"[견적] 품목 {r['Item']}, 파손율 {r['Damage_Rate']}단계, "
        f"실제 수리비 {int(r['Cost']):,}원, 시공업체 {r['Company_Info']}, "
        f"접수 {r['Report_Date']}, 완료 {r['Completion_Date']}, 소요 {int(r['Repair_Days'])}일."
    ), axis=1).tolist()
    est_metas = [{
        "source": "est",
        "Item": _normalize_item(r["Item"]), "Damage_Rate": _scalar(r["Damage_Rate"]),
        "Cost": _scalar(r["Cost"]), "Company_Info": _scalar(r["Company_Info"]),
        "Repair_Days": _scalar(r["Repair_Days"]), "Report_Date": _scalar(r["Report_Date"]),
    } for _, r in est_df.iterrows()]

    # REP
    rep_ids = [f"rep_{x}" for x in rep_df["Report_ID"]]
    rep_docs = rep_df.apply(lambda r: (
        f"[신고] 분류 {r['Category']}, 품목 {_normalize_item(r['Item'])}, 파손율 {r['Damage_Rate']}단계, "
        f"그룹 {r['Group_ID']}, 위도 {r['Lat']}, 경도 {r['Lng']}, "
        f"접수 {r['Report_Date']}, 담당 {r['Dept']} {r['Manager']}, 상태 {_normalize_status(r['Status'])}."
    ), axis=1).tolist()
    rep_metas = [{
        "source": "rep",
        "Category": _scalar(r["Category"]), "Item": _normalize_item(r["Item"]),
        "Damage_Rate": _scalar(r["Damage_Rate"]), "Group_ID": _scalar(r["Group_ID"]),
        "Status": _normalize_status(r["Status"]), "Report_Date": _scalar(r["Report_Date"]),
    } for _, r in rep_df.iterrows()]

    # EMG
    emg_ids = [f"emg_{x}" for x in emg_df["Group_ID"]]
    emg_docs = emg_df.apply(lambda r: (
        f"[긴급도] 그룹 {r['Group_ID']}, 누적 신고 {int(r['Frequency'])}건, "
        f"위치 위험도 {r['Location_Risk']}, 파손율 {r['Damage_Rate']}, "
        f"최초 {r['First_Report_Date']} ~ 최근 {r['Last_Report_Date']}, "
        f"경과 {int(r['Elapsed_Days'])}일, 종합점수 {r['Final_Score']}."
    ), axis=1).tolist()
    emg_metas = [{
        "source": "emg",
        "Group_ID": _scalar(r["Group_ID"]), "Frequency": _scalar(r["Frequency"]),
        "Location_Risk": _scalar(r["Location_Risk"]), "Damage_Rate": _scalar(r["Damage_Rate"]),
        "Final_Score": _scalar(r["Final_Score"]), "Elapsed_Days": _scalar(r["Elapsed_Days"]),
    } for _, r in emg_df.iterrows()]

    return (est_ids + rep_ids + emg_ids,
            est_docs + rep_docs + emg_docs,
            est_metas + rep_metas + emg_metas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="chroma_est/ 폴더를 지우고 처음부터")
    args = ap.parse_args()

    if args.reset and CHROMA_DIR.exists():
        print(f"🗑  {CHROMA_DIR} 삭제")
        shutil.rmtree(CHROMA_DIR)

    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # 기존 컬렉션이 존재하면 삭제(깨끗한 재빌드)
    try:
        client.delete_collection(COLLECTION)
        print(f"🗑  기존 '{COLLECTION}' 컬렉션 삭제")
    except Exception:
        pass

    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    ids, docs, metas = build_docs()
    print(f"📦 전체 {len(docs):,}건")

    print(f"🧠 임베딩 모델 로드: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    BATCH = 500
    from itertools import islice
    for i in range(0, len(docs), BATCH):
        batch_docs = docs[i:i+BATCH]
        batch_ids = ids[i:i+BATCH]
        batch_metas = metas[i:i+BATCH]
        embs = embedder.encode(batch_docs, batch_size=64, show_progress_bar=False).tolist()
        col.add(ids=batch_ids, documents=batch_docs, embeddings=embs, metadatas=batch_metas)
        print(f"  · {min(i+BATCH, len(docs)):,} / {len(docs):,}")

    print(f"✅ 완료. 총 {col.count():,}건")


if __name__ == "__main__":
    main()
