# 📂 데이터 파일 안내

서버 실행 및 ChromaDB 벡터 인덱스 구축에 필요한 CSV 파일들입니다.

## 파일 목록

| 파일명 | 용도 | 사용 위치 |
|--------|------|-----------|
| `EST_DB_기간.csv` | 품목별 견적·수리비·시공업체 이력 | `rebuild_chroma.py` → ChromaDB 인덱싱 |
| `Report_DB.csv` | 신고 내역 (위치, 품목, 상태, 담당자 등) | `rag.py` 런타임 조회 + ChromaDB 인덱싱 |
| `Report_DB_synthetic.csv` | 서울 등 추가 지역 확장용 합성 신고 데이터 | `rebuild_chroma.py` → 원본과 합쳐서 인덱싱 |
| `Emergency_DB.csv` | 그룹별 긴급도 점수 (빈도, 위치위험도, 파손율 등) | `rebuild_chroma.py` → ChromaDB 인덱싱 |
| `Emergency_DB_synthetic.csv` | 서울 등 추가 지역 확장용 합성 긴급도 데이터 | `rebuild_chroma.py` → 원본과 합쳐서 인덱싱 |
| `User_DB.csv` | 사용자(신고자) 정보 | `rag.py` 런타임 조회 |
| `Officer_DB.csv` | 지역·부서별 담당 공무원 배정 정보 | `officers.py` 민원 자동 배정 |
| `전국초중등학교위치표준데이터(1).csv` | 학교 좌표 → 보호구역 위험도 산정 | `zones.py` 위치 기반 판별 |

## 원본 vs 합성(_synthetic)

`seed_regional_reports.py`로 생성된 합성 데이터는 원본에 없는 지역(서울 등)의 그룹을 추가하기 위한 것입니다.

- **원본** (`Report_DB.csv`, `Emergency_DB.csv`): 프로젝트 기본 범위의 데이터
- **합성** (`*_synthetic.csv`): 지역 확장용으로 새로 생성한 그룹 (Group_ID가 `GRP_SEO_xxxx` 형식)

`rebuild_chroma.py`가 원본 + 합성을 합쳐서 ChromaDB에 인덱싱하며, 원본 CSV는 변경하지 않습니다.

## ChromaDB 구축

```bash
cd server
python scripts/rebuild_chroma.py --reset
```

> 첫 실행 시 임베딩 모델(ko-sroberta-multitask)이 Hugging Face에서 자동 다운로드되며, 수 분이 소요될 수 있습니다.

## 학습용 이미지 데이터

YOLO 모델 학습에 사용된 원본 이미지(239,300장)는 용량 문제로 저장소에 포함되어 있지 않습니다. 직접 재학습하려면 아래에서 다운로드하세요:

- [AI Hub - 노후 시설물 이미지](https://aihub.or.kr/)
