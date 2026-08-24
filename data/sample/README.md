# 데이터 안내

이 디렉토리에는 프로젝트 실행에 필요한 데이터 파일을 배치합니다.

## 필요한 파일 목록

| 파일명 | 설명 | 크기 | 출처 |
|--------|------|------|------|
| `EST_DB_기간.csv` | 시설물 수리 견적 이력 | - | 프로젝트 자체 수집 |
| `Report_DB.csv` | 과거 민원 신고 이력 | - | 프로젝트 자체 수집 |
| `Emergency_DB.csv` | 긴급도 계산용 데이터 | - | 프로젝트 자체 수집 |
| `User_DB.csv` | 사용자 계정 정보 | - | 데모용 생성 |
| `전국초중등학교위치표준데이터(1).csv` | 학교 위치 (보호구역 판단용) | - | [공공데이터포털](https://www.data.go.kr) |
| `chroma_est/` | ChromaDB 벡터 저장소 | ~수백MB | `scripts/rebuild_chroma.py`로 재생성 가능 |

## 샘플 데이터

`sample/` 디렉토리에 참고용 샘플 파일이 포함되어 있습니다.

## ChromaDB 재구축

CSV 데이터가 준비되면 벡터 DB를 재구축할 수 있습니다:
```bash
cd server
python scripts/rebuild_chroma.py
```
