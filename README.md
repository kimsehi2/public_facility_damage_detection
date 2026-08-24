# 🏙️ AI 기반 스마트 공공시설물 관리 플랫폼

노후 공공시설물 민원 처리의 자동화를 위한 AI 이미지 분석 + LLM 기반 풀스택 플랫폼입니다.

> 📌 본 저장소는 청년 AI·Big Data 아카데미 32기 팀 프로젝트(B반 1조)를 김세희 개인이 포트폴리오용으로 재구성한 버전입니다.

## 📌 프로젝트 배경

- 2024년 민원 신고 건수 **1,243.5만 건** (2021년 대비 6배 증가)
- 전체 민원의 약 **90%**가 안전 및 생활불편 관련
- 5건 중 1건은 '진행중' 상태로 행정 절차 지연 발생

## 🔧 시스템 아키텍처

```
[신고자 웹 UI]                         [관리자 대시보드]
     │                                       │
     ├── 사진 업로드                          ├── 민원 현황 모니터링
     ├── 챗봇 (gemma4:e2b)                   ├── RAG 기반 견적 보고서
     └── 민원 접수                            └── 관리자 챗봇 (gemma4:e4b)
          │                                       │
          └──────────── FastAPI 서버 ──────────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
        YOLO Detection  YOLO Seg    SIFT Matching
        (시설물 탐지)   (파손율 측정) (중복 판별)
              │             │             │
              └─────────────┼─────────────┘
                            │
                   ChromaDB + Ollama (LLM)
```

## 🛠 핵심 기능

### 1. Two-Stage YOLO 파이프라인
- **Stage 1 — Detection**: YOLO26n-OBB로 시설물 탐지 (22클래스 = 11종 × 정상/파손)
- **Stage 2 — Segmentation**: YOLOv8m-seg로 파손 영역 분할 → 파손율 자동 산정
- AI Hub 노후 시설물 이미지 **239,300장** 활용, 라벨 정제 **57,600개**

### 2. SIFT 기반 중복 민원 탐지
- YOLO 크롭 + CLAHE 전처리 + SIFT 디스크립터 + RANSAC
- 반경 20m 이내 동일 시설물 자동 그루핑 (Group_ID 부여)

### 3. LLM/RAG 챗봇 & 보고서
- **신고자 챗봇**: gemma4:e2b — 신고 현황 안내, 수리 소요 기간 예측
- **관리자 챗봇**: gemma4:e4b — 민원 요약 및 질의응답
- **RAG 보고서**: ChromaDB 벡터 검색 → 견적 보고서 자동 생성

### 4. 긴급도 자동 계산
```
긴급도 = 객체위험계수 × [(위치위험도×0.3) + (민원빈도×0.2) + (파손도×0.3) + (방치기간×0.2)]
```
서울시설공단·포항시설관리공단·포항남구청 공무원 **9명** 설문조사로 가중치 도출

## 📊 모델 성능

| 모델 | Precision | Recall | mAP@50 |
|------|-----------|--------|--------|
| YOLO26n-OBB (Detection) | 0.906 | 0.869 | **0.923** |
| YOLOv8m-seg (Segmentation) | 0.868 | 0.777 | **0.806** |

## 📁 프로젝트 구조

```
public-facility-damage-detection/
├── README.md
├── requirements.txt
├── .gitignore
│
├── server/                     # FastAPI 웹 서버
│   ├── main.py                 # 앱 진입점
│   ├── config.py               # 경로·모델·도메인 상수
│   ├── run.sh                  # 실행 스크립트
│   ├── routes/
│   │   ├── citizen.py          # 신고자 API (사진 업로드, 챗봇, 접수)
│   │   ├── admin.py            # 관리자 API (대시보드, RAG, 보고서)
│   │   └── pages.py            # 페이지 라우팅
│   ├── services/
│   │   ├── yolo_service.py     # Two-stage YOLO 추론
│   │   ├── feature_service.py  # SIFT 중복 탐지
│   │   ├── rag.py              # RAG 검색·보고서 생성
│   │   ├── llm.py              # Ollama LLM 호출
│   │   ├── storage.py          # 신고 저장 (JSON + ChromaDB)
│   │   ├── emergency.py        # 긴급도 계산
│   │   ├── geocoding.py        # 역지오코딩 (행정구역 판별)
│   │   ├── zones.py            # 보호구역 위험도
│   │   └── ...
│   ├── static/                 # 프론트엔드 (HTML/CSS/JS)
│   │   ├── index.html          # 신고자 채팅 UI
│   │   ├── admin.html          # 관리자 대시보드
│   │   └── ...
│   └── scripts/                # 유틸리티 스크립트
│       ├── rebuild_chroma.py   # ChromaDB 재구축
│       └── ...
│
├── models/                     # YOLO 모델 가중치 (.pt)
│   └── README.md               # 다운로드 안내
│
├── notebooks/                  # 모델 학습 과정 (Jupyter)
│   ├── YOLO_Detection.ipynb
│   ├── YOLO_Segmentation.ipynb
│   └── Feature_Matching_SIFT.ipynb
│
└── data/                       # 데이터 (Git 미포함, 별도 다운로드)
    ├── sample/                 # 샘플 데이터
    └── README.md               # 데이터 다운로드 안내
```

## 🚀 설치 및 실행

### 사전 요구사항

- Python 3.10+
- [Ollama](https://ollama.ai) 설치 및 실행
- YOLO 모델 가중치 파일 (→ `models/README.md` 참고)
- 데이터 파일 (→ `data/README.md` 참고)

### 1. 저장소 클론

```bash
git clone https://github.com/kimsehi2/public-facility-damage-detection.git
cd public-facility-damage-detection
```

### 2. Ollama 모델 다운로드

```bash
ollama pull gemma4:e2b    # 신고자 챗봇용 (경량)
ollama pull gemma4:e4b    # 관리자 RAG·챗봇용 (고품질)
```



### 4. 데이터 파일 배치

`data/README.md`를 참고하여 CSV 파일들을 `data/` 디렉토리에 넣으세요.

### 5. 서버 실행

```bash
cd server
bash run.sh
```

또는 수동으로:

```bash
cd server
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r ../requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```



## 🛠 사용 기술

| 분류 | 기술 |
|------|------|
| Backend | FastAPI, Uvicorn, Python 3.10 |
| AI 모델 | YOLO (Detection, OBB, Segmentation), SIFT (OpenCV) |
| LLM | Ollama (gemma4:e4b, gemma4:e2b, qwen3:4b) |
| RAG | ChromaDB, SentenceTransformers (ko-sroberta-multitask) |
| Frontend | HTML/CSS/JS (Single Page) |
| 데이터 | AI Hub 노후 시설물 이미지, Roboflow 라벨링 |

## 👥 팀원

| 이름 | 역할 |
|------|------|
| 김세희 | Roboflow 라벨링, YOLO Segmentation 모델 개발 |
| 송노건 | LLM 개발, PPT |
| 양호준 | 팀장, LLM 개발, 웹페이지 개발 |
| 이소현 | Roboflow 라벨링, PPT |
| 이호원 | Roboflow 라벨링, YOLO Detection 모델 개발 |

## 📄 라이선스

이 프로젝트는 학습 및 포트폴리오 목적으로 제작되었습니다.
