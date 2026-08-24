"""경로·모델·도메인 상수. 모든 곳에서 `from config import ...` 로 접근."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent              # .../server
ROOT_DIR = BASE_DIR.parent                             # 프로젝트 루트
DATA_DIR = ROOT_DIR / "data"                           # CSV·벡터DB 등 데이터

# ── 데이터 소스 ──
EST_CSV     = DATA_DIR / "EST_DB_기간.csv"
REPORT_CSV  = DATA_DIR / "Report_DB.csv"
EMERG_CSV   = DATA_DIR / "Emergency_DB.csv"
SCHOOLS_CSV = DATA_DIR / "전국초중등학교위치표준데이터(1).csv"

# ── 벡터 DB ──
CHROMA_DIR  = DATA_DIR / "chroma_est"
COLLECTION  = "estimates"

# ── 모델 ──
EMBED_MODEL = "jhgan/ko-sroberta-multitask"
# 민원인 챗봇: 응답 속도 우선 → 작은 e2b
# 관리자 RAG·챗봇: 추론 품질 우선 → e4b
LLM_MODEL_CITIZEN = os.getenv("LLM_MODEL_CITIZEN", "gemma4:e2b")
LLM_MODEL_ADMIN   = os.getenv("LLM_MODEL_ADMIN",   "gemma4:e4b")
LLM_MODEL = LLM_MODEL_ADMIN                              # 기존 임포트 호환
YOLO_WEIGHTS = ROOT_DIR / "models" / "yolo26n_best.pt"

# ── Damage Seg (Two-stage Stage 2) ──
# OBB → 시설 crop → 이 모델로 damage 영역 마스킹 → ratio → damage_rate 자동 산정
# yolov8s_total_lwooq (2026-04-26): yolov8s-seg, 2-class (damage/total) — 'damage' 만 추출
# 이전: damage_only_v3 (yolo11s, 1-class) — 백업으로 보존
DAMAGE_SEG_WEIGHTS = ROOT_DIR / "models" / "yolov8m_seg_best.pt"
DAMAGE_CROP_PADDING_RATIO = 0.10   # OBB box 외곽 10% 여유 (경계 damage 보존)

# damage_ratio (mask 픽셀 / crop 픽셀, 0.0~1.0) → damage_rate (1~3) 임계값
# 현 UI 단계: 1 경미, 2 중간, 3 심함
# 시설별 임계값 — 시설마다 "정상" 손상 면적이 달라서 (예: 맨홀 균열은 적은 %도 위험)
# 단위: 비율 (0~1.0). 사용자 제공값 % 를 /100 한 것.
ITEM_DAMAGE_THRESHOLDS = {
    "등받이있는벤치":   [0.05, 0.15],   # 부서진 부분 0/5/15%
    "등받이없는벤치":   [0.05, 0.12],   # 0/5/12%
    "보도블록":         [0.04, 0.14],   # 0/4/14%
    "점자블록":         [0.03, 0.07],   # 0/3/7%
    "보차도경계석":     [0.07, 0.10],   # 0/7/10%
    "볼라드":           [0.09, 0.14],   # 윗 부분 부서짐 0/9/14%
    "맨홀":             [0.02, 0.03],   # crack 0/2/3% (작은 균열도 위험)
    # 트랜치(804) — legacy 로 제외됨 (운영 중단)
    # 가로수보호덮개(602) — 비-ALLOWED 라 신고 자체 안 받음. 통계용 (참고)
    "가로수보호덮개":   [0.12, 0.37],   # 빈틈 0/12/37%
}
DAMAGE_RATE_THRESHOLDS_DEFAULT = [0.05, 0.20]
# 호환용 별칭 (구 코드)
DAMAGE_RATE_THRESHOLDS = DAMAGE_RATE_THRESHOLDS_DEFAULT

def damage_ratio_to_rate(ratio: float, item: str = "") -> int:
    """seg mask 면적 비율 → 1~3 단계. 시설별 임계 적용 (없으면 default)."""
    thresholds = ITEM_DAMAGE_THRESHOLDS.get(item) or DAMAGE_RATE_THRESHOLDS_DEFAULT
    r = float(ratio or 0.0)
    for i, t in enumerate(thresholds):
        if r < t:
            return i + 1
    return len(thresholds) + 1

# YOLO 클래스 → 한글 품목 (yolo26_n모델 22클래스 = 11 base × Normal/Damaged)
# Normal/Damaged suffix 는 yolo_service 에서 분리. 매핑은 base 이름만.
CLASS_TO_ITEM = {
    # 신고 대상 (ALLOWED_ITEMS) — 7종
    "BenchWithBack":      "등받이있는벤치",
    "BenchWithoutBack":   "등받이없는벤치",
    "SidewalkBlock":      "보도블록",
    "BrailleBlock":       "점자블록",
    "CurbStone":          "보차도경계석",
    "Bollard":            "볼라드",
    "Manhole":            "맨홀",
    # 모델은 인식하지만 신고 대상 제외 (운영 정책)
    "Trench":             "트랜치",
    "ProtectionFence":    "보호펜스",
    "JaywalkPrevention":  "무단횡단방지봉",
    "TreeCover":          "가로수보호덮개",
}

# 카탈로그·드롭다운·신고 가능 화이트리스트 — 운영 중인 7종
ALLOWED_ITEMS = {
    "등받이있는벤치", "등받이없는벤치",
    "보도블록", "점자블록", "보차도경계석",
    "볼라드", "맨홀",
}

# Estimate_DB / Report_DB 의 원본 Item 표기를 ALLOWED_ITEMS 명칭으로 정규화.
# (원본 CSV 는 그대로 두고 Chroma 메타에만 적용.)
ITEM_NORMALIZE = {
    "보호시설물볼라드":         "볼라드",
    "보호시설물무단횡단방지시설": "무단횡단방지시설",
    # 추후 비슷한 접두사 표기 발견되면 추가 (예: 원본 "보호시설물X" → "X")
}

# ── 품목 → 부서 매핑 (Officer_DB 조회 시 사용. 모든 region 공통 부서명) ──
# Report_DB 1만건 historical 통계로 결정된 기본 매핑.
ITEM_TO_DEPT = {
    "등받이있는벤치": "공원녹지과",
    "등받이없는벤치": "공원녹지과",
    "보도블록":       "도로과",
    "점자블록":       "도로과",
    "보차도경계석":   "도로과",
    "맨홀":          "도로과",
    "볼라드":        "시설관리과",
}

# ── 품목별 객체위험계수 (긴급도 식의 곱셈 항) ──
# 긴급도 = 객체위험계수 × [위치위험도×0.4 + 민원빈도×0.3 + 파손도×0.2 + 방치기간×0.1]
ITEM_BASE_URGENCY = {
    "맨홀":             1.0,   # 기준점 (가장 위험)
    "보도블록":         0.9,
    "점자블록":         0.9,
    "보차도경계석":     0.85,
    "볼라드":           0.8,
    "등받이있는벤치":   0.75,
    "등받이없는벤치":   0.7,
    # legacy (운영 외) 통계용으로만 유지
    "가로수보호덮개":   0.7,
}
# 매핑에 없는 품목은 1.0 (default)

# Report_DB 의 한글 Status 를 Live 와 동일한 영문 코드로 정규화.
STATUS_NORMALIZE = {
    "신고접수":   "pending",
    "처리진행중": "in_progress",
    "처리완료":   "completed",
    # "처리반려" / "반려" 는 rep 에 없음. 추후 생기면 "rejected" 로.
}

# ── 런타임 저장소 (서버 로컬) ──
UPLOADS_DIR    = BASE_DIR / "uploads"
LIVE_REPORTS   = BASE_DIR / "live_reports.json"          # 민원인 제출 신고 원장
SQLITE_PATH    = BASE_DIR / "reports.db"

# ── Feature matching (중복 신고 → 같은 Group_ID 묶기) ──
# YOLO 크롭 + SIFT + RANSAC homography 기반. inlier 수가 임계 이상이면 동일 물품.
FEATURE_MATCH_RADIUS_M   = 20     # 같은 물품 후보로 볼 최대 거리(m). 실측 후 50→20 로 좁힘.
FEATURE_SIFT_NFEATURES   = 3000   # SIFT keypoint 최대 개수
FEATURE_LOWE_RATIO       = 0.75   # Lowe's ratio test
FEATURE_MIN_GOOD_MATCHES = 4      # RANSAC 시도 최소 good match
FEATURE_MIN_INLIERS      = 4      # RANSAC inlier 임계 — 동일 물품 판정 (15→8→4 로 단계적 완화)
FEATURE_RANSAC_THRESH    = 5.0    # RANSAC reprojection error (px)

# ── 지리 상수 ──
DEFAULT_MAP_CENTER = (36.019, 129.343)   # 지도 초기 중심 (POSTECH 캠퍼스 부근). 전국 어디든 신고 가능.

# ── API ──
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── 선택: Google API (ragas 평가·제미나이 심사용) ──
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ── 보호구역 기반 위치 위험도 ──
# 신고 위치 P 의 300m 반경에 들어오는 보호구역 개수로 zone_score 산정
#   zone_score = ZONE_W_CHILD * n_child + ZONE_W_ELDER * n_elder
#   risk = damage_rate * zone_score * ITEM_BASE_URGENCY[item]
# 어린이(초·중·고·유치원·어린이집·특수학교) >> 노인·장애인 >> 비보호구역(0)
ZONE_RADIUS_M = 300
ZONE_W_CHILD = 5.0
ZONE_W_ELDER = 2.0

def risk_score(damage_rate: int, n_child: int = 0, n_elder: int = 0,
               item: str = "", elapsed_days: int = 0) -> float:
    """단일 신고의 위험도 (= per-report 버전 긴급도, frequency=1).

    식: 객체위험계수 × [위치×0.3 + 빈도×0.2 + 파손×0.3 + 방치×0.2]
       각 항 1/2/3 이산 점수. 범위 약 1.0 ~ 3.4 (단일 신고 max).
    """
    # 위치위험도 — 보호구역 종류 수
    has_child = (n_child or 0) > 0
    has_elder = (n_elder or 0) > 0
    if has_child and has_elder: loc = 3
    elif has_child or has_elder: loc = 2
    else: loc = 1

    # 민원빈도 — 단일 신고는 항상 1
    freq = 1

    # 파손도 — 1/2/3 단계 (시설별 임계로 이미 구간화됨)
    dmg = max(min(int(damage_rate or 1), 3), 1)

    # 방치기간 — 7/30일 임계
    if elapsed_days >= 30:  el = 3
    elif elapsed_days >= 7: el = 2
    else:                   el = 1

    weighted = loc * 0.3 + freq * 0.2 + dmg * 0.3 + el * 0.2
    item_w = ITEM_BASE_URGENCY.get(item, 1.0)
    return round(item_w * weighted, 2)


def risk_tier(r: float):
    """긴급도 tier — 1/2/3 식 분포 (객체위험계수 0.7~1.0, 점수 약 0.7 ~ 3.0)."""
    if r >= 2.4: return ("🔴", "red",    "매우 위험")
    if r >= 1.9: return ("🟠", "orange", "위험")
    if r >= 1.3: return ("🟡", "beige",  "주의")
    return ("🟢", "green", "낮음")

# 호환 alias — 기존 코드가 emergency_tier 부르고 있으면 risk_tier 와 동일하게 동작
emergency_tier = risk_tier
