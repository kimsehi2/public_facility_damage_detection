"""YOLO crop + SIFT + RANSAC inlier 기반 중복 신고 매칭.

기존 ORB 전체-이미지 매칭의 한계(배경 노이즈, 약한 조도 강건성)를 해결하기 위해
yolo_sift_matching 노트북의 접근법 채택:
  1) YOLO 로 시설물 bbox 탐지 → 그 영역만 크롭
  2) CLAHE 전처리(조도 정규화)
  3) SIFT descriptor 추출
  4) BFMatcher (NORM_L2) + Lowe ratio test → good matches
  5) RANSAC + Homography → inlier 검증 (기하 일관성)
  6) inlier 수 ≥ FEATURE_MIN_INLIERS 면 동일 물품 판정

- /report 시점에 YOLO 재추론 → bbox 추출 → 크롭 → SIFT feature 저장
- 같은 품목 + 일정 반경 후보와 매칭 (반경은 50m 로 완화 — SIFT가 강건함)
- 매칭 시각화 이미지: uploads/matches/<new>_vs_<old>.jpg
"""
import logging
from math import radians, sin, cos, asin, sqrt
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import (
    UPLOADS_DIR,
    FEATURE_LOWE_RATIO,
    FEATURE_MATCH_RADIUS_M,
    FEATURE_SIFT_NFEATURES,
    FEATURE_MIN_GOOD_MATCHES,
    FEATURE_MIN_INLIERS,
    FEATURE_RANSAC_THRESH,
)

log = logging.getLogger(__name__)

FEATURES_DIR = UPLOADS_DIR / "features"
CROPS_DIR = UPLOADS_DIR / "crops"
MATCHES_DIR = UPLOADS_DIR / "matches"
for _d in (FEATURES_DIR, CROPS_DIR, MATCHES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

_sift = None
_bf = cv2.BFMatcher(cv2.NORM_L2)


def _get_sift():
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=FEATURE_SIFT_NFEATURES)
    return _sift


def _preprocess_crop(crop_bgr, max_size: int = 1024):
    """리사이즈 + 그레이 + CLAHE (조도 보정). 반환: (gray, bgr_resized, scale_applied)."""
    h, w = crop_bgr.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1.0:
        crop_bgr = cv2.resize(
            crop_bgr, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        scale = 1.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray), crop_bgr, scale


def _crop_by_bbox(image_bgr, bbox):
    """안전 크롭. 잘못된 bbox 면 원본 반환."""
    x1, y1, x2, y2 = bbox
    h, w = image_bgr.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return image_bgr  # fallback: 원본 전체
    return image_bgr[y1:y2, x1:x2]


def _detect_best(image_path):
    """YOLO 추론 → confidence 최고 detection 의 (bbox, obb_points) 반환. 실패 시 (None, None)."""
    try:
        from services import yolo_service
        with open(image_path, "rb") as f:
            content = f.read()
        dets = yolo_service.detect(content)
    except Exception as e:
        log.warning(f"YOLO 추출 실패 {image_path}: {e}")
        return None, None
    if not dets:
        return None, None
    best = dets[0]  # confidence 최고 (이미 정렬됨)
    return best["bbox"], best.get("obb_points")


def _make_polygon_mask(shape, polygon_xy):
    """그레이 이미지 shape 에 맞는 polygon 마스크. polygon 안 255 / 밖 0."""
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.array(polygon_xy, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def extract(image_path) -> Optional[dict]:
    """YOLO OBB 폴리곤 마스크 + SIFT feature 추출. 결과는 save() 로 영속화.

    OBB 회전 박스의 4 꼭짓점으로 polygon 마스크 생성 → 마스크 안에서만 SIFT keypoint.
    배경(바닥/잔디) features 를 원천 차단해 매칭 신뢰도 향상.

    반환 dict:
      kps, desc, bbox, shape, crop_path (기존과 동일)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        log.warning(f"이미지 로드 실패: {image_path}")
        return None

    bbox, obb_points = _detect_best(image_path)
    if bbox is None or all(v == 0 for v in bbox):
        log.info(f"YOLO 탐지 없음 → 전체 이미지로 매칭: {image_path}")
        bbox = [0, 0, img.shape[1], img.shape[0]]
        obb_points = None

    crop_raw = _crop_by_bbox(img, bbox)
    gray, resized_bgr, scale = _preprocess_crop(crop_raw)

    # OBB 폴리곤 → 크롭 상대좌표 + resize 스케일 적용 → 그레이 shape 에 맞는 마스크
    mask = None
    if obb_points is not None:
        shifted = [((x - bbox[0]) * scale, (y - bbox[1]) * scale) for x, y in obb_points]
        try:
            mask = _make_polygon_mask(gray.shape, shifted)
        except Exception as e:
            log.warning(f"OBB 마스크 생성 실패 ({image_path}): {e}")
            mask = None

    sift = _get_sift()
    kp, desc = sift.detectAndCompute(gray, mask)
    if desc is None or len(kp) < 4:
        log.warning(f"SIFT feature 부족: {image_path} (kp={len(kp) if kp else 0}, masked={mask is not None})")
        return None

    kps = np.array([k.pt for k in kp], dtype=np.float32)

    # 크롭 이미지 저장 (시각화용) — 마스크 적용 전 원본 크롭
    rid = Path(image_path).stem
    crop_path = CROPS_DIR / f"{rid}.jpg"
    cv2.imwrite(str(crop_path), resized_bgr)

    return {
        "kps": kps,
        "desc": desc.astype(np.float32),
        "bbox": list(bbox),
        "shape": gray.shape,
        "crop_path": crop_path,
    }


def save(report_id: str, feat: dict) -> Path:
    path = FEATURES_DIR / f"{report_id}.npz"
    np.savez_compressed(
        path,
        kps=feat["kps"],
        desc=feat["desc"],
        bbox=np.array(feat["bbox"], dtype=np.int32),
        shape=np.array(feat["shape"], dtype=np.int32),
    )
    return path


def load(report_id: str) -> Optional[dict]:
    path = FEATURES_DIR / f"{report_id}.npz"
    if not path.exists():
        return None
    try:
        data = np.load(path)
        crop_path = CROPS_DIR / f"{report_id}.jpg"
        return {
            "kps": data["kps"],
            "desc": data["desc"],
            "bbox": data["bbox"].tolist(),
            "shape": tuple(int(x) for x in data["shape"]),
            "crop_path": crop_path if crop_path.exists() else None,
        }
    except Exception as e:
        log.warning(f"feature 로드 실패 {report_id}: {e}")
        return None


def match_inliers(feat_a: dict, feat_b: dict,
                  ratio: Optional[float] = None,
                  min_good: Optional[int] = None) -> dict:
    """두 feature 매칭. Lowe ratio + RANSAC homography → inlier 수.

    반환: {good_matches, inliers, total_kp_a, total_kp_b}
    """
    r = ratio if ratio is not None else FEATURE_LOWE_RATIO
    mg = min_good if min_good is not None else FEATURE_MIN_GOOD_MATCHES
    out = {"good_matches": 0, "inliers": 0,
           "total_kp_a": int(len(feat_a.get("kps", []))),
           "total_kp_b": int(len(feat_b.get("kps", [])))}

    da, db = feat_a.get("desc"), feat_b.get("desc")
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return out

    try:
        knn = _bf.knnMatch(da, db, k=2)
    except cv2.error as e:
        log.warning(f"knnMatch 실패: {e}")
        return out

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < r * n.distance:
            good.append(m)
    out["good_matches"] = len(good)

    if len(good) < mg:
        return out

    kps_a, kps_b = feat_a["kps"], feat_b["kps"]
    src_pts = np.float32([kps_a[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps_b[m.trainIdx] for m in good]).reshape(-1, 1, 2)
    try:
        _H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, FEATURE_RANSAC_THRESH)
    except cv2.error:
        return out
    if mask is not None:
        out["inliers"] = int(mask.sum())
    return out


def draw_match(img_a_path, id_a: str, img_b_path, id_b: str) -> Optional[Path]:
    """두 신고의 크롭 이미지 + SIFT 매칭선 시각화 → uploads/matches/<a>_vs_<b>.jpg.

    크롭 + 전처리한 그레이 이미지 위에 RANSAC inlier 만 그림.
    """
    feat_a = load(id_a)
    feat_b = load(id_b)
    if feat_a is None or feat_b is None:
        return None
    if feat_a.get("crop_path") is None or feat_b.get("crop_path") is None:
        return None

    crop_a = cv2.imread(str(feat_a["crop_path"]))
    crop_b = cv2.imread(str(feat_b["crop_path"]))
    if crop_a is None or crop_b is None:
        return None

    gray_a, _, _ = _preprocess_crop(crop_a)
    gray_b, _, _ = _preprocess_crop(crop_b)
    disp_a = cv2.cvtColor(gray_a, cv2.COLOR_GRAY2BGR)
    disp_b = cv2.cvtColor(gray_b, cv2.COLOR_GRAY2BGR)

    da, db = feat_a["desc"], feat_b["desc"]
    try:
        knn = _bf.knnMatch(da, db, k=2)
    except cv2.error:
        return None

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < FEATURE_LOWE_RATIO * n.distance:
            good.append(m)
    if len(good) < FEATURE_MIN_GOOD_MATCHES:
        return None

    src_pts = np.float32([feat_a["kps"][m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([feat_b["kps"][m.trainIdx] for m in good]).reshape(-1, 1, 2)
    try:
        _H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, FEATURE_RANSAC_THRESH)
    except cv2.error:
        return None
    inlier_matches = ([m for m, ok in zip(good, mask.ravel()) if ok][:60]
                      if mask is not None else good[:60])

    kp_a = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=10) for p in feat_a["kps"]]
    kp_b = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=10) for p in feat_b["kps"]]
    viz = cv2.drawMatches(
        disp_a, kp_a, disp_b, kp_b, inlier_matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=(200, 200, 200),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    save_path = MATCHES_DIR / f"{id_a}_vs_{id_b}.jpg"
    cv2.imwrite(str(save_path), viz)
    return save_path


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def find_nearby_candidates(reports: list, lat: float, lon: float, item: str,
                           radius_m: Optional[float] = None,
                           exclude_id: Optional[str] = None,
                           exclude_statuses=("completed", "rejected")) -> list:
    """같은 품목 + 지정 반경 이내의 기존 신고 후보 반환 (가까운 순).

    처리 종결된 신고(completed/rejected)는 매칭 후보에서 자동 제외.
    이유: 종결 케이스에 매칭돼 봐야 그 group_id 는 이미 닫힘 → 신규 신고는 fresh 그룹이 맞음.
    """
    r = radius_m if radius_m is not None else FEATURE_MATCH_RADIUS_M
    exclude_statuses = set(exclude_statuses or ())
    out = []
    for rep in reports:
        if exclude_id and rep.get("id") == exclude_id:
            continue
        if rep.get("item") != item:
            continue
        if rep.get("status") in exclude_statuses:
            continue
        loc = rep.get("location") or {}
        rlat, rlon = loc.get("lat"), loc.get("lon")
        if rlat is None or rlon is None:
            continue
        d = _haversine_m(lat, lon, float(rlat), float(rlon))
        if d <= r:
            out.append({"report": rep, "distance_m": d})
    out.sort(key=lambda x: x["distance_m"])
    return out


def resolve_group_id(new_feat: dict, candidates: list,
                     min_inliers: Optional[int] = None) -> dict:
    """후보들과 전수 매칭 → RANSAC inlier 수로 동일 물품 판정 (정책 A: Max).

    매칭 정책 A (Max):
      후보 N개 중 **inliers 가 가장 높은 1개** 가 임계 ≥ 면 그 후보의 group 에 합류.
      대표(representative) 1개와만 비교하지 않고 후보 전체와 비교 → 시점 다양성 강건.
      후보는 (같은 품목 + 일정 반경 + 활성 신고만) 으로 사전 필터됨.

    반환:
      duplicate_of: 매칭된 후보 ID (없으면 None)
      group_id:     매칭된 후보의 group (없으면 None — 호출 측 fresh 발급)
      match_score:  inlier 수 (없으면 0)
      match_candidates: 상위 3개 [{id, inliers, good_matches, distance_m, group_id}]
    """
    threshold = min_inliers if min_inliers is not None else FEATURE_MIN_INLIERS
    scored = []
    for cand in candidates:
        rep = cand["report"]
        old_feat = load(rep["id"])
        if old_feat is None:
            continue
        m = match_inliers(new_feat, old_feat)
        scored.append({
            "id": rep["id"],
            "inliers": int(m["inliers"]),
            "good_matches": int(m["good_matches"]),
            "distance_m": round(cand["distance_m"], 2),
            "group_id": rep.get("group_id"),
        })
    scored.sort(key=lambda s: -s["inliers"])

    best = scored[0] if scored else None
    if best and best["inliers"] >= threshold:
        return {
            "duplicate_of": best["id"],
            "group_id": best.get("group_id"),
            "match_score": best["inliers"],
            "match_candidates": scored[:3],
        }
    return {
        "duplicate_of": None,
        "group_id": None,
        "match_score": best["inliers"] if best else 0,
        "match_candidates": scored[:3],
    }
