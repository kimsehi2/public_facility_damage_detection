"""YOLO 추론 + EXIF GPS 추출."""
import io
import logging
from functools import lru_cache
from typing import Optional, Tuple, List, Dict
import numpy as np
from PIL import Image, ExifTags

from config import (
    YOLO_WEIGHTS,
    DAMAGE_SEG_WEIGHTS,
    DAMAGE_CROP_PADDING_RATIO,
    CLASS_TO_ITEM,
)

log = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _model():
    from ultralytics import YOLO
    log.info(f"YOLO 로드: {YOLO_WEIGHTS}")
    return YOLO(YOLO_WEIGHTS)


@lru_cache(maxsize=1)
def _damage_model():
    """Stage 2 damage seg 모델. 단일 클래스 'damage' 또는 멀티클래스(시설별 *_damage) 둘 다 호환."""
    from ultralytics import YOLO
    log.info(f"Damage Seg 로드: {DAMAGE_SEG_WEIGHTS}")
    return YOLO(DAMAGE_SEG_WEIGHTS)


def _damage_class_ids(model) -> list:
    """seg 모델의 클래스 중 'damage' 가 이름에 포함된 것만 추림.
    - facility_split (9-cls): 'total' 제외, 8개 *_damage 채택.
    - two-stage v1 (1-cls): 'damage' 1개 채택.
    - fallback: 이름에 damage 없는 모델이면 전 클래스 채택 (단, 비추천 — 잘못 매핑된 모델일 수 있음).
    """
    ids = [i for i, n in model.names.items() if "damage" in str(n).lower()]
    if ids:
        return ids
    return list(model.names.keys())

def detect(image_bytes: bytes) -> List[Dict]:
    """이미지 바이트 → 탐지 결과 리스트 (신뢰도 내림차순).

    OBB 모델은 `res.obb`, 일반 detection 은 `res.boxes` 에 결과가 담김 — 둘 다 지원.
    26-클래스 OBB 모델: 클래스명이 `{Base}_Normal|Damaged` 형태. `damaged=True` 만 신고 대상.
    레거시 5-클래스 모델은 suffix 없음 → 전부 파손으로 간주.
    """
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    m = _model()
    res = m.predict(pil, verbose=False)[0]
    dets: List[Dict] = []

    # OBB(Oriented BBox) vs 일반 Detection 둘 다 대응
    src = res.obb if getattr(res, "obb", None) is not None else res.boxes
    if src is None or getattr(src, "cls", None) is None:
        return dets

    cls_t = src.cls
    conf_t = src.conf
    n = int(cls_t.shape[0]) if hasattr(cls_t, "shape") else len(cls_t)
    # OBB 회전 박스 → 축 정렬 박스(AABB) 변환 (feature 매칭용 크롭에 사용)
    has_obb = res.obb is not None
    xyxyxyxy = src.xyxyxyxy.cpu().numpy() if has_obb and getattr(src, "xyxyxyxy", None) is not None else None
    xyxy = src.xyxy.cpu().numpy() if not has_obb and getattr(src, "xyxy", None) is not None else None
    for i in range(n):
        cls_name = m.names[int(cls_t[i])]
        base = cls_name.rsplit("_", 1)
        if len(base) == 2 and base[1] in ("Normal", "Damaged"):
            item_en, damaged = base[0], (base[1] == "Damaged")
        else:
            item_en, damaged = cls_name, True
        # bbox (axis-aligned) + obb_points (회전 박스 4 꼭짓점; feature 매칭 마스크용)
        obb_points = None
        if xyxyxyxy is not None:
            pts = xyxyxyxy[i]
            x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
            x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
            obb_points = pts.astype(int).tolist()   # [[x,y], [x,y], [x,y], [x,y]]
        elif xyxy is not None:
            x1, y1, x2, y2 = (int(v) for v in xyxy[i])
        else:
            x1 = y1 = x2 = y2 = 0
        dets.append({
            "class": cls_name,
            "item_ko": CLASS_TO_ITEM.get(item_en, "기타"),
            "item_en": item_en,
            "damaged": damaged,
            "confidence": float(conf_t[i]),
            "bbox": [x1, y1, x2, y2],
            "obb_points": obb_points,
        })
    dets.sort(key=lambda d: -d["confidence"])
    return dets

def detect_with_damage(image_bytes: bytes,
                       padding_ratio: Optional[float] = None) -> List[Dict]:
    """Two-stage 추론: OBB 시설 탐지 + 각 시설 crop 에서 damage seg → damage_ratio 산출.

    각 detection dict 에 추가되는 필드:
      - damage_pixels: int — damage 마스크 픽셀 수 (전 *_damage 클래스 union)
      - crop_area:    int — padded crop 의 총 픽셀 수
      - damage_ratio: float — damage_pixels / crop_area (0.0~1.0)

    OBB 탐지 0개면 빈 리스트 반환 (기존 detect 와 동일).
    Damaged=False (정상) 인 detection 에도 동일하게 ratio 계산 (보통 0 에 가까움) — 정책은 라우터에서.
    """
    dets = detect(image_bytes)
    if not dets:
        return dets

    pad = float(padding_ratio if padding_ratio is not None else DAMAGE_CROP_PADDING_RATIO)
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.asarray(pil)
    H, W = img.shape[:2]

    seg_model = _damage_model()
    damage_cls_ids = set(_damage_class_ids(seg_model))

    for det in dets:
        x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
        bw, bh = max(0, x2 - x1), max(0, y2 - y1)
        if bw <= 0 or bh <= 0:
            det["damage_pixels"] = 0
            det["crop_area"] = 0
            det["damage_ratio"] = 0.0
            continue
        pad_x, pad_y = int(bw * pad), int(bh * pad)
        cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        cx2, cy2 = min(W, x2 + pad_x), min(H, y2 + pad_y)
        if cx2 - cx1 < 10 or cy2 - cy1 < 10:
            det["damage_pixels"] = 0
            det["crop_area"] = int((cx2 - cx1) * (cy2 - cy1))
            det["damage_ratio"] = 0.0
            continue

        crop = img[cy1:cy2, cx1:cx2]
        try:
            seg_res = seg_model.predict(crop, verbose=False)[0]
        except Exception as e:
            log.warning(f"damage seg 실패: {e}")
            det["damage_pixels"] = 0
            det["crop_area"] = int(crop.shape[0] * crop.shape[1])
            det["damage_ratio"] = 0.0
            continue

        damage_pixels = 0
        total = 0
        if seg_res.masks is not None and len(seg_res.masks) > 0:
            masks = seg_res.masks.data.cpu().numpy()  # (N, h, w) — 보통 입력 비율로 다운샘플
            cls_arr = (seg_res.boxes.cls.cpu().numpy().astype(int)
                       if seg_res.boxes is not None else np.zeros(len(masks), dtype=int))
            mh, mw = masks.shape[1], masks.shape[2]
            total = int(mh * mw)
            union = np.zeros((mh, mw), dtype=bool)
            for i in range(masks.shape[0]):
                if int(cls_arr[i]) in damage_cls_ids:
                    union |= (masks[i] > 0.5)
            damage_pixels = int(union.sum())
        det["damage_pixels"] = damage_pixels
        det["crop_area"] = total or int((cx2 - cx1) * (cy2 - cy1))
        det["damage_ratio"] = round(damage_pixels / det["crop_area"], 4) if det["crop_area"] else 0.0

    return dets


def extract_gps(image_bytes: bytes) -> Optional[Tuple[float, float]]:
    """EXIF에서 GPS (lat, lon) 추출. 최신 PIL API + HEIC 지원.

    iPhone HEIC, Android 신형 JPEG 모두 대응. _getexif() (deprecated) 대신 getexif() + get_ifd(0x8825).
    """
    # HEIC 지원 — pillow-heif 설치 시 자동 등록
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass

    try:
        pil = Image.open(io.BytesIO(image_bytes))
        exif = pil.getexif()
    except Exception as e:
        log.warning(f"EXIF 읽기 실패: {e}")
        return None
    if not exif:
        log.info("이미지에 EXIF 없음")
        return None

    # GPSInfo IFD (tag 0x8825) — 최신 API
    gps_ifd = exif.get_ifd(0x8825) if hasattr(exif, "get_ifd") else None
    if not gps_ifd:
        # fallback — 옛날 방식
        for k, v in exif.items():
            if ExifTags.TAGS.get(k) == "GPSInfo" and isinstance(v, dict):
                gps_ifd = v
                break
    if not gps_ifd:
        log.info("EXIF에 GPSInfo 없음")
        return None

    gps = {ExifTags.GPSTAGS.get(t, t): vv for t, vv in gps_ifd.items()}
    if "GPSLatitude" not in gps or "GPSLongitude" not in gps:
        log.info(f"GPSInfo 있는데 좌표 부족: keys={list(gps.keys())}")
        return None

    def _to_dec(dms, ref):
        try:
            d = float(dms[0]) + float(dms[1]) / 60 + float(dms[2]) / 3600
        except Exception:
            return None
        return -d if ref in ("S", "W") else d

    lat = _to_dec(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
    lon = _to_dec(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
    if lat is None or lon is None:
        log.warning(f"좌표 변환 실패: lat_raw={gps.get('GPSLatitude')} lon_raw={gps.get('GPSLongitude')}")
        return None
    log.info(f"EXIF GPS 추출 성공: lat={lat:.5f} lon={lon:.5f}")
    return lat, lon
