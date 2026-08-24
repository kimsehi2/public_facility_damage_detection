"""전국 행정동 기반 오프라인 reverse-geocoding.

입력: (위도 lat, 경도 lon)
출력: {"sido": ..., "sigungu": ..., "emd": ..., "address": "..."} or None

요구사항:
- GeoJSON FeatureCollection (아래 경로 중 첫 번째 발견되는 파일 로드)
- 각 feature.geometry 는 Polygon/MultiPolygon
- feature.properties 에 시도/시군구/읍면동 이름 필드 포함

  대응 필드명 후보 (대소문자 무관):
    sido    = sidonm / ctp_kor_nm / sido_nm / sido
    sigungu = sggnm  / sig_kor_nm / sgg_nm  / sigungu
    emd     = emdnm  / emd_kor_nm / emd_nm  / adm_nm / dong

  adm_nm 한 필드에 "경기도 파주시 운정동" 형식으로 들어있는 GeoJSON도 지원 (공백 split 으로 분해).

성능:
- Shapely STRtree 로 O(log n) bounding-box 후보 필터링 후 contains() 확정
- 전국 행정동 ~3,500개 기준 평균 <5ms
"""
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from shapely.geometry import shape, Point
from shapely.strtree import STRtree

from config import BASE_DIR

log = logging.getLogger(__name__)

# 탐색 순서대로 첫 번째 발견되는 파일 사용
GEO_PATHS = [
    BASE_DIR / "data" / "korea_admin.geojson",
    BASE_DIR / "data" / "korea_hjd.geojson",
    BASE_DIR / "data" / "korea_emd.geojson",
]

SIDO_KEYS    = ["sidonm", "ctp_kor_nm", "CTP_KOR_NM", "sido_nm", "sido", "SIDO_NM"]
SIGUNGU_KEYS = ["sggnm", "sig_kor_nm", "SIG_KOR_NM", "sgg_nm", "sigungu", "SIGUNGU_NM"]
EMD_KEYS     = ["emdnm", "emd_kor_nm", "EMD_KOR_NM", "emd_nm", "adm_nm", "ADM_NM", "dong"]


def _pick(props: dict, keys: list[str]) -> str:
    for k in keys:
        v = props.get(k)
        if v:
            return str(v)
    return ""


def _find_geojson() -> Optional[Path]:
    for p in GEO_PATHS:
        if p.exists():
            return p
    return None


@lru_cache(maxsize=1)
def _load():
    """GeoJSON 로드 + STRtree 빌드. 파일 없으면 (None, [], []) 반환."""
    path = _find_geojson()
    if path is None:
        log.warning("행정동 GeoJSON 없음. 다음 중 하나에 배치: %s",
                    [str(p) for p in GEO_PATHS])
        return None, [], []

    log.info(f"GeoJSON 로드: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    features = data.get("features", []) if isinstance(data, dict) else []

    polys, meta = [], []
    for f in features:
        try:
            poly = shape(f["geometry"])
        except Exception:
            continue
        props = f.get("properties", {}) or {}
        sido = _pick(props, SIDO_KEYS)
        sgg  = _pick(props, SIGUNGU_KEYS)
        emd  = _pick(props, EMD_KEYS)
        # 케이스1: sido/sgg 있고 emd(=adm_nm)가 전체 주소인 경우 → prefix 제거
        if emd and sido and emd.startswith(sido):
            rest = emd[len(sido):].strip()
            if sgg and rest.startswith(sgg):
                emd = rest[len(sgg):].strip()
            else:
                emd = rest
        # 케이스2: sido 없이 emd에 전체 주소만 있는 경우 → 공백 분해
        elif not sido and emd and " " in emd:
            parts = emd.split()
            if len(parts) >= 3:
                sido, sgg, emd = parts[0], parts[1], " ".join(parts[2:])
        polys.append(poly)
        meta.append({"sido": sido, "sigungu": sgg, "emd": emd})

    if not polys:
        log.warning("GeoJSON 로드됐지만 feature 0개")
        return None, [], []

    tree = STRtree(polys)
    log.info(f"STRtree 빌드 완료: {len(polys):,} polygons")
    return tree, polys, meta


def reverse(lat: float, lon: float) -> Optional[dict]:
    """좌표 → 행정동 정보. 파일 없거나 매칭 실패 시 None."""
    tree, polys, meta = _load()
    if tree is None:
        return None
    pt = Point(lon, lat)  # shapely: (x=lon, y=lat)
    candidates = tree.query(pt)  # shapely 2.x: returns indices (np.ndarray)
    for idx in candidates:
        i = int(idx)
        if polys[i].contains(pt):
            m = meta[i]
            full = " ".join(x for x in (m["sido"], m["sigungu"], m["emd"]) if x)
            return {**m, "address": full}
    return None


def status() -> dict:
    """현재 로드 상태 확인용 (API/디버깅)."""
    tree, polys, _ = _load()
    path = _find_geojson()
    return {
        "loaded": tree is not None,
        "count": len(polys),
        "file": str(path) if path else None,
    }
