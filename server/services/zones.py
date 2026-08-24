"""보호구역 기반 위치 위험도.

`server/data/protection_zones.json` (어린이 보호구역 ∪ 중·고등 / 노인·장애인) 로드.
각 보호구역은 ZONE_RADIUS_M 반경의 원을 가지며, 신고 위치가 몇 개의 원에
겹치는지로 위험도를 산정한다.
"""
import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from config import BASE_DIR, ZONE_RADIUS_M

log = logging.getLogger(__name__)

ZONES_JSON = BASE_DIR / "data" / "protection_zones.json"


@lru_cache(maxsize=1)
def _zones():
    """보호구역 데이터 로드. {'child': (lats, lons, names, kinds, addrs),
                              'elder': (...) } 형태로 numpy 배열 캐시."""
    log.info(f"보호구역 로드: {ZONES_JSON}")
    records = json.loads(ZONES_JSON.read_text(encoding="utf-8"))
    out = {}
    for grp in ("child", "elder"):
        rs = [r for r in records if r["group"] == grp]
        out[grp] = {
            "lat":   np.array([r["lat"]  for r in rs], dtype=np.float64),
            "lon":   np.array([r["lon"]  for r in rs], dtype=np.float64),
            "name":  [r["name"] for r in rs],
            "kind":  [r["kind"] for r in rs],
            "addr":  [r["addr"] for r in rs],
        }
        log.info(f"  {grp}: {len(rs)}건")
    return out


def _haversine_m(lat1, lon1, lats, lons):
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lats);  lon2 = np.radians(lons)
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * 6371000 * np.arcsin(np.sqrt(a))


def score(lat: float, lon: float, radius: int = ZONE_RADIUS_M) -> dict:
    """신고 위치 (lat, lon) 에 대해 child/elder 보호구역 카운트 + 가까운 시설 목록.

    반환:
      {
        "n_child": int,
        "n_elder": int,
        "nearby_child": [{name, kind, distance_m, addr, lat, lon}, ... ≤5],
        "nearby_elder": [...],
        "nearest": {group, name, kind, distance_m, addr, lat, lon} 또는 None
      }
    """
    z = _zones()
    out = {"n_child": 0, "n_elder": 0,
           "nearby_child": [], "nearby_elder": [], "nearest": None}

    nearest_overall = None  # (dist, group, idx)
    for grp in ("child", "elder"):
        d = _haversine_m(lat, lon, z[grp]["lat"], z[grp]["lon"])
        idxs_in = np.where(d <= radius)[0]
        # 가까운 순으로 정렬해서 상위 5
        order = np.argsort(d)
        in_order = [j for j in order if j in set(idxs_in)][:5]
        if grp == "child":
            out["n_child"] = int(len(idxs_in))
        else:
            out["n_elder"] = int(len(idxs_in))
        out[f"nearby_{grp}"] = [{
            "name":       z[grp]["name"][j],
            "kind":       z[grp]["kind"][j],
            "distance_m": float(d[j]),
            "addr":       z[grp]["addr"][j],
            "lat":        float(z[grp]["lat"][j]),
            "lon":        float(z[grp]["lon"][j]),
        } for j in in_order]

        # 전체 최근접 추적 (반경 밖이라도)
        j_min = int(np.argmin(d))
        if nearest_overall is None or d[j_min] < nearest_overall[0]:
            nearest_overall = (float(d[j_min]), grp, j_min)

    if nearest_overall is not None:
        d, grp, j = nearest_overall
        out["nearest"] = {
            "group":      grp,
            "name":       z[grp]["name"][j],
            "kind":       z[grp]["kind"][j],
            "distance_m": d,
            "addr":       z[grp]["addr"][j],
            "lat":        float(z[grp]["lat"][j]),
            "lon":        float(z[grp]["lon"][j]),
        }
    return out
