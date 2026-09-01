import argparse
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from scipy.optimize import curve_fit
from shapely.geometry import Polygon

@dataclass(frozen=True)
class Config:
    in_points: Path
    out_head_points: Path
    out_toe_points: Path
    out_poly: Path
    shoreline_id: str
    transect_id: str
    distance: str
    elevation: str
    mhhw: str
    head_elev_mean: str
    elevation_nodata: tuple[float, ...]
    mhhw_nodata: tuple[float, ...]

def load_config(path: Path) -> Config:
    """
    Read a TOML configuration file into a Config.
    """

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    paths = raw["paths"]
    fields = raw["fields"]
    nodata = raw.get("nodata", {})

    return Config(
        in_points=Path(paths["in_points"]),
        out_head_points=Path(paths["out_head_points"]),
        out_toe_points=Path(paths["out_toe_points"]),
        out_poly=Path(paths["out_poly"]),
        shoreline_id=fields["shoreline_id"],
        transect_id=fields["transect_id"],
        distance=fields["distance"],
        elevation=fields["elevation"],
        mhhw=fields["mhhw"],
        head_elev_mean=fields["head_elev_mean"],
        elevation_nodata=tuple(nodata.get("elevation", [])),
        mhhw_nodata=tuple(nodata.get("mhhw", [])),
    )

def get_data(cfg: Config) -> gpd.GeoDataFrame:
    """
    mmm yummy data
    """

    data = gpd.read_file(cfg.in_points)
    if cfg.elevation_nodata:
        data[cfg.elevation] = data[cfg.elevation].replace(list(cfg.elevation_nodata), np.nan)
    if cfg.mhhw_nodata:
        data[cfg.mhhw] = data[cfg.mhhw].replace(list(cfg.mhhw_nodata), np.nan)

    return data

def extract_shore(data: gpd.GeoDataFrame, shore_id: int, cfg: Config) -> gpd.GeoDataFrame:
    """

    """

    shore = data[data[cfg.shoreline_id] == shore_id].copy()

    return shore

def extract_prof(data: gpd.GeoDataFrame, prof_id: int, cfg: Config) -> gpd.GeoDataFrame:
    """

    """

    prof = data[data[cfg.transect_id] == prof_id].copy()
    prof.sort_values(cfg.distance, inplace=True)
    prof.reset_index(drop=True, inplace=True)

    return prof

def _sigmoid(x, L, k, x0, b):

    with np.errstate(over="ignore"):
        return L / (1.0 + np.exp(-k * (x - x0))) + b

def _sigmoid_deriv2(x, L, k, x0, b):
    # analytic second derivative of _sigmoid: f'' = L k^2 s(1-s)(1-2s)
    with np.errstate(over="ignore"):
        s = 1.0 / (1.0 + np.exp(-k * (x - x0)))
    return L * k**2 * s * (1.0 - s) * (1.0 - 2.0 * s)

def delineate_transect(prof: gpd.GeoDataFrame, cfg: Config) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | None:
    """
    Delineate the toe and head of one transect profile (sorted by distance).

    The toe is anchored first at the point where elevation is closest to MHHW;
    the head is then the point of maximum concave curvature (minimum second
    derivative) of a least-squares sigmoid fit, restricted to points strictly
    landward of the toe. Landward is inferred per transect as the direction in
    which the fitted sigmoid rises. Returns None if either point cannot be
    delineated under those constraints.
    """

    x = prof[cfg.distance].to_numpy(dtype=float)
    y = prof[cfg.elevation].to_numpy(dtype=float)
    mhhw = prof[cfg.mhhw].to_numpy(dtype=float)

    mhhw_diff = np.abs(y - mhhw)
    if np.isnan(mhhw_diff).all():
        return None
    toe_pos = int(np.nanargmin(mhhw_diff))

    y_valid = ~np.isnan(y)
    x_fit = x[y_valid]
    y_fit = y[y_valid]

    if len(x_fit) < 4:
        return None

    b0 = float(np.min(y_fit))
    L0 = float(np.max(y_fit) - np.min(y_fit))
    x0 = float(np.mean(x_fit))
    k0 = 1.0 if (y_fit[-1] - y_fit[0]) >= 0 else -1.0
    p0 = [L0 if L0 != 0 else 1.0, k0, x0, b0]

    try:
        popt, _ = curve_fit(_sigmoid, x_fit, y_fit, p0=p0, maxfev=10000)
    except (RuntimeError, ValueError):
        return None

    rise = _sigmoid(x[-1], *popt) - _sigmoid(x[0], *popt)
    if rise == 0:
        return None
    landward_of_toe = (x > x[toe_pos]) if rise > 0 else (x < x[toe_pos])

    candidates = y_valid & landward_of_toe
    if not candidates.any():
        return None

    deriv2 = _sigmoid_deriv2(x, *popt)
    deriv2[~candidates] = np.nan
    try:
        head_pos = int(np.nanargmin(deriv2))
    except ValueError:
        return None

    head = prof.iloc[[head_pos]].copy()
    toe = prof.iloc[[toe_pos]].copy()

    return head, toe



def main():
    parser = argparse.ArgumentParser(description="Delineate bluff head/toe points and per-shoreline bluff polygons.")
    parser.add_argument("config", nargs="?", default="config.toml", type=Path,
                        help="path to TOML configuration file (default: ./config.toml)")
    args = parser.parse_args()
    cfg = load_config(args.config)

    in_points = get_data(cfg)
    shore_set = sorted(in_points[cfg.shoreline_id].unique())
    crs = in_points.crs
    head_rows = []
    toe_rows = []
    poly_records = []
    skipped = 0

    for S_i in tqdm(shore_set, ascii=True, desc="Shoreline"):
        head_geoms = []
        toe_geoms = []
        head_elevs = []

        shore = extract_shore(in_points, S_i, cfg)

        profiles_for_shore = sorted(shore[cfg.transect_id].unique())

        for P_i in tqdm(profiles_for_shore, ascii=True, desc="Profile"):
            profile_i = extract_prof(data=shore, prof_id=P_i, cfg=cfg)
            delineated = delineate_transect(profile_i, cfg)

            if delineated is None:
                skipped += 1
                continue

            head_i, toe_i = delineated
            head_rows.append(head_i)
            toe_rows.append(toe_i)
            head_geoms.append(head_i.geometry.iloc[0])
            toe_geoms.append(toe_i.geometry.iloc[0])
            head_elevs.append(float(head_i[cfg.elevation].iloc[0]))

        if len(head_geoms) > 1 and len(toe_geoms) > 1:
            poly_records.append({
                "geometry": Polygon(head_geoms + toe_geoms[::-1]),
                cfg.shoreline_id: S_i,
                cfg.head_elev_mean: float(np.mean(head_elevs)),
            })

    print(f'Total transects skipped: {skipped}')

    if head_rows:
        out_head = gpd.GeoDataFrame(pd.concat(head_rows, ignore_index=True), crs=crs)
        out_toe = gpd.GeoDataFrame(pd.concat(toe_rows, ignore_index=True), crs=crs)
        out_head.to_file(cfg.out_head_points)
        out_toe.to_file(cfg.out_toe_points)
    else:
        print("No head/toe points delineated; point outputs not written.")

    if poly_records:
        g_df = gpd.GeoDataFrame(poly_records, crs=crs)
        g_df.to_file(cfg.out_poly)
    else:
        print("No polygons generated; polygon output not written.")




if __name__ == "__main__":
    main()
