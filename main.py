print("bluff-delin v0.1.0")
print("Copyright (c) 2025, Dylan J. Roy-Leo, All rights reserved.")
print("Importing libraries...")
import argparse
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
from shapely.geometry import Polygon
print("Library import complete.")

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
    plot: bool
    plot_id: tuple[int, ...]

def load_config(path: Path) -> Config:
    """
    Read a TOML configuration file into a Config.
    """

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    paths = raw["paths"]
    fields = raw["fields"]
    nodata = raw.get("nodata", {})
    plotting = raw.get("plotting", {})

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
        plot=plotting.get("plot", False),
        plot_id=tuple(plotting.get("plot_id", [])),
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

    # orient the shore profile so that distance increases in the landward direction

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

def _fit_sigmoid(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """
    Least-squares sigmoid fit to the valid (non-NaN elevation) points of one
    profile. Returns the fitted parameters (L, k, x0, b), or None if there are
    too few valid points or the fit fails.

    Parameters are canonicalized to the L >= 0 form: (L, k, x0, b) and
    (-L, -k, x0, b+L) describe the identical curve, and an unbounded fit
    converges to either, so the sign of a raw fitted k is arbitrary. Flipping
    to L >= 0 leaves the curve (and its derivatives) untouched while making k
    comparable between transects: positive k rises with increasing distance.
    """

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

    L, k, x0, b = popt
    if L < 0:
        popt = np.array([-L, -k, x0, b + L])

    return popt

# The curvature minimum lands slightly seaward of the true bluff head, so the
# head is stepped this many points landward. Consecutive points along a
# transect are spaced 1 m apart in the input data, so this is a 5 m shift;
# revisit the value if the sampling interval ever changes.
HEAD_SHIFT = 5

def delineate_transect(prof: gpd.GeoDataFrame, cfg: Config) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | None:
    """
    Delineate the toe and head of one transect profile (sorted by distance).

    The toe is the point where elevation is closest to MHHW. The head is the
    point of maximum concave curvature (minimum second derivative) of a
    least-squares sigmoid fit, taken over all points with valid elevation;
    the second-derivative criterion is direction-insensitive, so no
    landward/seaward restriction is applied. That point sits slightly seaward
    of the true head, so it is then stepped HEAD_SHIFT points landward,
    clamped to the profile. Returns None if the fit fails or either point
    cannot be delineated.
    """

    x = prof[cfg.distance].to_numpy(dtype=float)
    y = prof[cfg.elevation].to_numpy(dtype=float)
    mhhw = prof[cfg.mhhw].to_numpy(dtype=float)

    mhhw_diff = np.abs(y - mhhw)
    if np.isnan(mhhw_diff).all():
        return None
    toe_pos = int(np.nanargmin(mhhw_diff))

    y_valid = ~np.isnan(y)
    popt = _fit_sigmoid(x, y)
    if popt is None:
        return None

    deriv2 = _sigmoid_deriv2(x, *popt)
    deriv2[~y_valid] = np.nan
    try:
        head_pos = int(np.nanargmin(deriv2))
    except ValueError:
        return None

    # step the head landward; popt is canonicalized to L >= 0, so the sign of
    # k gives the landward direction: k > 0 rises with increasing distance.
    # Clamping matters -- a bare negative index would silently wrap around to
    # the far, seaward end of the transect.
    head_pos += HEAD_SHIFT if popt[1] > 0 else -HEAD_SHIFT
    head_pos = int(np.clip(head_pos, 0, len(x) - 1))

    head = prof.iloc[[head_pos]].copy()
    toe = prof.iloc[[toe_pos]].copy()

    return head, toe

def plot_transect(prof: gpd.GeoDataFrame, head: gpd.GeoDataFrame, toe: gpd.GeoDataFrame,
                  prof_id: int, cfg: Config) -> None:
    """
    Plot one delineated transect profile: elevation and the fitted sigmoid as
    lines, with the identified head and toe marked as points, annotated with
    the transect ID and this transect's own fitted k (shown as n/a if the fit
    failed). Figures are created but not shown; call plt.show() once
    afterwards to display all of them.
    """

    x = prof[cfg.distance].to_numpy(dtype=float)
    y = prof[cfg.elevation].to_numpy(dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))
    ax1.plot(x, y, color="tab:blue", label="elevation")

    popt = _fit_sigmoid(x, y)
    if popt is not None:
        x_dense = np.linspace(np.nanmin(x), np.nanmax(x), 500)
        second_deriv = _sigmoid_deriv2(x_dense, *popt)
        ax2.plot(x_dense, second_deriv, color="tab:purple", label="sigmoid 2nd deriv")
        ax1.plot(x_dense, _sigmoid(x_dense, *popt), color="tab:orange",
                linestyle="--", label="sigmoid fit")

    ax1.scatter(head[cfg.distance], head[cfg.elevation], color="tab:red",
               zorder=3, label="head")
    ax1.scatter(toe[cfg.distance], toe[cfg.elevation], color="tab:green",
               zorder=3, label="toe")

    k_txt = f"k = {popt[1]:.4f}" if popt is not None else "k = n/a"
    ax1.text(0.02, 0.98, f"{cfg.transect_id}: {prof_id}\n{k_txt}",
            transform=ax1.transAxes, va="top")
    ax1.set_xlabel(cfg.distance)
    ax1.set_ylabel(cfg.elevation)
    ax1.legend(loc="lower right")
    ax2.set_xlabel(cfg.distance)
    ax2.set_ylabel("2nd derivative")
    ax2.legend(loc="lower right")
    fig.tight_layout()

def main():
    print("Initializing bluff delineation...")
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

    for S_i in tqdm(shore_set, ascii=True, desc="Delineating bluff geometry"):
        head_geoms = []
        toe_geoms = []
        head_elevs = []

        shore = extract_shore(in_points, S_i, cfg)
        profiles_for_shore = sorted(shore[cfg.transect_id].unique())

        for P_i in profiles_for_shore:
            profile_i = extract_prof(data=shore, prof_id=P_i, cfg=cfg)
            delineated = delineate_transect(profile_i, cfg)

            if delineated is None:
                if cfg.plot and P_i in cfg.plot_id:
                    tqdm.write(f"Transect {P_i} was skipped; no plot generated.")
                skipped += 1
                continue

            head_i, toe_i = delineated
            if cfg.plot and P_i in cfg.plot_id:
                plot_transect(profile_i, head_i, toe_i, P_i, cfg)
            head_rows.append(head_i)
            toe_rows.append(toe_i)
            head_geoms.append(head_i.geometry.iloc[0])
            toe_geoms.append(toe_i.geometry.iloc[0])
            head_elevs.append(float(head_i[cfg.elevation].iloc[0]))

        if len(head_geoms) > 1 and len(toe_geoms) > 1:
            poly_records.append({
                "geometry": Polygon(head_geoms + toe_geoms[::-1]),
                cfg.shoreline_id: S_i,
                # nanmean: the landward shift can land the head on a nodata
                # point, and one NaN would otherwise void the whole mean
                cfg.head_elev_mean: float(np.nanmean(head_elevs)) if not np.isnan(head_elevs).all() else np.nan,
            })

    print("Bluff delineation complete.")
    print(f'Total transects skipped: {skipped}')

    if not cfg.plot:
        print("Writing outputs...")
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
        print("All outputs written.")

    if cfg.plot and plt.get_fignums():
        print("Displaying transect plots...")
        plt.show()




if __name__ == "__main__":
    main()
