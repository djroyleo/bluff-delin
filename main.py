import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import curve_fit
from shapely.geometry import Polygon

IN_POINTS = Path("./data/in/in_points.shp")
OUT_POINTS = Path("./data/out/out_points.shp")
OUT_POLY = Path("./data/out/out_poly.shp")
TRANSECT_FIELD_NAME = "TRANS_ID"
SHORELINE_FIELD_NAME = "SHORE_ID"
ELEVATION_FIELD_NAME = "ELEV"
MHHW_FIELD_NAME = "MHHW"

def get_data(points) -> gpd.GeoDataFrame:
    """
    mmm yummy data
    """

    data = gpd.read_file(points)
    data.replace(-9999, np.nan, inplace=True)
    data.replace(0, np.nan, inplace=True) # change depending on bit-level of DEM data

    return data

def extract_shore(data:gpd.GeoDataFrame, shore_id:int) -> gpd.GeoDataFrame:
    """
    
    """

    shore = data[data[f'{SHORELINE_FIELD_NAME}'] == shore_id].copy()

    return shore

def extract_prof(data:gpd.GeoDataFrame, prof_id:int) -> gpd.GeoDataFrame:
    """

    """

    prof = data[data[f'{TRANSECT_FIELD_NAME}'] == prof_id].copy()

    return prof

def _sigmoid(x, L, k, x0, b):

    return L / (1.0 + np.exp(-k * (x - x0))) + b

def delineate_head(prof:gpd.GeoDataFrame) -> gpd.GeoDataFrame | None:
    """

    """

    x: pd.Series[float] = prof['ORIG_LEN']
    y: pd.Series[float] = prof[f'{ELEVATION_FIELD_NAME}']

    if y.isnull().all(): return None

    mask = y.notnull()
    x_fit = x[mask].to_numpy(dtype=float)
    y_fit = y[mask].to_numpy(dtype=float)

    if len(x_fit) < 4: return None

    b0 = float(np.min(y_fit))
    L0 = float(np.max(y_fit) - np.min(y_fit))
    x0 = float(np.mean(x_fit))
    k0 = 1.0 if (y_fit[-1] - y_fit[0]) >= 0 else -1.0
    p0 = [L0 if L0 != 0 else 1.0, k0, x0, b0]

    try:
        popt, _ = curve_fit(_sigmoid, x_fit, y_fit, p0=p0, maxfev=10000)
    except (RuntimeError, ValueError):
        return None

    x_all = x.to_numpy(dtype=float)
    y_sig = _sigmoid(x_all, *popt)

    deriv2 = np.gradient(np.gradient(y_sig))

    x_val = x.iat[int(np.nanargmin(deriv2))]
    head = prof[prof['ORIG_LEN'] == x_val].copy()

    return head

def extract_toe(prof:gpd.GeoDataFrame) -> gpd.GeoDataFrame | None:
    """

    """

    x: pd.Series[float] = prof['ORIG_LEN']
    y: pd.Series[float] = prof[f'{ELEVATION_FIELD_NAME}']
    mhhw: pd.Series[float] = prof[f'{MHHW_FIELD_NAME}']

    if y.isnull().all(): return None

    mhhw_diff = np.abs(y - mhhw)

    x_val = x.iat[int(mhhw_diff.argmin())]
    toe = prof[prof['ORIG_LEN'] == x_val].copy()

    return toe



def main():
    in_points = get_data(IN_POINTS)
    shore_set = 
    out_head = gpd.GeoDataFrame(columns=in_points.columns)
    out_toe = gpd.GeoDataFrame(columns=in_points.columns)
    poly_list = []
    skipped = 0

    for S_i in tqdm()
    


if __name__ == "__main__":
    main()
