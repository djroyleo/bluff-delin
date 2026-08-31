import numpy as np

import geopandas as gpd
from pathlib import Path

IN_POINTS = Path("./data/in/in_points.shp")
OUT_POINTS = Path("./data/out/out_points.shp")
OUT_POLY = Path("./data/out/out_poly.shp")


def get_data(points) -> gpd.GeoDataFrame:
    return points


def main():
    print("Hello from bluff-delin!")


if __name__ == "__main__":
    main()
