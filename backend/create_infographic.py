# This script generates an infographic of the HVC mine area with decorative
# tree illustrations. The style loosely follows the design referenced by
# "example.jpg" located in the data directory (if present).

import os
import random
from typing import List, Tuple

import shapefile  # pyshp
from shapely.geometry import shape, LineString, Polygon
from shapely.ops import transform
from pyproj import Transformer
from PIL import Image, ImageDraw

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RIVERS_SHP = os.path.join(DATA_DIR, "HVC_NamedStreams", "HVC_NamedStreams.shp")
MINE_SHP = os.path.join(
    DATA_DIR, "HVC_PermittedMineArea", "HVC_PermittedMineArea.shp"
)
OUTPUT_PATH = os.path.join(RESULTS_DIR, "hvc.jpg")

# Colours roughly inspired by the example infographic
BACKGROUND_COLOUR = (230, 240, 255)
RIVER_COLOUR = (65, 105, 225)
MINE_COLOUR = (205, 133, 63)
TREE_FILL = (34, 139, 34)
TREE_OUTLINE = (0, 100, 0)

IMAGE_SIZE = (1200, 900)
TREE_COUNT = 15


def load_shapes(shp_path: str) -> List[shape]:
    """Load all geometries from a shapefile."""
    sf = shapefile.Reader(shp_path)
    geoms = [shape(rec.__geo_interface__) for rec in sf.shapes()]
    return geoms


def get_total_bounds(geoms: List[shape]) -> Tuple[float, float, float, float]:
    minx = min(g.bounds[0] for g in geoms)
    miny = min(g.bounds[1] for g in geoms)
    maxx = max(g.bounds[2] for g in geoms)
    maxy = max(g.bounds[3] for g in geoms)
    return minx, miny, maxx, maxy


def project_geoms(geoms: List[shape]) -> List[shape]:
    """Project geometries from UTM zone 10N (EPSG:26910) to image coords."""
    transformer = Transformer.from_crs("epsg:26910", "epsg:4326", always_xy=True)
    return [transform(transformer.transform, g) for g in geoms]


def map_coords(x: float, y: float, bounds: Tuple[float, float, float, float]) -> Tuple[int, int]:
    """Map geographic coordinates to image pixel coordinates."""
    minx, miny, maxx, maxy = bounds
    w, h = IMAGE_SIZE
    scale_x = w / (maxx - minx)
    scale_y = h / (maxy - miny)
    px = int((x - minx) * scale_x)
    py = int(h - (y - miny) * scale_y)
    return px, py


def draw_tree(draw: ImageDraw.Draw, x: int, y: int, size: int) -> None:
    """Draw a simple triangular tree."""
    half = size // 2
    trunk_height = size // 3
    # Triangle for foliage
    draw.polygon(
        [(x, y - size), (x - half, y), (x + half, y)],
        fill=TREE_FILL,
        outline=TREE_OUTLINE,
    )
    # Trunk as a small rectangle
    draw.rectangle(
        [x - size // 10, y, x + size // 10, y + trunk_height],
        fill=(139, 69, 19),
        outline=(100, 50, 15),
    )


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rivers = load_shapes(RIVERS_SHP)
    mine_polys = load_shapes(MINE_SHP)

    all_geoms = rivers + mine_polys
    bounds = get_total_bounds(all_geoms)

    img = Image.new("RGB", IMAGE_SIZE, BACKGROUND_COLOUR)
    draw = ImageDraw.Draw(img)

    # Draw mine boundary polygons
    for poly in mine_polys:
        if isinstance(poly, Polygon):
            exterior = [map_coords(x, y, bounds) for x, y in poly.exterior.coords]
            draw.polygon(exterior, outline="black", fill=MINE_COLOUR)
            for interior in poly.interiors:
                interior_pts = [map_coords(x, y, bounds) for x, y in interior.coords]
                draw.polygon(interior_pts, outline="black", fill=BACKGROUND_COLOUR)

    # Draw rivers as blue lines
    for river in rivers:
        if isinstance(river, LineString):
            pts = [map_coords(x, y, bounds) for x, y in river.coords]
            draw.line(pts, fill=RIVER_COLOUR, width=3)

    # Add exaggerated random trees
    for _ in range(TREE_COUNT):
        rand_x = random.uniform(bounds[0], bounds[2])
        rand_y = random.uniform(bounds[1], bounds[3])
        px, py = map_coords(rand_x, rand_y, bounds)
        draw_tree(draw, px, py, size=40)

    img.save(OUTPUT_PATH, "JPEG")
    print(f"Saved infographic to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
