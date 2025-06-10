import os
import gzip
import logging
import argparse
import numpy as np
import plotly.graph_objects as go
import shapefile
from pyproj import Transformer
from shapely.geometry import LineString, Polygon, box

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RIVERS_SHP = os.path.join(DATA_DIR, 'HVC_NamedStreams', 'HVC_NamedStreams.shp')
MINE_SHP = os.path.join(DATA_DIR, 'HVC_PermittedMineArea', 'HVC_PermittedMineArea.shp')

# Default vertical exaggeration applied to the elevation data when plotting.
# Values < 1.0 will make the landscape appear flatter.
DEFAULT_EXAGGERATION = 0.00003

# Distance (in meters) to extend callout lines above the terrain
CALLOUT_LINE_HEIGHT_M = 500

# Parameters controlling the appearance of generated trees
TREE_TRUNK_HEIGHT_M = 15
TREE_CANOPY_HEIGHT_M = 25

# Area of interest bounding box
LONGITUDE_MIN, LONGITUDE_MAX = -121.363525, -120.7
LATITUDE_MIN, LATITUDE_MAX = 50.2, 50.7

# Configure basic logging so we can trace the data loaded and processed
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def read_hgt(file_path):
    """Read an .hgt or .hgt.gz file into a 2D numpy array."""
    logger.info(f"Reading elevation file: {file_path}")
    open_func = gzip.open if file_path.lower().endswith('.gz') else open
    with open_func(file_path, 'rb') as f:
        raw = f.read()
    data = np.frombuffer(raw, '>i2')
    size = int(np.sqrt(data.size))
    elev = data.reshape((size, size)).astype(float)
    elev[elev == -32768] = np.nan
    logger.info(
        f"Loaded {file_path} with shape {elev.shape}, "
        f"min {np.nanmin(elev):.2f}, max {np.nanmax(elev):.2f}"
    )
    return elev


def lat_lon_from_filename(filename):
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    lat_sign = 1 if name[0] in ('N', 'n') else -1
    lat = int(name[1:3]) * lat_sign
    lon_sign = 1 if name[3] in ('E', 'e') else -1
    lon = int(name[4:7]) * lon_sign
    coords = (lat, lon)
    logger.debug(f"Extracted coordinates {coords} from {filename}")
    return coords


def load_tiles():
    logger.info(f"Loading tiles from {DATA_DIR}")
    tiles = []
    for fname in os.listdir(DATA_DIR):
        if fname.lower().endswith(('.hgt', '.hgt.gz')):
            path = os.path.join(DATA_DIR, fname)
            elev = read_hgt(path)
            base = fname[:-3] if fname.lower().endswith('.gz') else fname
            lat, lon = lat_lon_from_filename(base)
            logger.info(f"Loaded tile {fname} at lat={lat}, lon={lon}")
            tiles.append((lat, lon, elev))
    logger.info(f"Total tiles loaded: {len(tiles)}")
    return tiles


def merge_tiles(tiles):
    """Merge adjacent tiles into a single elevation array."""
    if not tiles:
        raise ValueError('No .hgt files found in data directory')
    logger.info("Merging tiles into a single grid")
    lats = sorted(set(lat for lat, _, _ in tiles))
    lons = sorted(set(lon for _, lon, _ in tiles))
    tile_size = tiles[0][2].shape[0]
    grid = np.full((len(lats) * tile_size, len(lons) * tile_size), np.nan, dtype=float)
    for lat, lon, data in tiles:
        i = lats.index(lat)
        j = lons.index(lon)
        row = (len(lats) - 1 - i) * tile_size
        col = j * tile_size
        grid[row:row + tile_size, col:col + tile_size] = data.astype(float)
    lat0 = lats[0]
    lon0 = lons[0]
    lat_coords = np.linspace(lat0, lat0 + len(lats), grid.shape[0])
    lon_coords = np.linspace(lon0, lon0 + len(lons), grid.shape[1])
    logger.info(
        f"Merged grid shape {grid.shape}, lat range {lat_coords[0]}-{lat_coords[-1]}, "
        f"lon range {lon_coords[0]}-{lon_coords[-1]}"
    )
    return lat_coords, lon_coords, grid


def crop_extent(lat, lon, elevation, lat_min, lat_max, lon_min, lon_max):
    """Crop the elevation grid to the specified latitude and longitude bounds."""
    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    lon_mask = (lon >= lon_min) & (lon <= lon_max)
    cropped_elev = elevation[np.ix_(lat_mask, lon_mask)]
    cropped_lat = lat[lat_mask]
    cropped_lon = lon[lon_mask]
    logger.info(
        "Cropped grid to lat %.3f-%.3f, lon %.3f-%.3f -> shape %s",
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        cropped_elev.shape,
    )
    return cropped_lat, cropped_lon, cropped_elev


def cut_north_of_line(lat, lon, elevation, pt1, pt2):
    """Remove all elevation data north of the line defined by pt1 and pt2.

    Parameters
    ----------
    lat : np.ndarray
        1D array of latitude coordinates.
    lon : np.ndarray
        1D array of longitude coordinates.
    elevation : np.ndarray
        2D elevation grid with shape ``(lat.size, lon.size)``.
    pt1 : tuple[float, float]
        ``(lat, lon)`` of the first point on the line.
    pt2 : tuple[float, float]
        ``(lat, lon)`` of the second point on the line.

    Returns
    -------
    np.ndarray
        The modified elevation array with values north of the line set to NaN.
    """

    lat1, lon1 = pt1
    lat2, lon2 = pt2
    slope = (lat2 - lat1) / (lon2 - lon1)
    intercept = lat1 - slope * lon1

    logger.info(
        "Cutting terrain north of line between (%.6f, %.6f) and (%.6f, %.6f)",
        lat1,
        lon1,
        lat2,
        lon2,
    )

    for j, lon_val in enumerate(lon):
        cut_lat = slope * lon_val + intercept
        mask = lat > cut_lat
        elevation[mask, j] = np.nan

    return elevation


def load_rivers(lat_grid, lon_grid, elev_grid, shp_path=RIVERS_SHP):
    """Load river polylines clipped to the area of interest."""
    if not os.path.exists(shp_path):
        logger.warning("River shapefile not found: %s", shp_path)
        return []

    logger.info("Loading rivers from %s", shp_path)
    sf = shapefile.Reader(shp_path)
    transformer = Transformer.from_crs("epsg:26910", "epsg:4326", always_xy=True)

    bbox = box(
        LONGITUDE_MIN,
        LATITUDE_MIN,
        LONGITUDE_MAX,
        LATITUDE_MAX,
    )

    rivers = []
    for shp in sf.shapes():
        coords = [transformer.transform(x, y) for x, y in shp.points]
        line = LineString([(lon, lat) for lon, lat in coords])
        clipped = line.intersection(bbox)
        if clipped.is_empty:
            continue

        if clipped.geom_type == "LineString":
            segments = [clipped]
        elif clipped.geom_type == "MultiLineString":
            segments = list(clipped.geoms)
        else:
            continue

        for seg in segments:
            lons, lats = seg.xy
            zs = []
            for lat_pt, lon_pt in zip(lats, lons):
                i = int(np.abs(lat_grid - lat_pt).argmin())
                j = int(np.abs(lon_grid - lon_pt).argmin())
                zs.append(elev_grid[i, j])
            rivers.append((np.array(lats), np.array(lons), np.array(zs)))

    logger.info("Loaded %d river polylines", len(rivers))
    return rivers


def load_mine_boundary(lat_grid, lon_grid, elev_grid, shp_path=MINE_SHP):
    """Load mine boundary polygons as line segments clipped to the area."""
    if not os.path.exists(shp_path):
        logger.warning("Mine boundary shapefile not found: %s", shp_path)
        return []

    logger.info("Loading mine boundary from %s", shp_path)
    sf = shapefile.Reader(shp_path)
    transformer = Transformer.from_crs("epsg:26910", "epsg:4326", always_xy=True)

    bbox = box(
        LONGITUDE_MIN,
        LATITUDE_MIN,
        LONGITUDE_MAX,
        LATITUDE_MAX,
    )

    boundaries = []
    for shp in sf.shapes():
        coords = [transformer.transform(x, y) for x, y in shp.points]
        poly = Polygon([(lon, lat) for lon, lat in coords])
        boundary = poly.boundary
        clipped = boundary.intersection(bbox)
        if clipped.is_empty:
            continue

        if clipped.geom_type == "LineString":
            geoms = [clipped]
        elif clipped.geom_type in ("MultiLineString", "GeometryCollection"):
            geoms = [g for g in clipped.geoms if g.geom_type == "LineString"]
        else:
            continue

        for seg in geoms:
            lons, lats = seg.xy
            zs = []
            for lat_pt, lon_pt in zip(lats, lons):
                i = int(np.abs(lat_grid - lat_pt).argmin())
                j = int(np.abs(lon_grid - lon_pt).argmin())
                zs.append(elev_grid[i, j])
            boundaries.append((np.array(lats), np.array(lons), np.array(zs)))

    logger.info("Loaded %d mine boundary segment(s)", len(boundaries))
    return boundaries


def generate_tree_points(lat_grid, lon_grid, elev_grid, count=20, seed=42):
    """Randomly sample points on the grid to place hypothetical trees."""
    rng = np.random.default_rng(seed)
    lat_indices = rng.integers(0, len(lat_grid), size=count)
    lon_indices = rng.integers(0, len(lon_grid), size=count)
    trees = []
    for i, j in zip(lat_indices, lon_indices):
        trees.append((lat_grid[i], lon_grid[j], float(elev_grid[i, j])))
    logger.info("Generated %d tree location(s)", len(trees))
    return trees


def plot_elevation(
    lat,
    lon,
    elevation,
    *,
    cut_elevation=None,
    exaggeration: float = DEFAULT_EXAGGERATION,
    callouts=None,
    rivers=None,
    cut_rivers=None,
    mine_boundary=None,
    cut_mine_boundary=None,
    trees=None,
    cut_trees=None,
    apply_cutout: bool = True,
):
    """Plot the elevation grid as a 3D surface with optional callout points.

    Parameters
    ----------
    lat, lon : np.ndarray
        1D arrays defining the grid coordinates.
    elevation : np.ndarray
        Elevation grid without any cut applied.
    cut_elevation : np.ndarray or None
        Elevation grid with the cutout applied. If provided, a checkbox will be
        added to toggle between the full and cut views.
    apply_cutout : bool
        Whether the cutout should be enabled by default when the plot loads.
    """
    logger.info("Generating elevation plot")

    exag_levels = [0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009, 0.0001]
    if exaggeration not in exag_levels:
        exag_levels.insert(0, exaggeration)

    fig = go.Figure()
    for level in exag_levels:
        fig.add_surface(
            z=elevation * level,
            x=lon,
            y=lat,
            visible=False,
            colorscale="Earth",
        )
    if cut_elevation is not None:
        for level in exag_levels:
            fig.add_surface(
                z=cut_elevation * level,
                x=lon,
                y=lat,
                visible=False,
                colorscale="Earth",
            )

    callouts = callouts or []
    n_points = len(callouts)
    if n_points:
        logger.info("Adding %d callout point(s)", n_points)
    for level_idx, level in enumerate(exag_levels):
        for name, lat_pt, lon_pt in callouts:
            i = int(np.abs(lat - lat_pt).argmin())
            j = int(np.abs(lon - lon_pt).argmin())
            base = elevation[i, j] * level
            top = base + CALLOUT_LINE_HEIGHT_M * level
            fig.add_trace(
                go.Scatter3d(
                    x=[lon[j], lon[j]],
                    y=[lat[i], lat[i]],
                    z=[base, top],
                    mode="lines+text",
                    text=["", name],
                    textposition="top center",
                    line=dict(color="red", width=2),
                    visible=False,
                )
            )
    if cut_elevation is not None:
        for level_idx, level in enumerate(exag_levels):
            for name, lat_pt, lon_pt in callouts:
                i = int(np.abs(lat - lat_pt).argmin())
                j = int(np.abs(lon - lon_pt).argmin())
                base = cut_elevation[i, j] * level
                top = base + CALLOUT_LINE_HEIGHT_M * level
                fig.add_trace(
                    go.Scatter3d(
                        x=[lon[j], lon[j]],
                        y=[lat[i], lat[i]],
                        z=[base, top],
                        mode="lines+text",
                        text=["", name],
                        textposition="top center",
                        line=dict(color="red", width=2),
                        visible=False,
                    )
                )

    # Trees
    trees = trees or []
    n_trees = len(trees)
    if n_trees:
        logger.info("Adding %d tree(s)", n_trees)
    for level_idx, level in enumerate(exag_levels):
        for lat_pt, lon_pt, elev_pt in trees:
            base = elev_pt * level
            trunk_top = base + TREE_TRUNK_HEIGHT_M * level
            fig.add_trace(
                go.Scatter3d(
                    x=[lon_pt, lon_pt],
                    y=[lat_pt, lat_pt],
                    z=[base, trunk_top],
                    mode="lines",
                    line=dict(color="sienna", width=3),
                    visible=False,
                )
            )
            fig.add_trace(
                go.Cone(
                    x=[lon_pt],
                    y=[lat_pt],
                    z=[trunk_top],
                    u=[0],
                    v=[0],
                    w=[TREE_CANOPY_HEIGHT_M * level],
                    anchor="tail",
                    colorscale="Greens",
                    showscale=False,
                    visible=False,
                )
            )
    if cut_elevation is not None and cut_trees is not None:
        for level_idx, level in enumerate(exag_levels):
            for lat_pt, lon_pt, elev_pt in cut_trees:
                base = elev_pt * level
                trunk_top = base + TREE_TRUNK_HEIGHT_M * level
                fig.add_trace(
                    go.Scatter3d(
                        x=[lon_pt, lon_pt],
                        y=[lat_pt, lat_pt],
                        z=[base, trunk_top],
                        mode="lines",
                        line=dict(color="sienna", width=3),
                        visible=False,
                    )
                )
                fig.add_trace(
                    go.Cone(
                        x=[lon_pt],
                        y=[lat_pt],
                        z=[trunk_top],
                        u=[0],
                        v=[0],
                        w=[TREE_CANOPY_HEIGHT_M * level],
                        anchor="tail",
                        colorscale="Greens",
                        showscale=False,
                        visible=False,
                    )
                )

    rivers = rivers or []
    n_rivers = len(rivers)
    if n_rivers:
        logger.info("Adding %d river trace(s)", n_rivers)
    for level_idx, level in enumerate(exag_levels):
        xs, ys, zs = [], [], []
        for r_lat, r_lon, r_z in rivers:
            xs.extend(r_lon)
            xs.append(None)
            ys.extend(r_lat)
            ys.append(None)
            zs.extend(r_z * level)
            zs.append(None)
        if xs:
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line=dict(color="blue", width=2),
                    name="Rivers",
                    visible=False,
                )
            )
    if cut_rivers is not None:
        for level_idx, level in enumerate(exag_levels):
            xs, ys, zs = [], [], []
            for r_lat, r_lon, r_z in cut_rivers:
                xs.extend(r_lon)
                xs.append(None)
                ys.extend(r_lat)
                ys.append(None)
                zs.extend(r_z * level)
                zs.append(None)
            if xs:
                fig.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="lines",
                        line=dict(color="blue", width=2),
                        name="Rivers",
                        visible=False,
                    )
                )

    mine_boundary = mine_boundary or []
    n_mine = len(mine_boundary)
    if n_mine:
        logger.info("Adding %d mine boundary trace(s)", n_mine)
    for level_idx, level in enumerate(exag_levels):
        xs, ys, zs = [], [], []
        for r_lat, r_lon, r_z in mine_boundary:
            xs.extend(r_lon)
            xs.append(None)
            ys.extend(r_lat)
            ys.append(None)
            zs.extend(r_z * level)
            zs.append(None)
        if xs:
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line=dict(color="orange", width=3),
                    name="HVC Mine", 
                    visible=False,
                )
            )
    if cut_mine_boundary is not None:
        for level_idx, level in enumerate(exag_levels):
            xs, ys, zs = [], [], []
            for r_lat, r_lon, r_z in cut_mine_boundary:
                xs.extend(r_lon)
                xs.append(None)
                ys.extend(r_lat)
                ys.append(None)
                zs.extend(r_z * level)
                zs.append(None)
            if xs:
                fig.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="lines",
                        line=dict(color="orange", width=3),
                        name="HVC Mine", 
                        visible=False,
                    )
                )

    start_index = exag_levels.index(exaggeration)
    total_levels = len(exag_levels)

    # Calculate index offsets for each trace group so we can toggle them later
    offset = total_levels
    cut_surface_offset = None
    if cut_elevation is not None:
        cut_surface_offset = offset
        offset += total_levels
    callout_offset = offset
    offset += total_levels * n_points
    cut_callout_offset = None
    if cut_elevation is not None:
        cut_callout_offset = offset
        offset += total_levels * n_points
    rivers_offset = offset if n_rivers else None
    if n_rivers:
        offset += total_levels
    cut_rivers_offset = None
    if cut_elevation is not None and cut_rivers is not None:
        cut_rivers_offset = offset
        if n_rivers:
            offset += total_levels
    mine_offset = offset if n_mine else None
    if n_mine:
        offset += total_levels
    cut_mine_offset = None
    if cut_elevation is not None and cut_mine_boundary is not None:
        cut_mine_offset = offset
        if n_mine:
            offset += total_levels
    trees_offset = offset if n_trees else None
    if n_trees:
        offset += total_levels * 2 * n_trees
    cut_trees_offset = None
    if cut_elevation is not None and cut_trees is not None and n_trees:
        cut_trees_offset = offset
        offset += total_levels * 2 * n_trees

    # Hide everything by default
    for trace in fig.data:
        trace.visible = False

    if apply_cutout and cut_elevation is not None:
        fig.data[cut_surface_offset + start_index].visible = True
        for idx in range(n_points):
            fig.data[cut_callout_offset + start_index * n_points + idx].visible = True
        if n_rivers and cut_rivers_offset is not None:
            fig.data[cut_rivers_offset + start_index].visible = True
        if n_mine and cut_mine_offset is not None:
            fig.data[cut_mine_offset + start_index].visible = True
        if n_trees and cut_trees_offset is not None:
            for idx in range(2 * n_trees):
                fig.data[cut_trees_offset + start_index * 2 * n_trees + idx].visible = True
    else:
        fig.data[start_index].visible = True
        for idx in range(n_points):
            fig.data[callout_offset + start_index * n_points + idx].visible = True
        if n_rivers:
            fig.data[rivers_offset + start_index].visible = True
        if n_mine:
            fig.data[mine_offset + start_index].visible = True
        if n_trees:
            for idx in range(2 * n_trees):
                fig.data[trees_offset + start_index * 2 * n_trees + idx].visible = True

    steps = []
    for i, level in enumerate(exag_levels):
        visible = []
        # Full surfaces
        for j in range(total_levels):
            visible.append(j == i)
        # Cut surfaces
        if cut_elevation is not None:
            for j in range(total_levels):
                visible.append(j == i)
        # Callouts full
        for j in range(total_levels):
            for _ in range(n_points):
                visible.append(j == i)
        # Callouts cut
        if cut_elevation is not None:
            for j in range(total_levels):
                for _ in range(n_points):
                    visible.append(j == i)
        # Rivers full
        if n_rivers:
            for j in range(total_levels):
                visible.append(j == i)
        # Rivers cut
        if cut_elevation is not None and cut_rivers is not None and n_rivers:
            for j in range(total_levels):
                visible.append(j == i)
        # Mine boundary full
        if n_mine:
            for j in range(total_levels):
                visible.append(j == i)
        # Mine boundary cut
        if cut_elevation is not None and cut_mine_boundary is not None and n_mine:
            for j in range(total_levels):
                visible.append(j == i)
        # Trees full
        if n_trees:
            for j in range(total_levels):
                for _ in range(2 * n_trees):
                    visible.append(j == i)
        # Trees cut
        if cut_elevation is not None and cut_trees is not None and n_trees:
            for j in range(total_levels):
                for _ in range(2 * n_trees):
                    visible.append(j == i)
        step = dict(method="update", args=[{"visible": visible}], label=str(level))
        steps.append(step)

    updatemenus = []

    fig.update_layout(
        title="Terrain Elevation",
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Elevation (m)",
            aspectmode="data",
        ),
        sliders=[dict(
            active=start_index,
            currentvalue={"prefix": "Exaggeration: "},
            pad={"t": 50},
            steps=steps,
        )],
        updatemenus=updatemenus,
    )

    # Build index lists for JavaScript toggling
    full_indices = list(range(total_levels))
    if cut_elevation is not None:
        cut_indices = list(range(cut_surface_offset, cut_surface_offset + total_levels))
    else:
        cut_indices = []
    full_indices += list(range(callout_offset, callout_offset + total_levels * n_points))
    if cut_elevation is not None:
        cut_indices += list(range(cut_callout_offset, cut_callout_offset + total_levels * n_points))
    river_indices = []
    if n_rivers:
        river_indices += list(range(rivers_offset, rivers_offset + total_levels))
        if cut_rivers_offset is not None:
            river_indices += list(range(cut_rivers_offset, cut_rivers_offset + total_levels))
            cut_indices += list(range(cut_rivers_offset, cut_rivers_offset + total_levels))
        else:
            full_indices += list(range(rivers_offset, rivers_offset + total_levels))
    mine_indices = []
    if n_mine:
        mine_indices += list(range(mine_offset, mine_offset + total_levels))
        if cut_mine_offset is not None:
            mine_indices += list(range(cut_mine_offset, cut_mine_offset + total_levels))
            cut_indices += list(range(cut_mine_offset, cut_mine_offset + total_levels))
        else:
            full_indices += list(range(mine_offset, mine_offset + total_levels))
    tree_indices = []
    if n_trees:
        tree_indices += list(range(trees_offset, trees_offset + total_levels * 2 * n_trees))
        if cut_trees_offset is not None:
            tree_indices += list(range(cut_trees_offset, cut_trees_offset + total_levels * 2 * n_trees))
            cut_indices += list(range(cut_trees_offset, cut_trees_offset + total_levels * 2 * n_trees))
        else:
            full_indices += list(range(trees_offset, trees_offset + total_levels * 2 * n_trees))
    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # checkbox_html = f"""
    # <div style='padding:10px;'>
    #   <label><input type='checkbox' id='cutoutToggle' {'checked' if apply_cutout else ''}> Cutout</label>
    #   <label style='margin-left:20px'><input type='checkbox' id='riversToggle' checked> Rivers</label>
    # </div>
    # """

    # script = f"""
    # <script>
    #   const plotId = document.getElementsByClassName('plotly-graph-div')[0].id;
    #   const cutIndices = {cut_indices};
    #   const fullIndices = {full_indices};
    #   const riverIndices = {river_indices};
    #   document.getElementById('cutoutToggle').addEventListener('change', (e) => {{
    #     const on = e.target.checked;
    #     Plotly.restyle(plotId, {{visible: on}}, cutIndices);
    #     Plotly.restyle(plotId, {{visible: !on}}, fullIndices);
    #   }});
    #   document.getElementById('riversToggle').addEventListener('change', (e) => {{
    #     const show = e.target.checked;
    #     Plotly.restyle(plotId, {{visible: show}}, riverIndices);
    #   }});
    # </script>
    # """

    output_html = os.path.join(os.path.dirname(__file__), "elevation_plot.html")
    with open(output_html, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'></head><body>")
        # f.write(checkbox_html)
        f.write(plot_html)
        # f.write(script)
        f.write("</body></html>")
    logger.info(f"Saved plot to {output_html}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize elevation tiles')
    parser.add_argument(
        '--exaggeration',
        type=float,
        default=DEFAULT_EXAGGERATION,
        help='Vertical exaggeration factor (default: %(default)s)',
    )
    parser.add_argument(
        "--no-cutout",
        action="store_true",
        help="Start with the cutout disabled",
    )
    parser.add_argument(
        "--trees",
        type=int,
        default=0,
        help="Number of hypothetical trees to generate",
    )

    args = parser.parse_args()

    logger.info("Starting elevation visualization workflow")
    tiles = load_tiles()
    lat, lon, elev = merge_tiles(tiles)
    lat, lon, elev_full = crop_extent(
        lat,
        lon,
        elev,
        LATITUDE_MIN,
        LATITUDE_MAX,
        LONGITUDE_MIN,
        LONGITUDE_MAX,
    )

    # Create cutout version of the elevation grid
    elev_cut = cut_north_of_line(
        lat,
        lon,
        elev_full.copy(),
        (50.592647, -121.324994),  # Pukaist Creek
        (50.426341, -120.947810),  # Witches Brook
    )

    rivers_full = load_rivers(lat, lon, elev_full)
    rivers_cut = load_rivers(lat, lon, elev_cut)
    mine_full = load_mine_boundary(lat, lon, elev_full)
    mine_cut = load_mine_boundary(lat, lon, elev_cut)

    elev_downsampled = elev_full[::5, ::5]
    elev_cut_downsampled = elev_cut[::5, ::5]
    callouts = [
        ("Pukaist Creek", 50.592647, -121.324994),
        ("Witches Brook", 50.426341, -120.947810),
    ]

    trees_full = []
    trees_cut = []
    if args.trees > 0:
        trees_full = generate_tree_points(lat[::5], lon[::5], elev_downsampled, count=args.trees)
        for lat_pt, lon_pt, _ in trees_full:
            i = int(np.abs(lat[::5] - lat_pt).argmin())
            j = int(np.abs(lon[::5] - lon_pt).argmin())
            trees_cut.append((lat_pt, lon_pt, float(elev_cut_downsampled[i, j])))
    plot_elevation(
        lat[::5],
        lon[::5],
        elev_downsampled,
        cut_elevation=elev_cut_downsampled,
        exaggeration=args.exaggeration,
        callouts=callouts,
        rivers=rivers_full,
        cut_rivers=rivers_cut,
        mine_boundary=mine_full,
        cut_mine_boundary=mine_cut,
        trees=trees_full,
        cut_trees=trees_cut,
        apply_cutout=not args.no_cutout,
    )

    logger.info("Visualization complete")
