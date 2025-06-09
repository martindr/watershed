import os
import gzip
import logging
import argparse
import numpy as np
import plotly.graph_objects as go

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Default vertical exaggeration applied to the elevation data when plotting.
# Values < 1.0 will make the landscape appear flatter.
DEFAULT_EXAGGERATION = 0.00003

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


def plot_elevation(lat, lon, elevation, exaggeration: float = DEFAULT_EXAGGERATION):
    """Plot the elevation grid as a 3D surface."""
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

    start_index = exag_levels.index(exaggeration)
    fig.data[start_index].visible = True

    steps = []
    for i, level in enumerate(exag_levels):
        step = dict(
            method="update",
            args=[{"visible": [j == i for j in range(len(exag_levels))]}],
            label=str(level),
        )
        steps.append(step)

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
    )

    fig.show()    
    # output_html = os.path.join(os.path.dirname(__file__), "elevation_plot.html")
    # fig.write_html(output_html, auto_open=True)
    # logger.info(f"Saved plot to {output_html}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize elevation tiles')
    parser.add_argument(
        '--exaggeration',
        type=float,
        default=DEFAULT_EXAGGERATION,
        help='Vertical exaggeration factor (default: %(default)s)',
    )
    args = parser.parse_args()

    logger.info("Starting elevation visualization workflow")
    tiles = load_tiles()
    lat, lon, elev = merge_tiles(tiles)
    lat, lon, elev = crop_extent(
        lat,
        lon,
        elev,
        LATITUDE_MIN,
        LATITUDE_MAX,
        LONGITUDE_MIN,
        LONGITUDE_MAX,
    )
    plot_elevation(lat, lon, elev, exaggeration=args.exaggeration)
    logger.info("Visualization complete")
