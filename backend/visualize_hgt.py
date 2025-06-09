import os
import gzip
import numpy as np
import plotly.graph_objects as go

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def read_hgt(file_path):
    """Read an .hgt or .hgt.gz file into a 2D numpy array."""
    open_func = gzip.open if file_path.lower().endswith('.gz') else open
    with open_func(file_path, 'rb') as f:
        raw = f.read()
    data = np.frombuffer(raw, '>i2')
    size = int(np.sqrt(data.size))
    elev = data.reshape((size, size)).astype(float)
    elev[elev == -32768] = np.nan
    return elev


def lat_lon_from_filename(filename):
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    lat_sign = 1 if name[0] in ('N', 'n') else -1
    lat = int(name[1:3]) * lat_sign
    lon_sign = 1 if name[3] in ('E', 'e') else -1
    lon = int(name[4:7]) * lon_sign
    return lat, lon


def load_tiles():
    tiles = []
    for fname in os.listdir(DATA_DIR):
        if fname.lower().endswith(('.hgt', '.hgt.gz')):
            path = os.path.join(DATA_DIR, fname)
            elev = read_hgt(path)
            base = fname[:-3] if fname.lower().endswith('.gz') else fname
            lat, lon = lat_lon_from_filename(base)
            tiles.append((lat, lon, elev))
    return tiles


def merge_tiles(tiles):
    """Merge adjacent tiles into a single elevation array."""
    if not tiles:
        raise ValueError('No .hgt files found in data directory')
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
    return lat_coords, lon_coords, grid


def plot_elevation(lat, lon, elevation):
    fig = go.Figure(data=[go.Surface(z=elevation, x=lon, y=lat)])
    fig.update_layout(
        title='Terrain Elevation',
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Elevation (m)'
        ),
    )
    output_html = os.path.join(os.path.dirname(__file__), 'elevation_plot.html')
    fig.write_html(output_html, auto_open=True)
    print(f'Saved plot to {output_html}')


if __name__ == '__main__':
    tiles = load_tiles()
    lat, lon, elev = merge_tiles(tiles)
    plot_elevation(lat, lon, elev)
