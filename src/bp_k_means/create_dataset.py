
import logging
import zipfile
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests

INE_ZIP_URL = "https://www.ine.es/prodyser/cartografia/seccionado_2025.zip"
DATA_DIR = Path("data/downloads")
TARGET_CRS = "EPSG:25830"  # ETRS89 / UTM 30N
OUTPUT_DIR = Path("data/datasets")

# Columns typically present in SECC_CE_20250101.shp
SECC_ATTRIBUTE_COLS = [
    "CUSEC",   # census section id
    "CUMUN",   # municipality code (province + municipality, 5 digits)
    "CSEC",    # section code within municipality
    "CDIS",    # district code
    "CMUN",    # municipality code within province (3 digits)
    "CPRO",    # province code (2 digits)
    "CCA",     # autonomous community code
    "CUDIS",   # district id
    "CLAU2",   # internal key
    "NPRO",    # province name
    "NCA",     # autonomous community name
    "CNUT0",   # NUTS 0 code
    "CNUT1",   # NUTS 1 code
    "CNUT2",   # NUTS 2 code
    "CNUT3",   # NUTS 3 code
    "NMUN",    # municipality name
]

# From smallest to largest region
LEVEL_PRIORITY = ["CUSEC", "CUMUN", "CMUN", "CPRO", "CCA"]


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_seccionado_zip() -> Path:
    """
    Download seccionado_2025.zip from INE if it is not already present.
    """
    ensure_dirs()
    zip_path = DATA_DIR / "seccionado_2025.zip"

    if zip_path.exists():
        return zip_path

    logging.info(f"Downloading INE seccionado 2025 from {INE_ZIP_URL}")
    resp = requests.get(INE_ZIP_URL, stream=True)
    resp.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    logging.info(f"Saved {zip_path}")
    return zip_path


def extract_seccionado_zip(zip_path: Path) -> Path:
    """
    Extract seccionado_2025.zip into data directory if not already extracted.
    """
    extract_dir = DATA_DIR / "extracted"
    if extract_dir.exists():
        return extract_dir

    logging.info(f"Extracting {zip_path}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    logging.info(f"Extracted to {extract_dir}")
    return extract_dir


def find_seccionado_shapefile(extract_dir: Path) -> Path:
    """
    Find the SECC_CE_20250101.shp shapefile inside the extracted tree.
    """
    for shp in extract_dir.rglob("*.shp"):
        if "SECC_CE_20250101" in shp.name:
            return shp
    raise FileNotFoundError(f"No SECC_CE_20250101.shp found under {extract_dir}")


def load_seccionado_gdf() -> gpd.GeoDataFrame:
    """
    Ensure seccionado_2025 is downloaded and extracted, then load
    SECC_CE_20250101.shp into a GeoDataFrame in TARGET_CRS.
    """
    zip_path = download_seccionado_zip()
    extract_dir = extract_seccionado_zip(zip_path)
    shp_path = find_seccionado_shapefile(extract_dir)

    logging.info(f"Reading sections from {shp_path}")
    secc = gpd.read_file(shp_path)

    if secc.crs is None:
        secc = secc.set_crs(TARGET_CRS)
    elif secc.crs.to_string() != TARGET_CRS:
        secc = secc.to_crs(TARGET_CRS)

    logging.info(f"Loaded {len(secc)} census sections")
    return secc


def normalize_filter_values(allowed):
    """
    Normalise filter values to a set of strings.
    """
    if isinstance(allowed, (list, tuple, set)):
        return {str(v) for v in allowed}
    return {str(allowed)}


def filter_sections_by_codes(
    secc: gpd.GeoDataFrame,
    filter_by_codes: dict | None,
) -> gpd.GeoDataFrame:
    """
    Filter sections using all given codes at once, ordered from most to least
    restrictive according to LEVEL_PRIORITY.
    Example: {"CPRO": "28", "CMUN": "079"} for Madrid municipality.
    """
    if not filter_by_codes:
        return secc

    # sort keys by hierarchy: CUSEC > CUMUN > CMUN > CPRO > CCA
    order = {col: i for i, col in enumerate(LEVEL_PRIORITY)}
    cols_sorted = sorted(
        filter_by_codes.keys(),
        key=lambda c: order.get(c, len(order)),  # unknown columns go last
    )

    mask = pd.Series(True, index=secc.index)

    for col in cols_sorted:
        if col not in secc.columns:
            raise KeyError(f"Column {col} is not in sections file")
        allowed_str = normalize_filter_values(filter_by_codes[col])
        mask &= secc[col].astype(str).isin(allowed_str)

    return secc[mask].copy()


def clip_sections_to_nodes_bbox(
    secc: gpd.GeoDataFrame,
    nodes_utm: gpd.GeoDataFrame,
    margin_m: float = 2000.0,
) -> gpd.GeoDataFrame:
    """
    Bounding box clip of secc around the extent of nodes_utm, expanded by margin_m.
    Useful when you do not filter by codes.
    """
    minx, miny, maxx, maxy = nodes_utm.total_bounds
    minx -= margin_m
    miny -= margin_m
    maxx += margin_m
    maxy += margin_m
    secc_region = secc.cx[minx:maxx, miny:maxy].copy()
    return secc_region


def build_dataset_for_place(
    place: str,
    outfile: str,
    network_type: str = "drive",
    filter_by_codes: dict | None = None,
    drop_unmatched: bool = True,
) -> None:
    """
    Create one clustering dataset.

    Parameters
    ----------
    place : str
        Place name understood by osmnx, for example "Madrid, Spain".
    outfile : str
        Output Parquet file name or path.
    network_type : str
        osmnx network_type ("drive", "walk", "all", etc.).
    filter_by_codes : dict or None
        Dict like {"CPRO": "28", "CMUN": "079"} to restrict sections before join.
        If None, sections are just bbox clipped around the OSM nodes.
    drop_unmatched : bool
        If True, drop nodes that do not fall inside any census section.
    """
    ensure_dirs()
    out_path = Path(outfile)
    if not out_path.is_absolute():
        out_path = OUTPUT_DIR / out_path

    # 1. Load INE sections
    secc = load_seccionado_gdf()

    # 2. Get OSM graph and nodes
    logging.info(f"Downloading OSM graph for '{place}' (network_type='{network_type}')")
    G = ox.graph_from_place(place, network_type=network_type)
    nodes, _ = ox.graph_to_gdfs(G)
    nodes = nodes.reset_index()  # bring osmid from index to column
    logging.info(f"OSM nodes: {len(nodes)}")

    # 3. Reproject nodes to UTM 30N
    nodes_utm = nodes.to_crs(TARGET_CRS)

    # 4. Filter sections
    if filter_by_codes:
        secc_region = filter_sections_by_codes(secc, filter_by_codes)
        logging.info(f"Sections after code filter: {len(secc_region)}")
        if len(secc_region) == 0:
            raise RuntimeError("Code filter removed all sections")
        secc_region = clip_sections_to_nodes_bbox(secc_region, nodes_utm)
        logging.info(f"Sections after bbox clip of filtered set: {len(secc_region)}")
    else:
        secc_region = clip_sections_to_nodes_bbox(secc, nodes_utm)
        logging.info(f"Sections after bbox clip: {len(secc_region)}")

    # 5. Spatial join: OSM node -> census section
    cols_for_join = [c for c in SECC_ATTRIBUTE_COLS if c in secc_region.columns]
    cols_for_join.append("geometry")
    secc_region_small = secc_region[cols_for_join]

    logging.info("Running spatial join (point in polygon)")
    nodes_with_sec = gpd.sjoin(
        nodes_utm,
        secc_region_small,
        how="left",
        predicate="within",
    )

    # 6. Coordinates for clustering
    nodes_with_sec["x_utm"] = nodes_with_sec.geometry.x
    nodes_with_sec["y_utm"] = nodes_with_sec.geometry.y

    # Lon / lat in WGS84
    nodes_ll = nodes_with_sec.to_crs("EPSG:4326")
    nodes_with_sec["lon"] = nodes_ll.geometry.x
    nodes_with_sec["lat"] = nodes_ll.geometry.y

    # 7. Select final columns
    final_cols = ["osmid", "x_utm", "y_utm", "lon", "lat"]
    final_cols.extend([c for c in SECC_ATTRIBUTE_COLS if c in nodes_with_sec.columns])

    dataset = nodes_with_sec[final_cols].copy()

    if drop_unmatched and "CUSEC" in dataset.columns:
        before = len(dataset)
        dataset = dataset[dataset["CUSEC"].notna()]
        after = len(dataset)
        logging.info(f"Dropped {before - after} nodes outside any census section. Final: {after}")

    # 8. Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out_path, index=False)
    logging.info(f"Saved dataset to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example: Madrid municipality
    # Adjust codes if needed after inspecting the seccionado file.

    build_dataset_for_place(
        place="Madrid, Spain",
        outfile="madrid_osm_drive_nodes.parquet",
        network_type="drive",
        filter_by_codes={
            "CPRO": "28",   # province Madrid
            "CMUN": "079",  # municipality Madrid
        },
    )

    # build_dataset_for_place(
    #     place="Ajalvir, Madrid, Spain",
    #     outfile="madrid_osm_drive_nodes.parquet",
    #     network_type="drive",
    #     filter_by_codes={
    #         "CPRO": "28",
    #         "CMUN": "002",
    #     },
    # )

    # build_dataset_for_place(
    #     place="Becerril de la Sierra, Madrid, Spain",
    #     outfile="becerril_de_la_sierra_osm_drive_nodes.parquet",
    #     network_type="drive",
    #     filter_by_codes={
    #         "CPRO": "28",
    #         "CMUN": "018",
    #     },
    # )

    # build_dataset_for_place(
    #     place="Aranjuez, Madrid, Spain",
    #     outfile="aranjuez_osm_drive_nodes.parquet",
    #     network_type="drive",
    #     filter_by_codes={
    #         "CPRO": "28",
    #         "CMUN": "013",
    #     },
    # )

    # build_dataset_for_place(
    #     place="Gijón, Asturias, Spain",
    #     outfile="gijon_osm_drive_nodes.parquet",
    #     network_type="drive",
    #     filter_by_codes={
    #         "CPRO": "33",
    #         "CMUN": "024",
    #     },
    # )
