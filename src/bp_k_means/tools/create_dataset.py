"""Download, transform, and persist network clustering datasets."""

import math
import zipfile
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import requests

from bp_k_means.utils.logging import logger

INE_ZIP_URL = "https://www.ine.es/prodyser/cartografia/seccionado_2025.zip"
INE_POPULATION_CSV_URL = "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/69213.csv"
DATA_DIR = Path("data/downloads")
TARGET_CRS = "EPSG:25830"  # ETRS89 / UTM 30N
OUTPUT_DIR = Path("data/datasets")
MAX_EDGE_LENGTH_M = 100
# Columns typically present in SECC_CE_20250101.shp
SECC_ATTRIBUTE_COLS = [
    "CUSEC",  # census section id
    "CUMUN",  # municipality code (province + municipality, 5 digits)
    "CSEC",  # section code within municipality
    "CDIS",  # district code
    "CMUN",  # municipality code within province (3 digits)
    "CPRO",  # province code (2 digits)
    "CCA",  # autonomous community code
    "CUDIS",  # district id
    "CLAU2",  # internal key
    "NPRO",  # province name
    "NCA",  # autonomous community name
    "CNUT0",  # NUTS 0 code
    "CNUT1",  # NUTS 1 code
    "CNUT2",  # NUTS 2 code
    "CNUT3",  # NUTS 3 code
    "NMUN",  # municipality name
]

# From smallest to largest region
LEVEL_PRIORITY = ["CUSEC", "CUMUN", "CMUN", "CPRO", "CCA"]


def ensure_dirs() -> None:
    """Create the local directories used for downloaded and generated data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_seccionado_zip() -> Path:
    """Download seccionado_2025.zip from INE if it is not already present."""
    ensure_dirs()
    zip_path = DATA_DIR / "seccionado_2025.zip"

    if zip_path.exists():
        return zip_path

    logger.info("Downloading INE seccionado 2025 from %s", INE_ZIP_URL)
    resp = requests.get(INE_ZIP_URL, stream=True, timeout=60.0)
    resp.raise_for_status()
    with zip_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    logger.info("Saved %s", zip_path)
    return zip_path


def extract_seccionado_zip(zip_path: Path) -> Path:
    """Extract seccionado_2025.zip into data directory if not already extracted."""
    extract_dir = DATA_DIR / "extracted"
    if extract_dir.exists():
        return extract_dir

    logger.info("Extracting %s", zip_path)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    logger.info("Extracted to %s", extract_dir)
    return extract_dir


def find_seccionado_shapefile(extract_dir: Path) -> Path:
    """Find the SECC_CE_20250101.shp shapefile inside the extracted tree."""
    for shp in extract_dir.rglob("*.shp"):
        if "SECC_CE_20250101" in shp.name:
            return shp
    msg = f"No SECC_CE_20250101.shp found under {extract_dir}"
    raise FileNotFoundError(msg)


def load_seccionado_gdf() -> gpd.GeoDataFrame:
    """
    Ensure seccionado_2025 is downloaded and extracted.

    The extracted SECC_CE_20250101 shapefile is loaded into a GeoDataFrame in TARGET_CRS.
    """
    zip_path = download_seccionado_zip()
    extract_dir = extract_seccionado_zip(zip_path)
    shp_path = find_seccionado_shapefile(extract_dir)

    logger.info("Reading sections from %s", shp_path)
    secc = gpd.read_file(shp_path)

    if secc.crs is None:
        secc = secc.set_crs(TARGET_CRS)
    elif secc.crs.to_string() != TARGET_CRS:
        secc = secc.to_crs(TARGET_CRS)

    logger.info("Loaded %s census sections", len(secc))
    return secc


def download_population_csv() -> Path:
    """Download INE population CSV (69213.csv) if it is not already present."""
    ensure_dirs()
    csv_path = DATA_DIR / "69213.csv"

    if csv_path.exists():
        return csv_path

    logger.info("Downloading INE population data from %s", INE_POPULATION_CSV_URL)
    resp = requests.get(INE_POPULATION_CSV_URL, stream=True, timeout=60.0)
    resp.raise_for_status()
    with csv_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    logger.info("Saved %s", csv_path)
    return csv_path


def normalize_filter_values(allowed: object) -> set[str]:
    """Normalise filter values to a set of strings."""
    if isinstance(allowed, (list, tuple, set)):
        return {str(v) for v in allowed}
    return {str(allowed)}


def filter_sections_by_codes(
    secc: gpd.GeoDataFrame,
    filter_by_codes: dict | None,
) -> gpd.GeoDataFrame:
    """
    Filter sections using all given codes at once.

    Filters are applied from most to least restrictive according to LEVEL_PRIORITY.

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

    mask = pd.Series(data=True, index=secc.index)

    for col in cols_sorted:
        if col not in secc.columns:
            msg = f"Column {col} is not in sections file"
            raise KeyError(msg)
        allowed_str = normalize_filter_values(filter_by_codes[col])
        mask &= secc[col].astype(str).isin(list(allowed_str))

    return cast("gpd.GeoDataFrame", secc[mask].copy())


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
    return secc.cx[minx:maxx, miny:maxy].copy()


def _split_long_edges(
    nodes_utm: gpd.GeoDataFrame,
    edges_utm: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, list[dict[str, object]]]:
    """Split long network edges and return updated nodes, edges, and new edge records."""
    new_edges: list[dict[str, object]] = []
    new_nodes = []
    edges_to_remove = []
    next_osmid = nodes_utm["osmid"].max() + 1

    for idx, edge in edges_utm.iterrows():
        edge_length = edge.geometry.length
        if edge_length <= MAX_EDGE_LENGTH_M:
            continue

        num_segments = math.ceil(edge_length / MAX_EDGE_LENGTH_M)
        segment_length = edge_length / num_segments
        intermediate_nodes = [edge["u"]]
        for i in range(1, num_segments):
            point = edge.geometry.interpolate((i / num_segments) * edge_length)
            new_nodes.append({"osmid": next_osmid, "geometry": point})
            intermediate_nodes.append(next_osmid)
            next_osmid += 1

        intermediate_nodes.append(edge["v"])
        new_edges.extend(
            [
                {
                    "u": intermediate_nodes[i],
                    "v": intermediate_nodes[i + 1],
                    "distance": segment_length,
                }
                for i in range(len(intermediate_nodes) - 1)
            ]
        )
        edges_to_remove.append(idx)

    if not new_nodes:
        return nodes_utm, edges_utm, new_edges

    new_nodes_gdf = gpd.GeoDataFrame(new_nodes, crs=TARGET_CRS)
    logger.info("Created %s new nodes from splitting edges", len(new_nodes_gdf))
    nodes_utm = gpd.GeoDataFrame(
        pd.concat([nodes_utm, new_nodes_gdf], ignore_index=True), crs=TARGET_CRS
    )
    logger.info("Total nodes after splitting: %s", len(nodes_utm))
    edges_utm = cast("gpd.GeoDataFrame", edges_utm.drop(edges_to_remove))
    logger.info("Removed %s split edges", len(edges_to_remove))
    logger.info("Created %s new edge segments", len(new_edges))
    return nodes_utm, edges_utm, new_edges


def _prepare_sections(
    secc: gpd.GeoDataFrame,
    nodes_utm: gpd.GeoDataFrame,
    filter_by_codes: dict | None,
) -> gpd.GeoDataFrame:
    """Filter and spatially clip census sections for a network's nodes."""
    if filter_by_codes:
        secc_region = filter_sections_by_codes(secc, filter_by_codes)
        logger.info("Sections after code filter: %s", len(secc_region))
        if secc_region.empty:
            msg = "Code filter removed all sections"
            raise RuntimeError(msg)
        secc_region = clip_sections_to_nodes_bbox(secc_region, nodes_utm)
        logger.info("Sections after bbox clip of filtered set: %s", len(secc_region))
        return secc_region

    secc_region = clip_sections_to_nodes_bbox(secc, nodes_utm)
    logger.info("Sections after bbox clip: %s", len(secc_region))
    return secc_region


def _build_adjacency_dataset(
    edges_utm: gpd.GeoDataFrame,
    new_edges: list[dict[str, object]],
) -> pd.DataFrame:
    """Build an edge table with a distance for every usable network edge."""
    adjacency_records = []
    for _idx, edge in edges_utm.iterrows():
        if "u" not in edge or "v" not in edge:
            continue
        if "distance" not in edge or bool(pd.isna(edge["distance"])):
            distance = edge.geometry.length if "geometry" in edge else 0
        else:
            distance = edge["distance"]
        adjacency_records.append({"u": edge["u"], "v": edge["v"], "distance": distance})

    adjacency_records.extend(new_edges)
    return pd.DataFrame(adjacency_records)


def build_dataset_for_place(
    place: str,
    outfile: str,
    network_type: str = "drive",
    filter_by_codes: dict | None = None,
    *,
    drop_unmatched: bool = True,
    split_long_edges: bool = False,
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
    filter_by_codes : dict | None
        Dict like {"CPRO": "28", "CMUN": "079"} to restrict sections before join.
        If None, sections are just bbox clipped around the OSM nodes.
    drop_unmatched : bool
        If True, drop nodes that do not fall inside any census section.
    split_long_edges : bool
        If True, split edges longer than MAX_EDGE_LENGTH_M into shorter segments.
    """
    ensure_dirs()
    out_path = Path(Path(outfile).stem + "_split.parquet") if split_long_edges else Path(outfile)
    if not out_path.is_absolute():
        out_path = OUTPUT_DIR / out_path

    # 1. Load INE sections
    secc = load_seccionado_gdf()

    # 2. Get OSM graph and nodes
    logger.info("Downloading OSM graph for '%s' (network_type='%s')", place, network_type)
    G = ox.graph_from_place(place, network_type=network_type)
    nodes, edges = cast("tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]", ox.graph_to_gdfs(G))
    nodes = cast("gpd.GeoDataFrame", nodes.reset_index())  # bring osmid from index to column
    logger.info("OSM nodes: %s", len(nodes))

    # 3. Reproject nodes and edges to UTM 30N
    nodes_utm = cast("gpd.GeoDataFrame", nodes.to_crs(TARGET_CRS))
    edges_utm = cast("gpd.GeoDataFrame", edges.to_crs(TARGET_CRS))
    # Bring u and v from the index to columns.
    edges_utm = cast("gpd.GeoDataFrame", edges_utm.reset_index())

    # 3a. Split long edges if requested
    if split_long_edges:
        logger.info("Splitting edges longer than %sm", MAX_EDGE_LENGTH_M)
        nodes_utm, edges_utm, new_edges = _split_long_edges(nodes_utm, edges_utm)
    else:
        new_edges = []

    # 4. Filter sections
    secc_region = _prepare_sections(secc, nodes_utm, filter_by_codes)

    # 5. Spatial join: OSM node -> census section
    cols_for_join = [c for c in SECC_ATTRIBUTE_COLS if c in secc_region.columns]
    cols_for_join.append("geometry")
    secc_region_small = cast("gpd.GeoDataFrame", secc_region[cols_for_join])

    logger.info("Running spatial join (point in polygon)")
    nodes_with_sec = cast(
        "gpd.GeoDataFrame",
        gpd.sjoin(
            nodes_utm,
            secc_region_small,
            how="left",
            predicate="within",
        ),
    )

    # Deduplicate: a node on a shared boundary can match multiple sections
    before_dedup = len(nodes_with_sec)
    nodes_with_sec = nodes_with_sec[~nodes_with_sec["osmid"].duplicated(keep="first")].copy()
    after_dedup = len(nodes_with_sec)
    if before_dedup != after_dedup:
        logger.info("Removed %s duplicate nodes after spatial join", before_dedup - after_dedup)

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

    dataset = cast("pd.DataFrame", nodes_with_sec[final_cols].copy())

    if drop_unmatched and "CUSEC" in dataset.columns:
        before = len(dataset)
        dataset = dataset[dataset["CUSEC"].notna()]
        after = len(dataset)
        logger.info("Dropped %s nodes outside any census section. Final: %s", before - after, after)

    # 8. Build adjacency dataset
    adjacency_df = _build_adjacency_dataset(edges_utm, new_edges)
    logger.info("Built adjacency dataset with %s edges", len(adjacency_df))

    # 9. Save nodes dataset
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out_path, index=False)
    logger.info("Saved nodes dataset to %s", out_path)

    # 10. Save adjacency dataset
    adjacency_path = out_path.parent / f"{out_path.stem}_edges{out_path.suffix}"
    adjacency_df.to_parquet(adjacency_path, index=False)
    logger.info("Saved adjacency dataset to %s", adjacency_path)


def _cusec_matches_filters(cusec: str, filter_by_codes: dict) -> bool:
    """Return whether a CUSEC code satisfies the requested geographic filters."""
    code_values = {
        "CUSEC": cusec,
        "CPRO": cusec[:2],
        "CMUN": cusec[2:5],
        "CUMUN": cusec[:5],
    }
    return all(
        col not in code_values or code_values[col] in normalize_filter_values(value)
        for col, value in filter_by_codes.items()
    )


def _load_population_counts(
    population_file: Path,
    filter_by_codes: dict | None,
) -> dict[str, float]:
    """Load and aggregate the latest available population count per CUSEC."""
    logger.info("Loading population data from %s", population_file)
    try:
        pop_df: Any = pd.read_csv(population_file, encoding="utf-8", sep=";", dtype=str)
    except UnicodeDecodeError:
        pop_df = pd.read_csv(population_file, encoding="ISO-8859-1", sep=";", dtype=str)

    logger.info("Loaded %s rows from population file", len(pop_df))
    if "Periodo" in pop_df.columns:
        latest_period = pop_df["Periodo"].max()
        logger.info("Filtering for latest period: %s", latest_period)
        pop_df = pop_df[pop_df["Periodo"] == latest_period].copy()
    if "Sexo" in pop_df.columns and "Total" in pop_df["Sexo"].unique():
        logger.info("Filtering for Sexo='Total'")
        pop_df = pop_df[pop_df["Sexo"] == "Total"].copy()
    if "Edad" in pop_df.columns and "Todas las edades" in pop_df["Edad"].unique():
        logger.info("Filtering for Edad='Todas las edades'")
        pop_df = pop_df[pop_df["Edad"] == "Todas las edades"].copy()
    if "Secciones" not in pop_df.columns:
        msg = "Population file must have a 'Secciones' column"
        raise KeyError(msg)

    pop_df["CUSEC"] = pop_df["Secciones"].str.extract(r"(\d{10})")[0]
    pop_df = pop_df[pop_df["CUSEC"].notna()].copy()
    logger.info("Extracted %s valid CUSEC codes", len(pop_df))

    value_cols: list[str] = [
        str(col) for col in pop_df.columns if col not in ["Secciones", "CUSEC"]
    ]
    if "Total" in pop_df.columns:
        pop_col = "Total"
    elif value_cols:
        pop_col = value_cols[-1]
    else:
        msg = "Cannot determine population column in the CSV"
        raise ValueError(msg)

    pop_df[pop_col] = pd.to_numeric(
        pop_df[pop_col].astype(str).str.replace(".", "").str.replace(",", "."),
        errors="coerce",
    )
    pop_df = pop_df[pop_df[pop_col].notna()].copy()
    cusec_population = {
        str(cusec): float(population)
        for cusec, population in pop_df.groupby("CUSEC")[pop_col].sum().items()
    }
    logger.info("Total CUSECs with population: %s", len(cusec_population))

    if filter_by_codes:
        cusec_population = {
            cusec: population
            for cusec, population in cusec_population.items()
            if _cusec_matches_filters(cusec, filter_by_codes)
        }
        logger.info("After filtering: %s CUSECs", len(cusec_population))
    return cusec_population


def _generate_population_points(
    nodes_with_cusec: gpd.GeoDataFrame,
    cusec_population: dict[str, float],
    gaussian_std_m: float,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], int]:
    """Generate Gaussian-distributed person coordinates around network nodes."""
    all_persons: list[dict[str, object]] = []
    total_population = 0
    for cusec, cusec_nodes in cast("Any", nodes_with_cusec.groupby("CUSEC")):
        if cusec not in cusec_population:
            continue
        population = int(cusec_population[cusec])
        if population <= 0:
            continue

        total_population += population
        num_nodes = len(cusec_nodes)
        persons_per_node, remainder = divmod(population, num_nodes)
        for idx, (_node_idx, node) in enumerate(cusec_nodes.iterrows()):
            num_persons = persons_per_node + (idx < remainder)
            if num_persons == 0:
                continue
            x_offsets = rng.normal(0, gaussian_std_m, num_persons)
            y_offsets = rng.normal(0, gaussian_std_m, num_persons)
            all_persons.extend(
                [
                    {
                        "x_utm": node.geometry.x + x_offsets[i],
                        "y_utm": node.geometry.y + y_offsets[i],
                        "CUSEC": cusec,
                        "node_osmid": node["osmid"],
                    }
                    for i in range(num_persons)
                ]
            )
    return all_persons, total_population


def build_population_dataset(
    place: str,
    population_file: str | Path,
    outfile: str,
    network_type: str = "drive",
    filter_by_codes: dict | None = None,
    gaussian_std_m: float = 100.0,
    random_seed: int | None = None,
) -> None:
    """
    Create a population-based dataset with Gaussian-distributed coordinates.

    Parameters
    ----------
    place : str
        Place name understood by osmnx, for example "Madrid, Spain".
    population_file : str | Path
        Path to the INE population CSV file (or compatible format).
    outfile : str
        Output Parquet file name or path.
    network_type : str
        osmnx network_type ("drive", "walk", "all", etc.).
    filter_by_codes : dict | None
        Dict like {"CPRO": "28", "CMUN": "079"} to restrict sections before join.
        If None, all sections in the population file are used.
    gaussian_std_m : float
        Standard deviation in meters for Gaussian distribution of person coordinates.
    random_seed : int | None
        Random seed for reproducibility. If None, no seed is set.
    """
    ensure_dirs()
    out_path = Path(outfile)
    if not out_path.is_absolute():
        out_path = OUTPUT_DIR / out_path

    rng = np.random.default_rng(random_seed)
    population_file = Path(population_file)
    cusec_population = _load_population_counts(population_file, filter_by_codes)

    # 5. Get OSM graph and nodes
    logger.info("Downloading OSM graph for '%s' (network_type='%s')", place, network_type)
    G = ox.graph_from_place(place, network_type=network_type)
    nodes, _ = cast("tuple[gpd.GeoDataFrame, Any]", ox.graph_to_gdfs(G))
    nodes = cast("gpd.GeoDataFrame", nodes.reset_index())
    logger.info("OSM nodes: %s", len(nodes))

    # 6. Reproject nodes to UTM 30N
    nodes_utm = nodes.to_crs(TARGET_CRS)

    # 7. Load sections and join nodes to CUSECs
    secc = load_seccionado_gdf()

    # Filter sections to those in our population data
    secc = secc[secc["CUSEC"].isin(list(cusec_population))].copy()
    logger.info("Sections matching population data: %s", len(secc))

    # Spatial join to assign CUSEC to each node
    logger.info("Running spatial join (point in polygon)")
    secc_small = cast("gpd.GeoDataFrame", secc[["CUSEC", "geometry"]])
    nodes_with_cusec = cast(
        "gpd.GeoDataFrame",
        gpd.sjoin(nodes_utm, secc_small, how="inner", predicate="within"),
    )
    logger.info("Nodes within census sections: %s", len(nodes_with_cusec))

    # 8. Generate population points
    all_persons, total_population = _generate_population_points(
        nodes_with_cusec,
        cusec_population,
        gaussian_std_m,
        rng,
    )

    logger.info(
        "Generated %s person coordinates from %s total population",
        len(all_persons),
        total_population,
    )

    # 10. Create GeoDataFrame and add lat/lon
    persons_df = pd.DataFrame(all_persons)
    if persons_df.empty:
        logger.warning("No persons were generated. Check if 'place' and 'filter_by_codes' overlap.")
        return

    persons_gdf = gpd.GeoDataFrame(
        persons_df,
        geometry=gpd.points_from_xy(persons_df["x_utm"], persons_df["y_utm"]),
        crs=TARGET_CRS,
    )

    # Add WGS84 coordinates for person locations
    persons_ll = persons_gdf.to_crs("EPSG:4326")
    persons_gdf["lon"] = persons_ll.geometry.x
    persons_gdf["lat"] = persons_ll.geometry.y

    # 11. Find nearest node to each person and store its coordinates
    nodes_nearest = cast("gpd.GeoDataFrame", nodes_utm[["osmid", "geometry"]])
    nodes_nearest = cast(
        "gpd.GeoDataFrame",
        nodes_nearest.rename(columns={"osmid": "nearest_node_osmid"}),
    )
    persons_nearest = cast(
        "gpd.GeoDataFrame",
        gpd.sjoin_nearest(
            persons_gdf,
            nodes_nearest,
            how="left",
            distance_col="nearest_node_dist",
        ),
    )
    node_geom: Any = nodes_utm.set_index("osmid")["geometry"]
    nearest_geom = gpd.GeoSeries(
        persons_nearest["nearest_node_osmid"].map(node_geom),
        crs=TARGET_CRS,
    )
    persons_nearest["nearest_node_x_utm"] = nearest_geom.x
    persons_nearest["nearest_node_y_utm"] = nearest_geom.y
    nearest_geom_ll = nearest_geom.to_crs("EPSG:4326")
    persons_nearest["nearest_node_lon"] = nearest_geom_ll.x
    persons_nearest["nearest_node_lat"] = nearest_geom_ll.y

    # 12. Select final columns
    final_df = persons_nearest[
        [
            "x_utm",
            "y_utm",
            "lon",
            "lat",
            "CUSEC",
            "nearest_node_osmid",
            "nearest_node_x_utm",
            "nearest_node_y_utm",
            "nearest_node_lon",
            "nearest_node_lat",
            "nearest_node_dist",
        ]
    ].copy()

    # 13. Save dataset
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(out_path, index=False)
    logger.info("Saved population dataset to %s (%s persons)", out_path, len(final_df))

    # 14. Save graph
    graph_path = out_path.parent / f"{out_path.stem}_graph.graphml"
    ox.save_graphml(G, filepath=graph_path)
    logger.info("Saved graph to %s", graph_path)

    # 15. Save distance matrix for graph nodes
    node_list = list(G.nodes)
    node_index = {node_id: idx for idx, node_id in enumerate(node_list)}
    dist_matrix = np.full((len(node_list), len(node_list)), np.inf, dtype=float)
    for source, lengths in nx.all_pairs_dijkstra_path_length(G, weight="length"):
        source_idx = node_index[source]
        for target, distance in lengths.items():
            dist_matrix[source_idx, node_index[target]] = distance

    matrix_path = out_path.parent / f"{out_path.stem}_dist_matrix.npy"
    np.save(matrix_path, dist_matrix)
    nodes_path = out_path.parent / f"{out_path.stem}_dist_matrix_nodes.csv"
    pd.DataFrame({"node_id": node_list}).to_csv(nodes_path, index=False)
    logger.info("Saved distance matrix to %s and node order to %s", matrix_path, nodes_path)


if __name__ == "__main__":
    # Example: Madrid municipality
    # Adjust codes if needed after inspecting the seccionado file.

    build_dataset_for_place(
        place="Community of Madrid, Spain",
        outfile="com_madrid_osm_drive_nodes_split.parquet",
        network_type="drive",
        filter_by_codes={
            "CPRO": "28",
        },
        split_long_edges=True,
    )

    build_dataset_for_place(
        place="Castile and León, Spain",
        outfile="castile_and_leon_osm_drive_nodes.parquet",
        network_type="drive",
        filter_by_codes={
            "CCA": "07",
        },
    )

    # Very small size
    # build_dataset_for_place(
    #     place="Ajalvir, Madrid, Spain",
    #     outfile="ajalvir_osm_drive_nodes.parquet",
    #     network_type="drive",
    #     filter_by_codes={
    #         "CPRO": "28",
    #         "CMUN": "002",
    #     },
    # )
    # build_dataset_for_place(
    #     place="Torrelaguna, Madrid, Spain",
    #     outfile="torrelaguna_osm_drive_nodes.parquet",
    #     network_type="drive",
    #     filter_by_codes={
    #         "CPRO": "28",
    #         "CMUN": "151",
    #     },
    # )
    # build_dataset_for_place(
    #     place="Colmenar de Oreja, Madrid, Spain",
    #     outfile="colmenar_de_oreja_osm_drive_nodes.parquet",
    #     network_type="drive",
    #     filter_by_codes={
    #         "CPRO": "28",
    #         "CMUN": "043",
    #     },
    # )

    # # Small size
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
    #     place="Cercedilla, Madrid, Spain",
    #     outfile="cercedilla_osm_drive_nodes.parquet",
    #     network_type="drive",
    #     filter_by_codes={
    #         "CPRO": "28",
    #         "CMUN": "038",
    #     },
    # )
    # build_dataset_for_place(
    #     place="Manzanares el Real, Madrid, Spain",
    #     outfile="manzanares_el_real_osm_drive_nodes.parquet",
    #     network_type="drive",
    #     filter_by_codes={
    #         "CPRO": "28",
    #         "CMUN": "082",
    #     },
    # )

    # # Medium size
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
    #     place="Segovia, Castile and León, Spain",
    #     outfile="segovia_osm_drive_nodes.parquet",
    #     network_type="drive",
    #     filter_by_codes={
    #         "CPRO": "40",
    #         "CMUN": "194",
    #     },
    # )
    # build_dataset_for_place(
    #     place="Cáceres, Extremadura, Spain",
    #     outfile="caceres_osm_drive_nodes.parquet",
    #     network_type="drive",
    #     filter_by_codes={
    #         "CPRO": "10",
    #         "CMUN": "037",
    #     },
    # )

    # # Large size
    # build_dataset_for_place(
    #     place="Gijón, Asturias, Spain",
    #     outfile="gijon_osm_drive_nodes.parquet",
    #     network_type="drive",
    #     filter_by_codes={
    #         "CPRO": "33",
    #         "CMUN": "024",
    #     },
    # )
    # build_dataset_for_place(
    #     place="Valladolid, Castile and León, Spain",
    #     outfile="valladolid_osm_drive_nodes.parquet",
    #     network_type="drive",
    #     filter_by_codes={
    #         "CPRO": "47",
    #         "CMUN": "186",
    #     },
    # )
    # build_dataset_for_place(
    #     place="Salamanca, Castile and León, Spain",
    #     outfile="salamanca_osm_drive_nodes.parquet",
    #     network_type="drive",
    #     filter_by_codes={
    #         "CPRO": "37",
    #         "CMUN": "274",
    #     },
    # )

    # # Very large size
    # build_dataset_for_place(
    #     place="Madrid, Community of Madrid, Spain",
    #     outfile="madrid_osm_drive_nodes.parquet",
    #     network_type="drive",
    #     filter_by_codes={
    #         "CPRO": "28",
    #         "CMUN": "079",
    #     },
    # )
    # build_dataset_for_place(
    #     place="Valencia, Valencian Community, Spain",
    #     outfile="valencia_osm_drive_nodes.parquet",
    #     network_type="drive",
    #     filter_by_codes={
    #         "CPRO": "46",
    #         "CMUN": "250",
    #     },
    # )
    # build_dataset_for_place(
    #     place="Barcelona, Catalonia, Spain",
    #     outfile="barcelona_osm_drive_nodes.parquet",
    #     network_type="drive",
    #     filter_by_codes={
    #         "CPRO": "08",
    #         "CMUN": "019",
    #     },
    # )
