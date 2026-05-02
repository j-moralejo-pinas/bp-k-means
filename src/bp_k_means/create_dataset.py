import logging
import math
import zipfile
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import requests

INE_ZIP_URL = "https://www.ine.es/prodyser/cartografia/seccionado_2025.zip"
INE_POPULATION_CSV_URL = "https://www.ine.es/jaxiT3/files/t/es/csv_bdsc/69213.csv"
DATA_DIR = Path("data/downloads")
TARGET_CRS = "EPSG:25830"  # ETRS89 / UTM 30N
OUTPUT_DIR = Path("data/datasets")

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


def download_population_csv() -> Path:
    """
    Download INE population CSV (69213.csv) if it is not already present.
    """
    ensure_dirs()
    csv_path = DATA_DIR / "69213.csv"

    if csv_path.exists():
        return csv_path

    logging.info(f"Downloading INE population data from {INE_POPULATION_CSV_URL}")
    resp = requests.get(INE_POPULATION_CSV_URL, stream=True)
    resp.raise_for_status()
    with open(csv_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    logging.info(f"Saved {csv_path}")
    return csv_path


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
    filter_by_codes : dict or None
        Dict like {"CPRO": "28", "CMUN": "079"} to restrict sections before join.
        If None, sections are just bbox clipped around the OSM nodes.
    drop_unmatched : bool
        If True, drop nodes that do not fall inside any census section.
    split_long_edges : bool
        If True, split edges longer than 100m into segments shorter than 100m.
    """
    ensure_dirs()
    out_path = Path(Path(outfile).stem + "_split.parquet") if split_long_edges else Path(outfile)
    if not out_path.is_absolute():
        out_path = OUTPUT_DIR / out_path

    # 1. Load INE sections
    secc = load_seccionado_gdf()

    # 2. Get OSM graph and nodes
    logging.info(f"Downloading OSM graph for '{place}' (network_type='{network_type}')")
    G = ox.graph_from_place(place, network_type=network_type)
    nodes, edges = ox.graph_to_gdfs(G)
    nodes = nodes.reset_index()  # bring osmid from index to column
    logging.info(f"OSM nodes: {len(nodes)}")

    # 3. Reproject nodes and edges to UTM 30N
    nodes_utm = nodes.to_crs(TARGET_CRS)
    edges_utm = edges.to_crs(TARGET_CRS)
    edges_utm = edges_utm.reset_index()  # bring u, v from index to columns

    # 3a. Split long edges if requested
    new_edges = []  # Initialize outside to ensure it's always defined
    if split_long_edges:
        logging.info("Splitting edges longer than 100m")
        new_nodes = []
        edges_to_remove = []
        next_osmid = nodes_utm["osmid"].max() + 1

        for idx, edge in edges_utm.iterrows():
            edge_length = edge.geometry.length
            if edge_length > 100:
                # Calculate minimum number of segments needed
                num_segments = math.ceil(edge_length / 100)
                segment_length = edge_length / num_segments

                # Track intermediate node IDs for this edge
                intermediate_nodes = [edge["u"]]

                # Create intermediate points along the edge
                for i in range(1, num_segments):
                    # Distance along the line for this point
                    distance = (i / num_segments) * edge_length
                    # Interpolate point at this distance
                    point = edge.geometry.interpolate(distance)
                    new_nodes.append(
                        {
                            "osmid": next_osmid,
                            "geometry": point,
                        }
                    )
                    intermediate_nodes.append(next_osmid)
                    next_osmid += 1

                intermediate_nodes.append(edge["v"])

                # Create new edges between consecutive nodes
                for i in range(len(intermediate_nodes) - 1):
                    new_edges.append(
                        {
                            "u": intermediate_nodes[i],
                            "v": intermediate_nodes[i + 1],
                            "distance": segment_length,
                        }
                    )

                # Mark original edge for removal
                edges_to_remove.append(idx)

        if new_nodes:
            new_nodes_gdf = gpd.GeoDataFrame(new_nodes, crs=TARGET_CRS)
            logging.info(f"Created {len(new_nodes_gdf)} new nodes from splitting edges")
            nodes_utm = pd.concat([nodes_utm, new_nodes_gdf], ignore_index=True)
            logging.info(f"Total nodes after splitting: {len(nodes_utm)}")

            # Remove split edges and add new segmented edges
            edges_utm = edges_utm.drop(edges_to_remove)
            logging.info(f"Removed {len(edges_to_remove)} split edges")

            if new_edges:
                new_edges_df = pd.DataFrame(new_edges)
                logging.info(f"Created {len(new_edges_df)} new edge segments")

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

    # Deduplicate: a node on a shared boundary can match multiple sections
    before_dedup = len(nodes_with_sec)
    nodes_with_sec = nodes_with_sec[~nodes_with_sec["osmid"].duplicated(keep="first")].copy()
    after_dedup = len(nodes_with_sec)
    if before_dedup != after_dedup:
        logging.info(f"Removed {before_dedup - after_dedup} duplicate nodes after spatial join")

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

    # 8. Build adjacency dataset
    adjacency_records = []
    for idx, edge in edges_utm.iterrows():
        if "u" in edge and "v" in edge:
            # Calculate distance if not already present
            if "distance" not in edge or pd.isna(edge["distance"]):
                distance = edge.geometry.length if "geometry" in edge else 0
            else:
                distance = edge["distance"]
            adjacency_records.append(
                {
                    "u": edge["u"],
                    "v": edge["v"],
                    "distance": distance,
                }
            )

    # Add new edges from splitting if they exist
    if new_edges:
        adjacency_records.extend(new_edges)

    adjacency_df = pd.DataFrame(adjacency_records)
    logging.info(f"Built adjacency dataset with {len(adjacency_df)} edges")

    # 9. Save nodes dataset
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out_path, index=False)
    logging.info(f"Saved nodes dataset to {out_path}")

    # 10. Save adjacency dataset
    adjacency_path = out_path.parent / f"{out_path.stem}_edges{out_path.suffix}"
    adjacency_df.to_parquet(adjacency_path, index=False)
    logging.info(f"Saved adjacency dataset to {adjacency_path}")


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
    population_file : str or Path
        Path to the INE population CSV file (or compatible format).
    outfile : str
        Output Parquet file name or path.
    network_type : str
        osmnx network_type ("drive", "walk", "all", etc.).
    filter_by_codes : dict or None
        Dict like {"CPRO": "28", "CMUN": "079"} to restrict sections before join.
        If None, all sections in the population file are used.
    gaussian_std_m : float
        Standard deviation in meters for Gaussian distribution of person coordinates.
    random_seed : int or None
        Random seed for reproducibility. If None, no seed is set.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    ensure_dirs()
    out_path = Path(outfile)
    if not out_path.is_absolute():
        out_path = OUTPUT_DIR / out_path

    # 1. Load population data
    population_file = Path(population_file)
    logging.info(f"Loading population data from {population_file}")

    # Read CSV with proper encoding (INE files are often in ISO-8859-1 or similar)
    # Read as string initially to avoid mixed type inference on population columns
    try:
        pop_df = pd.read_csv(population_file, encoding="utf-8", sep=";", dtype=str)
    except UnicodeDecodeError:
        pop_df = pd.read_csv(population_file, encoding="ISO-8859-1", sep=";", dtype=str)

    logging.info(f"Loaded {len(pop_df)} rows from population file")

    # Filter distinct periods, sex and age groups to avoid duplicates
    if "Periodo" in pop_df.columns:
        latest_period = pop_df["Periodo"].max()
        logging.info(f"Filtering for latest period: {latest_period}")
        pop_df = pop_df[pop_df["Periodo"] == latest_period].copy()

    if "Sexo" in pop_df.columns and "Total" in pop_df["Sexo"].unique():
        logging.info("Filtering for Sexo='Total'")
        pop_df = pop_df[pop_df["Sexo"] == "Total"].copy()

    if "Edad" in pop_df.columns and "Todas las edades" in pop_df["Edad"].unique():
        logging.info("Filtering for Edad='Todas las edades'")
        pop_df = pop_df[pop_df["Edad"] == "Todas las edades"].copy()

    # 2. Extract CUSEC codes from "Secciones" column
    if "Secciones" not in pop_df.columns:
        raise KeyError("Population file must have a 'Secciones' column")

    pop_df["CUSEC"] = pop_df["Secciones"].str.extract(r"(\d{10})")[0]
    pop_df = pop_df[pop_df["CUSEC"].notna()].copy()
    logging.info(f"Extracted {len(pop_df)} valid CUSEC codes")

    # 3. Get total population per CUSEC
    # Assuming the last column or a column named "Total" contains population
    # Adjust this based on actual CSV structure
    value_cols = [col for col in pop_df.columns if col not in ["Secciones", "CUSEC"]]
    if "Total" in pop_df.columns:
        pop_col = "Total"
    elif len(value_cols) > 0:
        pop_col = value_cols[-1]  # Use last numeric column
    else:
        raise ValueError("Cannot determine population column in the CSV")

    # Clean and convert population values
    pop_df[pop_col] = pd.to_numeric(
        pop_df[pop_col].astype(str).str.replace(".", "").str.replace(",", "."), errors="coerce"
    )
    pop_df = pop_df[pop_df[pop_col].notna()].copy()

    # Group by CUSEC and sum population
    cusec_population = pop_df.groupby("CUSEC")[pop_col].sum().to_dict()
    logging.info(f"Total CUSECs with population: {len(cusec_population)}")

    # 4. Filter by codes if specified
    if filter_by_codes:
        # Filter CUSECs based on the codes
        filtered_cusecs = set()
        for cusec in cusec_population.keys():
            match = True
            for col, value in filter_by_codes.items():
                allowed_str = normalize_filter_values(value)
                if col == "CUSEC":
                    if cusec not in allowed_str:
                        match = False
                        break
                elif col == "CPRO":
                    if cusec[:2] not in allowed_str:
                        match = False
                        break
                elif col == "CMUN":
                    if cusec[2:5] not in allowed_str:
                        match = False
                        break
                elif col == "CUMUN":
                    if cusec[:5] not in allowed_str:
                        match = False
                        break
            if match:
                filtered_cusecs.add(cusec)

        cusec_population = {k: v for k, v in cusec_population.items() if k in filtered_cusecs}
        logging.info(f"After filtering: {len(cusec_population)} CUSECs")

    # 5. Get OSM graph and nodes
    logging.info(f"Downloading OSM graph for '{place}' (network_type='{network_type}')")
    G = ox.graph_from_place(place, network_type=network_type)
    nodes, _ = ox.graph_to_gdfs(G)
    nodes = nodes.reset_index()
    logging.info(f"OSM nodes: {len(nodes)}")

    # 6. Reproject nodes to UTM 30N
    nodes_utm = nodes.to_crs(TARGET_CRS)

    # 7. Load sections and join nodes to CUSECs
    secc = load_seccionado_gdf()

    # Filter sections to those in our population data
    secc = secc[secc["CUSEC"].isin(cusec_population.keys())].copy()
    logging.info(f"Sections matching population data: {len(secc)}")

    # Spatial join to assign CUSEC to each node
    logging.info("Running spatial join (point in polygon)")
    nodes_with_cusec = gpd.sjoin(
        nodes_utm, secc[["CUSEC", "geometry"]], how="inner", predicate="within"
    )
    logging.info(f"Nodes within census sections: {len(nodes_with_cusec)}")

    # 8. Group nodes by CUSEC
    nodes_by_cusec = nodes_with_cusec.groupby("CUSEC")

    # 9. Generate population points
    all_persons = []
    total_population = 0

    for cusec, cusec_nodes in nodes_by_cusec:
        if cusec not in cusec_population:
            continue

        population = int(cusec_population[cusec])
        if population <= 0:
            continue

        total_population += population
        num_nodes = len(cusec_nodes)

        # Distribute population equally across nodes
        persons_per_node = population // num_nodes
        remainder = population % num_nodes

        for idx, (node_idx, node) in enumerate(cusec_nodes.iterrows()):
            # Assign persons to this node
            num_persons = persons_per_node + (1 if idx < remainder else 0)

            if num_persons == 0:
                continue

            # Generate random coordinates using Gaussian distribution
            # Node coordinates in UTM
            node_x = node.geometry.x
            node_y = node.geometry.y

            # Generate random offsets
            x_offsets = np.random.normal(0, gaussian_std_m, num_persons)
            y_offsets = np.random.normal(0, gaussian_std_m, num_persons)

            # Create person records
            for i in range(num_persons):
                all_persons.append(
                    {
                        "x_utm": node_x + x_offsets[i],
                        "y_utm": node_y + y_offsets[i],
                        "CUSEC": cusec,
                        "node_osmid": node["osmid"],
                    }
                )

    logging.info(
        f"Generated {len(all_persons)} person coordinates from {total_population} total population"
    )

    # 10. Create GeoDataFrame and add lat/lon
    persons_df = pd.DataFrame(all_persons)
    if persons_df.empty:
        logging.warning(
            "No persons were generated. Check if 'place' and 'filter_by_codes' overlap."
        )
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
    nodes_nearest = nodes_utm[["osmid", "geometry"]].rename(columns={"osmid": "nearest_node_osmid"})
    persons_nearest = gpd.sjoin_nearest(
        persons_gdf,
        nodes_nearest,
        how="left",
        distance_col="nearest_node_dist",
    )
    node_geom = nodes_utm.set_index("osmid")["geometry"]
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
    logging.info(f"Saved population dataset to {out_path} ({len(final_df)} persons)")

    # 14. Save graph
    graph_path = out_path.parent / f"{out_path.stem}_graph.graphml"
    ox.save_graphml(G, filepath=graph_path)
    logging.info(f"Saved graph to {graph_path}")

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
    logging.info(f"Saved distance matrix to {matrix_path} and node order to {nodes_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
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

    # build_population_dataset(
    #     place="Madrid, Spain",
    #     population_file=download_population_csv(),
    #     outfile="madrid_population_gaussian.parquet",
    #     network_type="drive",
    #     filter_by_codes={
    #         "CPRO": "28",  # province Madrid
    #         "CMUN": "079",  # municipality Madrid
    #     },
    #     gaussian_std_m=100.0,
    #     random_seed=42,
    # )

    # build_population_dataset(
    #     place="Ajalvir, Madrid, Spain",
    #     population_file=download_population_csv(),
    #     outfile="ajalvir_population_gaussian.parquet",
    #     network_type="drive",
    #     filter_by_codes={
    #         "CPRO": "28",  # province Madrid
    #         "CMUN": "002",  # municipality Ajalvir
    #     },
    #     gaussian_std_m=100.0,
    #     random_seed=42,
    # )
