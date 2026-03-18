#!/usr/bin/env python
# coding: utf-8

"""
Berkeley FLARE (Fire Lineage and Reconstruction Engine)
-------------------------------------------------------
Core algorithm for reconstructing wildfire events from burned-area observations
using a lineage-aware, multi-object-tracking framework.

This module identifies and links spatiotemporally continuous burned-area objects
to reconstruct independent fire events and their evolution, including ignition,
propagation, merging, splitting, and extinction.

Author: Tianjiao Pu (TJ)
Current Affiliation: High Meadows Environmental Institute, Princeton University
Project: Berkeley FLARE
Last updated: March 2026

Reference
---------
Pu, T. et al. A Song of Water and Fire: An Event-Centric Framework to Track
Megafire Dynamics in the World’s Largest Wetland (under review, 2026).

Inputs
------
- Daily or time-resolved burned-area masks
- Spatially indexed fire objects
- Object attributes and adjacency / overlap relationships

Outputs
-------
- Reconstructed fire-event database
- Event-level attributes (e.g., duration, extent, peak size)
- Lineage relationships among objects and events

Notes
-----
This script is intended as the core reconstruction engine of the FLARE framework.
For an end-to-end demonstration workflow, see:
    demo/Fire Event Reconstruction Workflow.ipynb
"""


import gc
import numpy as np
import sqlite3
import zlib
from pathlib import Path
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import load_npz, csr_matrix
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import find_objects
from collections import defaultdict

# =========================================================
# 1. Object clustering with DBSCAN
# =========================================================
def treat_dbscan(binary_array, eps, min_samples):
    """
    Apply DBSCAN clustering to a binary fire mask.

    Parameters
    ----------
    binary_array : 2D numpy array
        Binary raster where burned pixels = 1.
    eps : float
        DBSCAN neighborhood radius (in pixels).
    min_samples : int
        Minimum number of pixels to form a cluster.

    Returns
    -------
    labeled_array : 2D numpy array
        0  = background
       -1  = noise pixels
        1+ = cluster labels
    """

    points = np.argwhere(binary_array == 1)

    labeled_array = np.zeros_like(binary_array, dtype=int)

    if points.shape[0] == 0:
        return labeled_array

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    for (y, x), label in zip(points, labels):
        labeled_array[y, x] = -1 if label == -1 else label + 1

    # Flip vertically to match geographic orientation
    return np.flipud(labeled_array)


def run_dbscan_year(input_npz,output_npz,npy_shape=(1792, 1671),eps=3,min_samples=3):
    """
    Run daily DBSCAN clustering on a burned-area dataset and save results as a sparse matrix.

    Parameters
    ----------
    input_npz : str
        Path to the input sparse burned-area file with shape (n_days, nrows*ncols).
    output_npz : str
        Path to save the DBSCAN-labeled sparse matrix.
    npy_shape : tuple
        Spatial raster shape (nrows, ncols).
    eps : float
        DBSCAN neighborhood radius (pixels).
    min_samples : int
        Minimum pixels required to form a cluster.

    Output
    ------
    Saves a sparse matrix with shape (n_days, nrows*ncols), where each row is a
    flattened labeled raster:
    0 = background, -1 = noise, 1+ = fire clusters.
    """
    bar_width = 30
    sparse_matrix = load_npz(input_npz)
    flattened_array_tmp = sparse_matrix.toarray()

    # Reshape flattened raster to (rows, cols, days)
    fire_occurence = flattened_array_tmp.reshape(npy_shape[0], npy_shape[1], flattened_array_tmp.shape[0])

    scaned_array = []
    total_days = flattened_array_tmp.shape[0]
    for i in range(total_days):
        labeled_array = treat_dbscan(fire_occurence[:, :, i],eps,min_samples)
        scaned_array.append(labeled_array)
        del labeled_array # Keep explicit deletion to preserve original behavior
        gc.collect()
        # Progress bar update every 25 days, and always on the last day
        if (i + 1) % 25 == 0 or (i + 1) == total_days:
            progress = (i + 1) / total_days
            filled = int(bar_width * progress)
            bar = "█" * filled + "-" * (bar_width - filled)
            print(f"[{bar}] {i+1}/{total_days} days ({progress*100:.1f}%)")
    dense_array = np.array(scaned_array)
    flattened_array = dense_array.reshape(dense_array.shape[0], -1) # Flatten spatial dimensions -> (days, pixels)
    sparse_matrix = csr_matrix(flattened_array)
    sp.save_npz(output_npz, sparse_matrix)
    print(f"Saved to {output_npz}")


# =========================================================
# 2. Object tracking with Hungarian assignment
# =========================================================
def extract_objects(frame):
    # Keep valid object labels only (exclude background=0 and noise=-1)
    unique_labels = np.unique(frame)
    unique_labels = unique_labels[(unique_labels != 0) & (unique_labels != -1)]

    # Build one binary mask per object
    objects = [(frame == label) for label in unique_labels]

    return frame, objects, unique_labels


def compute_bounding_box(obj):
    # Find rows/cols containing object pixels
    rows = np.any(obj, axis=1)
    cols = np.any(obj, axis=0)

    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return (rmin, rmax, cmin, cmax)
    else:
        return None


def bounding_boxes_overlap(box1, box2):
    # Check whether two bounding boxes overlap
    rmin1, rmax1, cmin1, cmax1 = box1
    rmin2, rmax2, cmin2, cmax2 = box2

    return not (
        rmax1 < rmin2 or
        rmax2 < rmin1 or
        cmax1 < cmin2 or
        cmax2 < cmin1
    )


def cost_matrix(objects1, objects2):
    num_objects1 = len(objects1)
    num_objects2 = len(objects2)

    # Convert masks to sparse matrices for faster overlap calculation
    objects1_sparse = [csr_matrix(obj) for obj in objects1]
    objects2_sparse = [csr_matrix(obj) for obj in objects2]

    # Precompute bounding boxes
    bounding_boxes1 = [compute_bounding_box(obj.toarray()) for obj in objects1_sparse]
    bounding_boxes2 = [compute_bounding_box(obj.toarray()) for obj in objects2_sparse]

    # Default cost = 1 (no match)
    cost = np.ones((num_objects1, num_objects2))

    for i, (obj1, box1) in enumerate(zip(objects1_sparse, bounding_boxes1)):
        for j, (obj2, box2) in enumerate(zip(objects2_sparse, bounding_boxes2)):
            if bounding_boxes_overlap(box1, box2):
                intersection = obj1.multiply(obj2).sum()
                union = obj1.sum() + obj2.sum() - intersection

                if union > 0:
                    cost[i, j] = 1 - intersection / union
                else:
                    cost[i, j] = 1
            else:
                cost[i, j] = 1

    return cost


def track_objects(frames):
    # Match objects between two consecutive frames
    tracks = []

    frame1, frame2 = frames[0], frames[1]
    _, objects1, labels1 = extract_objects(frame1)
    _, objects2, labels2 = extract_objects(frame2)

    # Initial one-to-one matching using Hungarian assignment
    cost = cost_matrix(objects1, objects2)
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_objects1 = set()
    matched_objects2 = set()

    # Keep valid matches only
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] < 1:
            tracks.append([labels1[i], labels2[j]])
            matched_objects1.add(i)
            matched_objects2.add(j)

    # Handle merges: one object in frame1 overlaps multiple labels in frame2
    for i in range(len(objects1)):
        add_label = np.unique(frame2[np.where(frame1 == labels1[i])])

        for label_index in add_label:
            if label_index > 0:
                j = np.where(labels2 == label_index)[0][0]
                tracks.append([labels1[i], labels2[j]])
                matched_objects1.add(i)
                matched_objects2.add(j)

    # Handle splits: one object in frame2 overlaps multiple labels in frame1
    for j in range(len(objects2)):
        add_label = np.unique(frame1[np.where(frame2 == labels2[j])])

        for label_index in add_label:
            if label_index > 0:
                i = np.where(labels1 == label_index)[0][0]
                tracks.append([labels1[i], labels2[j]])
                matched_objects1.add(i)
                matched_objects2.add(j)

    # Remove duplicate links
    tracks = [list(pair) for pair in set(tuple(track) for track in tracks)]

    return tracks


# =========================================================
# 3. Config
# =========================================================
UID_DAY_FACTOR = 1_000_000  # assumes daily object labels < 1,000,000


def make_uid(doy: int, obj_id: int) -> int:
    return int(doy) * UID_DAY_FACTOR + int(obj_id)


def parse_uid(uid: int):
    doy = uid // UID_DAY_FACTOR
    obj_id = uid % UID_DAY_FACTOR
    return doy, obj_id


def compress_binary_mask(mask: np.ndarray) -> bytes:
    return zlib.compress(mask.astype(np.uint8).tobytes())


def decompress_binary_mask(mask_blob: bytes, shape):
    return np.frombuffer(zlib.decompress(mask_blob), dtype=np.uint8).reshape(shape)


# =========================================================
# 4. Tree structure
# =========================================================
class TreeNode:
    def __init__(self, time, value):
        self.time = int(time)
        self.value = int(value)
        self.children = []
        self.parents = []

    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parents.append(self)


class Tree:
    def __init__(self):
        self.nodes = {}

    def add_link(self, parent_time, parent_value, child_time, child_value):
        parent_key = (int(parent_time), int(parent_value))
        child_key = (int(child_time), int(child_value))

        if parent_key not in self.nodes:
            self.nodes[parent_key] = TreeNode(*parent_key)
        if child_key not in self.nodes:
            self.nodes[child_key] = TreeNode(*child_key)

        self.nodes[parent_key].add_child(self.nodes[child_key])

    def build_graph(self):
        G = nx.DiGraph()
        for node in self.nodes.values():
            parent_uid = make_uid(node.time, node.value)
            G.add_node(parent_uid)
            for child in node.children:
                child_uid = make_uid(child.time, child.value)
                G.add_edge(parent_uid, child_uid)
        return G


# =========================================================
# 5. Loaders
# =========================================================
def build_tree_from_links(npy_path):
    """
    Expected:
        _, link_data = np.load(npy_path, allow_pickle=True)

    link_data[t] contains iterable of (parent_id, child_id)
    linking day t -> t+1.
    """
    print("Step 1: Building Tree from link data...")

    arr = np.load(npy_path, allow_pickle=True)
    _, link_data = arr

    tree = Tree()
    for t, links in enumerate(link_data):
        if links is None or len(links) == 0:
            continue
        for parent_id, child_id in links:
            tree.add_link(t, int(parent_id), t + 1, int(child_id))

    return tree, link_data


def load_daily_object_maps(npz_path, raster_shape):
    """
    Load sparse daily labeled rasters and reshape to (n_days, nrows, ncols).
    """
    raw = sp.load_npz(npz_path).toarray()
    nrows, ncols = raster_shape
    return raw.reshape((-1, nrows, ncols))


# =========================================================
# 6. Graph / lineage helpers
# =========================================================
def get_uid_to_lineage_map(tree):
    """
    Event definition:
    each lineage/event = one undirected connected component.
    """
    print("Step 2: Building connectivity graph and identifying events...")

    G = tree.build_graph().to_undirected()
    components = list(nx.connected_components(G))

    uid_to_lineage = {}
    for lineage_id, comp in enumerate(components, start=1):
        for uid in comp:
            uid_to_lineage[uid] = lineage_id

    print(f"Total independent fire events found: {len(components)}")
    return uid_to_lineage, components


def build_transition_stats(link_data):
    """
    Build parent/child counts and transition list.
    """
    parent_to_children = defaultdict(set)
    child_to_parents = defaultdict(set)
    transition_rows = []

    for t, links in enumerate(link_data):
        if links is None or len(links) == 0:
            continue

        for link in links:
            p_id = int(link[0])
            c_id = int(link[1])

            parent_uid = make_uid(t, p_id)
            child_uid = make_uid(t + 1, c_id)

            parent_to_children[parent_uid].add(child_uid)
            child_to_parents[child_uid].add(parent_uid)

            transition_rows.append((parent_uid, child_uid, t, t + 1))

    return parent_to_children, child_to_parents, transition_rows


def classify_link_type(parent_uid, child_uid, parent_to_children, child_to_parents):
    n_children = len(parent_to_children.get(parent_uid, []))
    n_parents = len(child_to_parents.get(child_uid, []))

    if n_children > 1 and n_parents > 1:
        return "merge_split_complex"
    elif n_children > 1:
        return "split"
    elif n_parents > 1:
        return "merge"
    else:
        return "continuation"


# =========================================================
# 7. Spatial metadata helper
# =========================================================
def build_spatial_metadata_from_gdalinfo(year):
    """
    Based on the GeoTIFF metadata you already extracted from gdalinfo.
    Assumes same grid as the daily maps.
    """
    return {
        "year": int(year),
        "grid_width": 1671,
        "grid_height": 1792,
        "crs_epsg": 4326,
        "crs_name": "WGS 84",
        "projection_wkt": (
            'GEOGCRS["WGS 84",'
            'DATUM["World Geodetic System 1984",'
            'ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],'
            'PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],'
            'CS[ellipsoidal,2],'
            'AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],'
            'AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],'
            'ID["EPSG",4326]]'
        ),
        "origin_x": -61.503155927243043,
        "origin_y": -13.995752126582145,
        "pixel_size_x": 0.004491576420598,
        "pixel_size_y": -0.004491576420598,
        "bounds_left": -61.503155927243043,
        "bounds_right": -53.997731715023645,
        "bounds_top": -13.995752126582145,
        "bounds_bottom": -22.04465706843416,
        # affine transform parameters
        "transform_a": 0.004491576420598,
        "transform_b": 0.0,
        "transform_c": -61.503155927243043,
        "transform_d": 0.0,
        "transform_e": -0.004491576420598,
        "transform_f": -13.995752126582145,
    }


# =========================================================
# 8. Schema
# =========================================================
def init_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA synchronous=NORMAL;")
    c.execute("PRAGMA temp_store=MEMORY;")
    c.execute("PRAGMA foreign_keys=ON;")

    c.execute("""
        CREATE TABLE IF NOT EXISTS fire_objects (
            uid INTEGER PRIMARY KEY,
            lineage_id INTEGER,
            year INTEGER NOT NULL,
            doy INTEGER NOT NULL,
            orig_id INTEGER NOT NULL,

            bbox_y_min INTEGER,
            bbox_x_min INTEGER,
            bbox_y_max INTEGER,
            bbox_x_max INTEGER,
            shape_h INTEGER,
            shape_w INTEGER,

            area_pixels INTEGER NOT NULL,

            n_parents INTEGER NOT NULL DEFAULT 0,
            n_children INTEGER NOT NULL DEFAULT 0,
            is_ignition INTEGER NOT NULL DEFAULT 0,
            is_terminal INTEGER NOT NULL DEFAULT 0,

            mask_blob BLOB NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS fire_transitions (
            parent_uid INTEGER NOT NULL,
            child_uid INTEGER NOT NULL,
            parent_doy INTEGER NOT NULL,
            child_doy INTEGER NOT NULL,
            link_type TEXT NOT NULL,
            PRIMARY KEY (parent_uid, child_uid),
            FOREIGN KEY(parent_uid) REFERENCES fire_objects(uid),
            FOREIGN KEY(child_uid) REFERENCES fire_objects(uid)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS fire_events (
            lineage_id INTEGER PRIMARY KEY,
            year INTEGER NOT NULL,

            start_doy INTEGER NOT NULL,
            end_doy INTEGER NOT NULL,
            duration_days INTEGER NOT NULL,

            peak_object_uid INTEGER NOT NULL,
            peak_object_pixels INTEGER NOT NULL,

            n_objects INTEGER NOT NULL,
            n_ignitions INTEGER NOT NULL,
            n_terminals INTEGER NOT NULL,

            FOREIGN KEY(peak_object_uid) REFERENCES fire_objects(uid)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS event_days (
            lineage_id INTEGER NOT NULL,
            doy INTEGER NOT NULL,
            n_objects INTEGER NOT NULL,
            sum_pixels INTEGER NOT NULL,
            max_object_pixels INTEGER NOT NULL,
            PRIMARY KEY (lineage_id, doy)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS project_metadata (
            field TEXT PRIMARY KEY,
            content TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS spatial_metadata (
            year INTEGER PRIMARY KEY,
            grid_width INTEGER NOT NULL,
            grid_height INTEGER NOT NULL,
            crs_epsg INTEGER,
            crs_name TEXT,
            projection_wkt TEXT,
            origin_x REAL NOT NULL,
            origin_y REAL NOT NULL,
            pixel_size_x REAL NOT NULL,
            pixel_size_y REAL NOT NULL,
            bounds_left REAL NOT NULL,
            bounds_right REAL NOT NULL,
            bounds_top REAL NOT NULL,
            bounds_bottom REAL NOT NULL,
            transform_a REAL,
            transform_b REAL,
            transform_c REAL,
            transform_d REAL,
            transform_e REAL,
            transform_f REAL
        )
    """)

    # indexes
    c.execute("CREATE INDEX IF NOT EXISTS idx_fire_objects_lineage ON fire_objects(lineage_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_fire_objects_lineage_doy ON fire_objects(lineage_id, doy)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_fire_objects_area ON fire_objects(area_pixels DESC)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_fire_objects_doy ON fire_objects(doy)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_fire_objects_ignition ON fire_objects(is_ignition)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_fire_objects_terminal ON fire_objects(is_terminal)")

    c.execute("CREATE INDEX IF NOT EXISTS idx_fire_transitions_parent ON fire_transitions(parent_uid)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_fire_transitions_child ON fire_transitions(child_uid)")

    c.execute("CREATE INDEX IF NOT EXISTS idx_fire_events_peak_pixels ON fire_events(peak_object_pixels DESC)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_event_days_lineage ON event_days(lineage_id)")

    conn.commit()
    return conn


# =========================================================
# 9. Ingest fire_objects
# =========================================================
def ingest_fire_objects(conn, daily_maps, year, uid_to_lineage, parent_to_children, child_to_parents):
    print("Step 3: Processing daily object maps and ingesting fire_objects...")
    c = conn.cursor()
    n_days = daily_maps.shape[0]

    for doy in range(n_days):
        daily_map = daily_maps[doy]

        # IMPORTANT:
        # find_objects index i corresponds to label i+1
        slices = find_objects(daily_map)

        batch = []

        for i, slc in enumerate(slices):
            if slc is None:
                continue

            obj_id = i + 1
            uid = make_uid(doy, obj_id)
            lineage_id = uid_to_lineage.get(uid, -1)

            local_mask = (daily_map[slc] == obj_id).astype(np.uint8)
            area_pixels = int(local_mask.sum())

            n_parents = len(child_to_parents.get(uid, []))
            n_children = len(parent_to_children.get(uid, []))
            is_ignition = 1 if n_parents == 0 else 0
            is_terminal = 1 if n_children == 0 else 0

            batch.append((
                uid,
                lineage_id,
                int(year),
                int(doy),
                int(obj_id),
                int(slc[0].start),
                int(slc[1].start),
                int(slc[0].stop),
                int(slc[1].stop),
                int(local_mask.shape[0]),
                int(local_mask.shape[1]),
                area_pixels,
                n_parents,
                n_children,
                is_ignition,
                is_terminal,
                compress_binary_mask(local_mask),
            ))

        c.executemany("""
            INSERT OR REPLACE INTO fire_objects (
                uid, lineage_id, year, doy, orig_id,
                bbox_y_min, bbox_x_min, bbox_y_max, bbox_x_max,
                shape_h, shape_w, area_pixels,
                n_parents, n_children, is_ignition, is_terminal,
                mask_blob
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch)

        if (doy + 1) % 25 == 0 or (doy + 1) == n_days:
            print(f"  fire_objects: processed {doy + 1}/{n_days} days")

    conn.commit()


# =========================================================
# 10. Ingest transitions
# =========================================================
def ingest_fire_transitions(conn, transition_rows, parent_to_children, child_to_parents):
    print("Step 4: Saving transition links...")
    c = conn.cursor()

    batch = []
    for parent_uid, child_uid, parent_doy, child_doy in transition_rows:
        link_type = classify_link_type(
            parent_uid,
            child_uid,
            parent_to_children,
            child_to_parents
        )
        batch.append((
            int(parent_uid),
            int(child_uid),
            int(parent_doy),
            int(child_doy),
            link_type
        ))

    c.executemany("""
        INSERT OR IGNORE INTO fire_transitions (
            parent_uid, child_uid, parent_doy, child_doy, link_type
        )
        VALUES (?, ?, ?, ?, ?)
    """, batch)

    conn.commit()
    print(f"Successfully inserted {len(batch)} transition links.")


# =========================================================
# 11. Ingest fire_events according to your definition
# =========================================================
def ingest_fire_events(conn, year):
    """
    Event ranking definition:
    - largest event = lineage containing the largest single-day object
    - top 10 events = independent lineages with largest peak objects

    So fire_events stores peak_object_uid and peak_object_pixels.
    """
    print("Step 5: Building fire_events summary table...")
    c = conn.cursor()

    c.execute("DELETE FROM fire_events")

    # one row per lineage, based on the peak object
    c.execute("""
        INSERT INTO fire_events (
            lineage_id, year,
            start_doy, end_doy, duration_days,
            peak_object_uid, peak_object_pixels,
            n_objects, n_ignitions, n_terminals
        )
        WITH ranked AS (
            SELECT
                lineage_id,
                uid,
                doy,
                area_pixels,
                ROW_NUMBER() OVER (
                    PARTITION BY lineage_id
                    ORDER BY area_pixels DESC, doy ASC, uid ASC
                ) AS rn
            FROM fire_objects
            WHERE lineage_id > 0
        )
        SELECT
            fo.lineage_id,
            fo.year,
            MIN(fo.doy) AS start_doy,
            MAX(fo.doy) AS end_doy,
            MAX(fo.doy) - MIN(fo.doy) + 1 AS duration_days,
            MAX(CASE WHEN r.rn = 1 THEN r.uid END) AS peak_object_uid,
            MAX(CASE WHEN r.rn = 1 THEN r.area_pixels END) AS peak_object_pixels,
            COUNT(*) AS n_objects,
            SUM(CASE WHEN fo.is_ignition = 1 THEN 1 ELSE 0 END) AS n_ignitions,
            SUM(CASE WHEN fo.is_terminal = 1 THEN 1 ELSE 0 END) AS n_terminals
        FROM fire_objects fo
        LEFT JOIN ranked r
            ON fo.lineage_id = r.lineage_id
        WHERE fo.lineage_id > 0
        GROUP BY fo.lineage_id, fo.year
    """)

    conn.commit()
    print("fire_events summary built.")


def ingest_event_days(conn):
    print("Step 6: Building event_days table...")
    c = conn.cursor()

    c.execute("DELETE FROM event_days")

    c.execute("""
        INSERT INTO event_days (
            lineage_id, doy, n_objects, sum_pixels, max_object_pixels
        )
        SELECT
            lineage_id,
            doy,
            COUNT(*) AS n_objects,
            SUM(area_pixels) AS sum_pixels,
            MAX(area_pixels) AS max_object_pixels
        FROM fire_objects
        WHERE lineage_id > 0
        GROUP BY lineage_id, doy
    """)

    conn.commit()
    print("event_days table built.")


# =========================================================
# 12. Metadata
# =========================================================
def inject_project_metadata(conn, year):
    print(f"Step 7: Injecting project metadata for {year}...")
    c = conn.cursor()

    metadata_content = [
        ("project", "A Song of Water and Fire - Berkeley FLARE (Fire Lineage and Reconstruction Engine)"),
        ("author", "Tianjiao Pu, Robinson Negrón-Juárez, and Cynthia Gerlein-Safdi"),
        ("email", "putianjiao@berkeley.edu, cgerlein@berkeley.edu"),
        ("lab", "Water and Carbon Lab"),
        ("institution", "University of California, Berkeley"),
        ("link", "https://sites.google.com/berkeley.edu/gerlein-safdi/"),
        ("study region", "Pantanal"),
        ("year", str(year)),
        ("description", "Lineage-resolved wildfire object/event SQLite database"),
        ("event_definition", "Each fire event is an undirected connected component of temporally linked objects."),
        ("largest_event_definition", "Largest event is defined as the lineage containing the largest single-day object."),
        ("data_source", "MCD64A1 Version 6.1 Burned Area data product: https://lpdaac.usgs.gov/products/mcd64a1v061/)"),
        ("version", "1.0 (2025-08)")
    ]

    c.executemany("""
        INSERT OR REPLACE INTO project_metadata (field, content)
        VALUES (?, ?)
    """, metadata_content)

    conn.commit()


def inject_spatial_metadata(conn, spatial_meta):
    print(f"Step 8: Injecting spatial metadata for {spatial_meta['year']}...")
    c = conn.cursor()

    c.execute("""
        INSERT OR REPLACE INTO spatial_metadata (
            year,
            grid_width, grid_height,
            crs_epsg, crs_name, projection_wkt,
            origin_x, origin_y,
            pixel_size_x, pixel_size_y,
            bounds_left, bounds_right, bounds_top, bounds_bottom,
            transform_a, transform_b, transform_c, transform_d, transform_e, transform_f
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        spatial_meta["year"],
        spatial_meta["grid_width"],
        spatial_meta["grid_height"],
        spatial_meta["crs_epsg"],
        spatial_meta["crs_name"],
        spatial_meta["projection_wkt"],
        spatial_meta["origin_x"],
        spatial_meta["origin_y"],
        spatial_meta["pixel_size_x"],
        spatial_meta["pixel_size_y"],
        spatial_meta["bounds_left"],
        spatial_meta["bounds_right"],
        spatial_meta["bounds_top"],
        spatial_meta["bounds_bottom"],
        spatial_meta["transform_a"],
        spatial_meta["transform_b"],
        spatial_meta["transform_c"],
        spatial_meta["transform_d"],
        spatial_meta["transform_e"],
        spatial_meta["transform_f"],
    ))

    conn.commit()


# =========================================================
# 13. Main builder
# =========================================================
def build_year_fire_db(
    year,
    npz_path,
    npy_path,
    db_path,
    raster_shape=(1792, 1671),
):
    npz_path = Path(npz_path)
    npy_path = Path(npy_path)
    db_path = Path(db_path)

    print("=" * 72)
    print(f"Building fire database for year {year}")
    print("=" * 72)
    print(f"Object map file : {npz_path}")
    print(f"Link file       : {npy_path}")
    print(f"Output database : {db_path}")

    tree, link_data = build_tree_from_links(npy_path)
    uid_to_lineage, _ = get_uid_to_lineage_map(tree)
    parent_to_children, child_to_parents, transition_rows = build_transition_stats(link_data)
    daily_maps = load_daily_object_maps(npz_path, raster_shape)
    spatial_meta = build_spatial_metadata_from_gdalinfo(year)

    conn = init_db(db_path)
    try:
        ingest_fire_objects(
            conn=conn,
            daily_maps=daily_maps,
            year=year,
            uid_to_lineage=uid_to_lineage,
            parent_to_children=parent_to_children,
            child_to_parents=child_to_parents,
        )

        ingest_fire_transitions(
            conn=conn,
            transition_rows=transition_rows,
            parent_to_children=parent_to_children,
            child_to_parents=child_to_parents,
        )

        ingest_fire_events(conn, year)
        ingest_event_days(conn)
        inject_project_metadata(conn, year)
        inject_spatial_metadata(conn, spatial_meta)

    finally:
        conn.close()

    print(f"\nSUCCESS: Database saved at {db_path}")
    return str(db_path)


# =========================================================
# 14. Review helpers
# =========================================================
def review_fire_db(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [x[0] for x in cur.fetchall()]
    print("Tables:", tables)

    for t in tables:
        print("\n" + "=" * 60)
        print(f"TABLE: {t}")
        print("=" * 60)

        cur.execute(f"PRAGMA table_info({t});")
        for row in cur.fetchall():
            print(row)

        cur.execute(f"SELECT COUNT(*) FROM {t};")
        print("Row count:", cur.fetchone()[0])

        cur.execute(f"SELECT * FROM {t} LIMIT 3;")
        rows = cur.fetchall()
        print("Preview:")
        for r in rows:
            print(r)

    conn.close()


def get_top_events(db_path, top_n=10):
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT *
        FROM fire_events
        ORDER BY peak_object_pixels DESC, lineage_id ASC
        LIMIT ?
    """, (int(top_n),)).fetchall()
    conn.close()
    return rows
