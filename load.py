#!/usr/bin/env python3
import argparse
import concurrent.futures
import multiprocessing
import os
import queue
import tempfile
import threading
import uuid as py_uuid
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import fiona
import fsspec
import geopandas as gpd
import pandas as pd
from pyogrio.errors import DataSourceError
from shapely.ops import unary_union

# Connection pool for DuckDB connections
connection_pool = queue.Queue()
pool_initialized = False
pool_lock = threading.Lock()

TMP = tempfile.TemporaryDirectory()
SENTINEL = object()  # Shutdown signal


def initialize_connection_pool(db_path: str, pool_size: int = 8):
    """Initialize the connection pool."""
    global pool_initialized
    with pool_lock:
        if not pool_initialized:
            for _ in range(pool_size):
                conn = duckdb.connect(db_path)
                try:
                    conn.execute("INSTALL spatial;")
                except Exception:
                    pass  # Extension might already be installed
                conn.execute("LOAD spatial;")
                connection_pool.put(conn)
            pool_initialized = True


def get_connection() -> duckdb.DuckDBPyConnection:
    """Get a connection from the pool."""
    return connection_pool.get()


def return_connection(conn: duckdb.DuckDBPyConnection):
    """Return a connection to the pool."""
    connection_pool.put(conn)


def detect_hydrotable_schema(csv_files: List[str]) -> Dict[str, str]:
    """
    Analyze CSV files to detect all unique columns and their types.
    Returns a mapping of column names to SQL types.
    """
    all_columns = set()
    sample_data = {}

    for csv_file in csv_files[:3]:  # Sample first 3 files
        try:
            df = pd.read_csv(csv_file, nrows=100)  # Sample first 100 rows
            all_columns.update(df.columns)
            for col in df.columns:
                if col not in sample_data:
                    sample_data[col] = df[col].dropna().tolist()
                else:
                    sample_data[col].extend(df[col].dropna().tolist()[:10])
        except Exception as e:
            print(f"Warning: Could not sample {csv_file}: {e}")
            continue

    # Determine column types
    schema = {}
    for col in all_columns:
        if col.lower() in ["hydroid", "huc", "lakeid"]:
            schema[col] = "TEXT"
        elif col.lower() in ["feature_id"]:
            schema[col] = "BIGINT"
        elif col.lower() == "stage" or "discharge" in col.lower():
            schema[col] = "DECIMAL[]"  # Array for aggregated values
        elif "calb" in col.lower():
            if col.lower() == "calb_applied":
                schema[col] = "BOOLEAN"  # Boolean for calibration applied flag
            else:
                schema[col] = "DECIMAL[]"  # Array for other calibration values
        else:
            # Try to infer type from sample data
            if col in sample_data and sample_data[col]:
                try:
                    pd.to_numeric(sample_data[col])
                    schema[col] = "DECIMAL"
                except:
                    schema[col] = "TEXT"
            else:
                schema[col] = "TEXT"

    return schema


def adapt_hydrotable_schema(conn: duckdb.DuckDBPyConnection, new_schema: Dict[str, str], hand_ver: str):
    """
    Adapt the Hydrotables schema to include new columns discovered in the CSV files.
    """
    # Get current schema
    try:
        current_cols = conn.execute("DESCRIBE Hydrotables").fetchall()
        current_schema = {row[0]: row[1] for row in current_cols}
    except:
        current_schema = {}

    # Find new columns that need to be added
    new_columns = []
    for col_name, col_type in new_schema.items():
        # Skip core columns that are always present
        if col_name.lower() in ["hydroid", "stage", "feature_id", "huc", "lakeid"]:
            continue

        # Check if this is a discharge or calb column
        if "discharge" in col_name.lower() or "calb" in col_name.lower():
            if col_name not in current_schema:
                new_columns.append((col_name, col_type))

    # Add new columns
    for col_name, col_type in new_columns:
        try:
            alter_sql = f"ALTER TABLE Hydrotables ADD COLUMN {col_name} {col_type}"
            conn.execute(alter_sql)
            print(f"Added new column: {col_name} ({col_type})")
        except Exception as e:
            print(f"Warning: Could not add column {col_name}: {e}")


def process_hydrotable_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process hydrotable dataframe with dynamic column handling.
    """
    # Check required columns exist
    required_cols = ["stage", "feature_id", "HydroID", "HUC", "LakeID"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Convert core columns
    df["stage"] = pd.to_numeric(df["stage"], errors="coerce")
    df["feature_id"] = pd.to_numeric(df["feature_id"], errors="coerce")
    df["HydroID"] = df["HydroID"].astype(str)

    # Convert numeric columns (discharge and calb columns)
    for col in df.columns:
        if "discharge" in col.lower():
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif "calb" in col.lower():
            if col.lower() == "calb_applied":
                # Convert to boolean - handle various boolean representations
                df[col] = df[col].map(
                    lambda x: (
                        True
                        if str(x).lower() in ["true", "1", "yes", "y"]
                        else False if str(x).lower() in ["false", "0", "no", "n"] else None
                    )
                )
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Create aggregation dictionary - keys must be actual column names in dataframe
    agg_dict = {
        "feature_id": lambda s: s.dropna().iloc[0] if not s.dropna().empty else None,
        "HUC": lambda s: s.dropna().iloc[0] if not s.dropna().empty else None,
        "LakeID": lambda s: s.dropna().iloc[0] if not s.dropna().empty else None,
        "stage": lambda v: [float(x) for x in v.dropna() if pd.notna(x)] if any(pd.notna(x) for x in v) else None,
    }

    # Add dynamic discharge and calb columns
    def make_aggregator(column_name):
        def aggregator(v):
            clean_values = [float(x) for x in v.dropna() if pd.notna(x)]
            return clean_values if clean_values else None  # Return None instead of empty array

        return aggregator

    def make_boolean_aggregator(column_name):
        return lambda v: v.dropna().iloc[0] if not v.dropna().empty else None

    for col in df.columns:
        if "discharge" in col.lower():
            agg_dict[col] = make_aggregator(col)
        elif "calb" in col.lower():
            if col.lower() == "calb_applied":
                agg_dict[col] = make_boolean_aggregator(col)
            else:
                agg_dict[col] = make_aggregator(col)

    # Group and aggregate
    df = df.sort_values(["HydroID", "stage"])
    try:
        grp = df.groupby("HydroID").agg(agg_dict).reset_index()
        return grp
    except Exception as e:
        print(f"  Aggregation failed: {e}")
        print(f"  DataFrame columns: {list(df.columns)}")
        print(f"  Aggregation dict: {agg_dict}")
        raise


def initialize_database(db_path: str, schema_path: str):
    """Initialize the DuckDB database with schema."""
    conn = duckdb.connect(db_path)

    # Read and execute schema
    with open(schema_path, "r") as f:
        schema_sql = f.read()

    # Execute each statement separately
    statements = [stmt.strip() for stmt in schema_sql.split(";") if stmt.strip()]
    for stmt in statements:
        if stmt:
            try:
                conn.execute(stmt)
            except Exception as e:
                print(f"Warning: Failed to execute statement: {stmt[:50]}... Error: {e}")

    conn.close()
    print(f"Database initialized at: {db_path}")


def fetch_local(path: str) -> str:
    """Download S3 files to local temp directory."""
    low = path.lower()
    if not low.startswith(("s3://", "s3a://")):
        return path

    fs, anon_path = fsspec.core.url_to_fs(path)
    basename = Path(anon_path).name
    fd, local_path = tempfile.mkstemp(suffix=f"{basename}", dir=str(Path(TMP.name)))
    os.close(fd)
    fs.get(anon_path, local_path)
    return local_path


def list_branch_dirs(hand_dir: str) -> List[str]:
    """List all branch directories."""
    fs, root = fsspec.core.url_to_fs(hand_dir)
    scheme = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
    branches: List[str] = []
    for info in fs.ls(root, detail=True):
        if info["type"] != "directory":
            continue
        br_root = f"{info['name']}/branches"
        if not fs.exists(br_root):
            continue
        for sub in fs.ls(br_root, detail=True):
            if sub["type"] == "directory":
                uri = f"{scheme}://{sub['name']}" if scheme != "file" else sub["name"]
                branches.append(uri)
    return branches


def read_gpkg_fallback(path: str) -> gpd.GeoDataFrame:
    """Read GPKG with fallback to Fiona."""
    try:
        return gpd.read_file(path)
    except DataSourceError:
        with fiona.open(path, driver="GPKG") as src:
            return gpd.GeoDataFrame.from_features(src, crs=src.crs)


def process_branch(args: Tuple[str, str, str]) -> Optional[Dict[str, Any]]:
    """Process one branch directory and return data for batch insertion."""
    d, hand_ver, nwm_ver_str = args
    print(f"Processing branch: {d}")
    nwm_ver = Decimal(nwm_ver_str)

    try:
        # Process catchment geometry union
        fs, anon = fsspec.core.url_to_fs(d)
        gpkg_list = fs.glob(f"{anon}/*gw_catchments*.gpkg")
        geoms = []
        catch_crs = None

        for anon_fp in gpkg_list:
            scheme = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
            uri = f"{scheme}://{anon_fp}" if scheme != "file" else anon_fp
            loc = fetch_local(uri)
            try:
                gdf = read_gpkg_fallback(loc)
                if not gdf.empty:
                    catch_crs = catch_crs or gdf.crs.to_string()
                    geoms.append(unary_union(gdf.geometry))
            except Exception as e:
                print(f"  ERROR: could not open {loc!r} as GPKG: {e}")
            finally:
                if os.path.exists(loc):
                    os.remove(loc)

        if not geoms:
            print(f"  No catchment geometries found in {d}")
            return None

        # Create catchment record data
        merged = unary_union(geoms)
        parts = d.split(f"{hand_ver}/", 1)
        rel_uri = f"{hand_ver}/{parts[1]}" if len(parts) == 2 else d
        cid = py_uuid.uuid5(py_uuid.NAMESPACE_DNS, f"{Path(rel_uri)}:{merged.wkt}")

        # Prepare result data structure
        result_data = {
            "catchment": {
                "catchment_id": str(cid),
                "hand_version_id": hand_ver,
                "geometry_wkt": merged.wkt,
                "additional_attributes": None,
            },
            "hydrotables": [],
            "rem_rasters": [],
            "catchment_rasters": [],
        }

        # Process hydrotables with dynamic schema detection
        csvs = fs.glob(f"{anon}/hydroTable_*.csv")
        if csvs:
            # Download all CSV files first for schema detection
            local_csv_files = []
            for anon_fp in csvs:
                scheme = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
                uri = f"{scheme}://{anon_fp}" if scheme != "file" else anon_fp
                loc = fetch_local(uri)
                local_csv_files.append(loc)

            # Detect schema from CSV files
            detected_schema = detect_hydrotable_schema(local_csv_files)

            # Store schema info for later batch schema adaptation
            result_data["detected_schema"] = detected_schema

            # Process CSV files
            pieces = []
            for loc in local_csv_files:
                try:
                    df_part = pd.read_csv(loc)
                    pieces.append(df_part)
                except Exception as e:
                    print(f"  couldn't read CSV: {loc} because of {e}")
                finally:
                    if os.path.exists(loc):
                        os.remove(loc)

            if pieces:
                df = pd.concat(pieces, ignore_index=True)

                # Process data with dynamic column handling
                grp = process_hydrotable_data(df)

                # Collect hydrotable records for batch insertion
                for _, r in grp.iterrows():
                    hydrotable_record = {
                        "catchment_id": str(cid),
                        "hand_version_id": hand_ver,
                        "HydroID": r["HydroID"],
                        "nwm_feature_id": (
                            int(r["feature_id"]) if "feature_id" in r.index and pd.notna(r["feature_id"]) else None
                        ),
                        "nwm_version_id": (
                            float(nwm_ver) if "feature_id" in r.index and pd.notna(r["feature_id"]) else None
                        ),
                        "stage": r["stage"] if "stage" in r.index else None,
                        "huc_id": (str(r["HUC"]) if "HUC" in r.index and pd.notna(r["HUC"]) else None),
                        "lake_id": (str(r["LakeID"]) if "LakeID" in r.index and pd.notna(r["LakeID"]) else None),
                    }

                    # Add dynamic discharge and calb columns
                    for col in grp.columns:
                        if ("discharge" in col.lower() or "calb" in col.lower()) and col not in hydrotable_record:
                            hydrotable_record[col] = r[col] if col in r.index else None

                    result_data["hydrotables"].append(hydrotable_record)

        # Process REM rasters
        rem_tifs = fs.glob(f"{anon}/*rem_zeroed*.tif")
        rem_ids = []
        if rem_tifs:
            if len(rem_tifs) > 1:
                print(f"WARNING: Multiple REM rasters found in {anon}")

            rem_tif = rem_tifs[0]
            scheme = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
            uri = f"{scheme}://{rem_tif}" if scheme != "file" else rem_tif
            parts = uri.split(f"{hand_ver}/", 1)
            rel_uri = f"{hand_ver}/{parts[1]}" if len(parts) == 2 else uri
            rid = py_uuid.uuid5(py_uuid.NAMESPACE_DNS, f"{cid}:{Path(rel_uri)}")
            rem_ids.append(rid)

            # Collect REM raster data
            result_data["rem_rasters"].append(
                {
                    "rem_raster_id": str(rid),
                    "catchment_id": str(cid),
                    "hand_version_id": hand_ver,
                    "raster_path": uri,
                    "metadata": None,
                }
            )

        # Process catchment rasters
        catch_tifs = fs.glob(f"{anon}/*gw_catchments_reaches*.tif")
        if catch_tifs and rem_ids:
            if len(catch_tifs) > 1:
                print(f"WARNING: Multiple catchment rasters found in {anon}")

            catch_tif = catch_tifs[0]
            scheme = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
            uri = f"{scheme}://{catch_tif}" if scheme != "file" else catch_tif
            parts = uri.split(f"{hand_ver}/", 1)
            rel_uri = f"{hand_ver}/{parts[1]}" if len(parts) == 2 else uri
            crid = py_uuid.uuid5(py_uuid.NAMESPACE_DNS, f"{rem_ids[0]}:{Path(rel_uri)}")

            # Collect catchment raster data
            result_data["catchment_rasters"].append(
                {
                    "catchment_raster_id": str(crid),
                    "rem_raster_id": str(rem_ids[0]),
                    "raster_path": uri,
                    "metadata": None,
                }
            )

        print(f"  Successfully processed branch: {d}")
        return result_data

    except Exception as e:
        print(f"  ERROR processing branch {d}: {e}")
        return None


def batch_insert_data(db_path: str, batch_data: List[Dict[str, Any]]):
    """
    Perform batch insertions into the database using efficient batch operations.
    """
    if not batch_data:
        return

    conn = duckdb.connect(db_path)

    try:
        # Load required extensions
        try:
            conn.execute("INSTALL spatial; LOAD spatial;")
        except:
            pass

        conn.execute("BEGIN TRANSACTION;")

        # Collect all unique schemas for adaptation
        all_schemas = {}
        for data in batch_data:
            if "detected_schema" in data:
                all_schemas.update(data["detected_schema"])
        if all_schemas:
            adapt_hydrotable_schema(conn, all_schemas, "batch")

        # Batch insert catchments
        catchment_records = [data["catchment"] for data in batch_data if data and "catchment" in data]
        if catchment_records:
            print(f"Batch inserting {len(catchment_records)} catchments...")
            conn.executemany(
                """
                INSERT INTO Catchments (catchment_id, hand_version_id, geometry, additional_attributes)
                VALUES (?, ?, ST_GeomFromText(?), ?)
                ON CONFLICT (catchment_id) DO NOTHING
                """,
                [
                    (
                        r["catchment_id"],
                        r["hand_version_id"],
                        r["geometry_wkt"],
                        r["additional_attributes"],
                    )
                    for r in catchment_records
                ],
            )

        # Batch insert hydrotables
        all_hydrotable_records = [
            ht for data in batch_data if data and "hydrotables" in data for ht in data["hydrotables"]
        ]
        if all_hydrotable_records:
            print(f"Batch inserting {len(all_hydrotable_records)} hydrotable records...")
            sample = all_hydrotable_records[0]
            core_columns = [
                "catchment_id",
                "hand_version_id",
                "HydroID",
                "nwm_feature_id",
                "nwm_version_id",
                "stage",
                "huc_id",
                "lake_id",
            ]
            dynamic_columns = [
                k for k in sample.keys() if k not in core_columns and ("discharge" in k.lower() or "calb" in k.lower())
            ]
            all_columns = core_columns + dynamic_columns

            conn.executemany(
                f"""
                INSERT INTO Hydrotables ({', '.join(all_columns)})
                VALUES ({', '.join(['?']*len(all_columns))})
                ON CONFLICT (catchment_id, hand_version_id, HydroID) DO NOTHING
                """,
                [tuple(rec.get(col) for col in all_columns) for rec in all_hydrotable_records],
            )

        # Batch insert REM rasters
        all_rem_rasters = [rr for data in batch_data if data and "rem_rasters" in data for rr in data["rem_rasters"]]
        if all_rem_rasters:
            print(f"Batch inserting {len(all_rem_rasters)} REM rasters...")
            conn.executemany(
                """
                INSERT INTO HAND_REM_Rasters (rem_raster_id, catchment_id, hand_version_id, raster_path, metadata)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (rem_raster_id) DO NOTHING
                """,
                [
                    (
                        r["rem_raster_id"],
                        r["catchment_id"],
                        r["hand_version_id"],
                        r["raster_path"],
                        r["metadata"],
                    )
                    for r in all_rem_rasters
                ],
            )

        # Batch insert catchment rasters
        all_catchment_rasters = [
            cr for data in batch_data if data and "catchment_rasters" in data for cr in data["catchment_rasters"]
        ]
        if all_catchment_rasters:
            print(f"Batch inserting {len(all_catchment_rasters)} catchment rasters...")
            conn.executemany(
                """
                INSERT INTO HAND_Catchment_Rasters (catchment_raster_id, rem_raster_id, raster_path, metadata)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (catchment_raster_id) DO NOTHING
                """,
                [
                    (
                        r["catchment_raster_id"],
                        r["rem_raster_id"],
                        r["raster_path"],
                        r["metadata"],
                    )
                    for r in all_catchment_rasters
                ],
            )

        conn.execute("COMMIT;")
        print(f"Successfully batch inserted data from {len(batch_data)} branches")

    except Exception as e:
        conn.execute("ROLLBACK;")
        print(f"Error in batch insert: {e}")
        raise
    finally:
        conn.close()


def batch_writer(db_path: str, result_queue: queue.Queue, batch_size: int):
    """
    Batch writer thread that accumulates results and inserts to database
    when batch size is reached or on shutdown.
    """
    batch = []
    while True:
        item = result_queue.get()
        if item is SENTINEL:
            break

        batch.append(item)
        if len(batch) >= batch_size:
            try:
                batch_insert_data(db_path, batch)
            except Exception as e:
                print(f"CRITICAL: Batch insert failed: {e}")
                raise
            batch = []

    # Process any remaining items
    if batch:
        try:
            batch_insert_data(db_path, batch)
        except Exception as e:
            print(f"CRITICAL: Final batch insert failed: {e}")
            raise


def load_hand_suite(
    db_path: str,
    hand_dir: str,
    hand_ver: str,
    nwm_ver: Decimal,
    batch_size: int = 200,
):
    """Load HAND data suite into DuckDB with batch processing."""
    # Find all branch dirs
    branch_dirs = list_branch_dirs(hand_dir)
    if not branch_dirs:
        print("No branch directories found → exiting")
        return

    print(f"Found {len(branch_dirs)} branch directories to process")
    args_list = [(d, hand_ver, str(nwm_ver)) for d in branch_dirs]

    # we are doing consumer producer pattern here
    # result_queue is consumer waiting for data from producers
    result_queue = queue.Queue()
    writer_thread = threading.Thread(target=batch_writer, args=(db_path, result_queue, batch_size), daemon=True)
    writer_thread.start()

    successful_count = 0
    total_count = len(branch_dirs)

    # These are producers that will process branches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(process_branch, args) for args in args_list]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    result_queue.put(result)
                    successful_count += 1
            except Exception as e:
                print(f"Error processing branch: {e}")

    # stop the writher thread
    # and wait for it to finish
    result_queue.put(SENTINEL)
    writer_thread.join()

    print(f"Successfully processed {successful_count}/{total_count} branches")


def partition_tables_to_parquet(db_path: str, output_dir: str, h3_resolution: int = 1):
    """Partition tables from DuckDB to parquet files using H3 spatial indexing."""
    conn = duckdb.connect(db_path)

    # Load required extensions
    print("Loading DuckDB extensions...")
    try:
        conn.execute("INSTALL httpfs;")
        conn.execute("LOAD httpfs;")
        conn.execute("INSTALL aws;")
        conn.execute("LOAD aws;")
        conn.execute("INSTALL spatial;")
        conn.execute("LOAD spatial;")
        conn.execute("INSTALL h3 FROM community;")
        conn.execute("LOAD h3;")
    except Exception as e:
        print(f"Error loading extensions: {e}")
        raise

    # Configure AWS settings if using S3
    if output_dir.startswith("s3://"):
        print("Configuring AWS settings for S3 access...")
        try:
            # Try to configure AWS credentials from environment or AWS config
            conn.execute("SET s3_region='us-east-1';")  # Default region
            # You may need to set these if not using default AWS credentials
            # conn.execute("SET s3_access_key_id='your-access-key';")
            # conn.execute("SET s3_secret_access_key='your-secret-key';")
        except Exception as e:
            print(f"Warning: Could not configure AWS settings: {e}")

    # Test S3 connectivity if using S3
    if output_dir.startswith("s3://"):
        print("Testing S3 connectivity...")
        try:
            # Try a simple operation to test connectivity
            conn.execute("SELECT 1;")
        except Exception as e:
            print(f"Error with S3 connectivity test: {e}")
            print(
                "Make sure your AWS credentials are configured (aws configure) and you have write access to the S3 bucket."
            )
            raise

    # Create indexes if they don't exist
    try:
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS catchments_geom_idx
              ON catchments
              USING RTREE (geometry);
        """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hydro_catchment_id ON hydrotables (catchment_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hrr_catchment_id ON hand_rem_rasters (catchment_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hcr_rem_raster_id ON hand_catchment_rasters (rem_raster_id);")
    except Exception as e:
        print(f"Warning: Could not create indexes: {e}")

    # Set H3 resolution variable
    conn.execute(f"SET VARIABLE h3_resolution = {h3_resolution};")

    # Ensure output directory ends with /
    if not output_dir.endswith("/"):
        output_dir += "/"

    print("Partitioning catchments table...")
    # Partition catchments with H3 spatial indexing
    conn.execute(
        f"""
        COPY (
            SELECT
                c.*,
                -- Get centroid, transform to EPSG:4326, then get H3 cell at resolution {h3_resolution}
                h3_latlng_to_cell(
                    ST_Y(ST_Transform(ST_Centroid(c.geometry), 'EPSG:5070', 'EPSG:4326', true)), -- Latitude
                    ST_X(ST_Transform(ST_Centroid(c.geometry), 'EPSG:5070', 'EPSG:4326', true)), -- Longitude
                    getvariable('h3_resolution')
                ) AS h3_partition_key
            FROM catchments c
        ) TO '{output_dir}catchments/'
        WITH (FORMAT PARQUET, PARTITION_BY (h3_partition_key), OVERWRITE_OR_IGNORE 1);
    """
    )

    print("Creating catchment H3 mapping...")
    # Create temp table for catchment to H3 mapping
    conn.execute(
        f"""
        CREATE TEMP TABLE catchment_h3_map AS
        SELECT
            catchment_id,
            h3_latlng_to_cell(
                ST_Y(ST_Transform(ST_Centroid(geometry), 'EPSG:5070', 'EPSG:4326', true)),
                ST_X(ST_Transform(ST_Centroid(geometry), 'EPSG:5070', 'EPSG:4326', true)),
                getvariable('h3_resolution')
            ) AS h3_partition_key
        FROM catchments;
    """
    )

    print("Partitioning hydrotables...")
    # Partition hydrotables
    conn.execute(
        f"""
        COPY (
            SELECT
                ht.*,
                chm.h3_partition_key
            FROM hydrotables ht
            JOIN catchment_h3_map chm ON ht.catchment_id = chm.catchment_id
        ) TO '{output_dir}hydrotables/'
        WITH (FORMAT PARQUET, PARTITION_BY (h3_partition_key), OVERWRITE_OR_IGNORE 1);
    """
    )

    print("Exporting HAND REM rasters (unpartitioned)...")
    # Export hand_rem_rasters as single parquet file
    conn.execute(
        f"""
        COPY hand_rem_rasters TO '{output_dir}hand_rem_rasters.parquet'
        WITH (FORMAT PARQUET, OVERWRITE_OR_IGNORE 1);
    """
    )

    print("Exporting HAND catchment rasters (unpartitioned)...")
    # Export hand_catchment_rasters as single parquet file
    conn.execute(
        f"""
        COPY hand_catchment_rasters TO '{output_dir}hand_catchment_rasters.parquet'
        WITH (FORMAT PARQUET, OVERWRITE_OR_IGNORE 1);
    """
    )

    print("Creating catchment H3 lookup table...")
    # Create H3 lookup table
    conn.execute(
        f"""
        COPY (
            WITH CatchmentCentroids AS (
                -- First, get the H3 cell for the centroid of each catchment
                SELECT
                    c.catchment_id,
                    h3_latlng_to_cell(
                        ST_Y(ST_Transform(ST_Centroid(c.geometry), 'EPSG:5070', 'EPSG:4326', true)), -- Latitude
                        ST_X(ST_Transform(ST_Centroid(c.geometry), 'EPSG:5070', 'EPSG:4326', true)), -- Longitude
                        getvariable('h3_resolution')
                    ) as h3_centroid_cell
                FROM catchments c
                WHERE c.geometry IS NOT NULL AND NOT ST_IsEmpty(c.geometry)
            )
            SELECT
                cc.catchment_id,
                -- For each centroid cell, get the cell itself and all neighbors in a 1-cell radius (k=1)
                unnest(h3_grid_disk(cc.h3_centroid_cell, 1)) AS h3_covering_cell_key
            FROM
                CatchmentCentroids cc
        ) TO '{output_dir}catchment_h3_lookup.parquet'
        WITH (FORMAT PARQUET, OVERWRITE_OR_IGNORE 1);
    """
    )

    # Clean up temp tables
    conn.execute("DROP TABLE IF EXISTS catchment_h3_map;")

    conn.close()
    print(f"Tables partitioned successfully to: {output_dir}")


def main():
    try:
        p = argparse.ArgumentParser()
        p.add_argument("--db-path", required=True, help="Path to DuckDB database file")
        p.add_argument(
            "--schema-path",
            default="./schema/hand-index-v0.1.sql",
            help="Path to DuckDB schema SQL file",
        )
        p.add_argument(
            "--hand-dir",
            required=True,
            help="Root of your HAND HUC8 tree (local path or s3://…)",
        )
        p.add_argument("--hand-version", required=True, help="A text id for this HAND run")
        p.add_argument("--nwm-version", required=True, help="NWM version (decimal)")
        p.add_argument(
            "--init-db",
            action="store_true",
            help="Initialize database with schema (use for new databases)",
        )
        p.add_argument(
            "--output-dir",
            help="Output directory for partitioned parquet files (local path or s3://…). If provided, will partition tables after loading.",
        )
        p.add_argument(
            "--skip-load",
            action="store_true",
            help="Skip loading data to .ddb file if it already exists, only partition existing .ddb",
        )
        p.add_argument(
            "--h3-resolution",
            type=int,
            default=1,
            help="H3 resolution for spatial partitioning (default: 1)",
        )
        p.add_argument(
            "--batch-size",
            type=int,
            default=200,
            help="Number of branches to process in each batch (default: 100)",
        )
        args = p.parse_args()

        # Check if database exists for skip-load option
        db_exists = os.path.exists(args.db_path)

        # Initialize database if requested
        if args.init_db:
            initialize_database(args.db_path, args.schema_path)

        # Load data if not skipping or if database doesn't exist
        if not args.skip_load or not db_exists:
            hand_ver = args.hand_version
            nwm_ver = Decimal(args.nwm_version)

            if args.skip_load and not db_exists:
                print(f"Warning: --skip-load specified but database {args.db_path} does not exist. Loading data...")

            load_hand_suite(args.db_path, args.hand_dir, hand_ver, nwm_ver, args.batch_size)
            print(f"\nData loaded into {args.db_path}")
        else:
            print(f"Skipping data load, using existing database: {args.db_path}")

        # Partition tables if output directory is provided
        if args.output_dir:
            print(f"\nPartitioning tables to: {args.output_dir}")
            partition_tables_to_parquet(args.db_path, args.output_dir, args.h3_resolution)

        print(f"\nDONE.")

    finally:
        if TMP:
            print(f"Cleaning up temporary directory: {TMP.name}")
            try:
                TMP.cleanup()
            except Exception as e:
                print(f"Error cleaning up temporary directory {TMP.name}: {e}")


if __name__ == "__main__":
    main()
