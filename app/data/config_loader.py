"""
Data loader for JSON configuration files.
Handles loading and feature extraction from FireXpert configuration JSONs.
"""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_list(x: Any) -> List[Any]:
    """Safely convert value to list."""
    return x if isinstance(x, list) else []


def extract_site_features(cfg: Dict[str, Any], site_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract features from a FireXpert configuration JSON.
    
    Flattens nested JSON structure into a single feature row suitable for ML.
    
    Args:
        cfg: Configuration dictionary loaded from JSON
        site_id: Optional site identifier
        
    Returns:
        Dictionary of extracted features
    """
    out: Dict[str, Any] = {}
    
    # Basic metadata
    out["site_id"] = site_id
    out["created_by_name"] = (cfg.get("created_by") or {}).get("name")
    out["version_number"] = (cfg.get("version") or {}).get("number")
    out["version_schema"] = (cfg.get("version") or {}).get("schema")
    out["version_date"] = (cfg.get("version") or {}).get("date")
    
    # System structure
    system = cfg.get("system") or {}
    panels = _safe_list(system.get("panels"))
    out["n_panels"] = len(panels)
    
    # Aggregate containers
    type_counter = Counter()
    protocol_counter = Counter()
    zone_counter = Counter()
    
    n_loop_controllers = 0
    n_loops_total = 0
    addresses_total = 0
    addresses_per_loop: List[int] = []
    
    manual_call_points = 0
    has_zone_disables = 0
    
    # Threshold aggregates
    thresh_values_by_name = defaultdict(list)
    
    output_function_counter = Counter()
    monitored_output_true = 0
    
    # Panel metadata (use first panel if multiple)
    out["panel_name_first"] = panels[0].get("name") if panels else None
    out["primary_language_first"] = panels[0].get("primary_language") if panels else None
    out["secondary_language_first"] = panels[0].get("secondary_language") if panels else None
    
    # Iterate through panels, loop controllers, loops, and addresses
    for p in panels:
        lcs = _safe_list(p.get("loop_controllers"))
        n_loop_controllers += len(lcs)
        
        for lc in lcs:
            loops = _safe_list(lc.get("loops"))
            n_loops_total += len(loops)
            
            for loop in loops:
                addrs = _safe_list(loop.get("addresses"))
                n_addr = len(addrs)
                addresses_total += n_addr
                addresses_per_loop.append(n_addr)
                
                for a in addrs:
                    a_type = a.get("type")
                    if a_type:
                        type_counter[a_type] += 1
                    
                    prot = a.get("protocol_type")
                    if prot:
                        protocol_counter[prot] += 1
                    
                    zone = a.get("zone")
                    if zone is not None:
                        zone_counter[zone] += 1
                    
                    if _safe_list(a.get("zone_disables")):
                        has_zone_disables += 1
                    
                    # Manual call point detection
                    if a.get("input_function") == "Manual call point":
                        manual_call_points += 1
                    
                    # Output module details
                    if "output_control" in a and isinstance(a["output_control"], dict):
                        oc = a["output_control"]
                        of = oc.get("output_function")
                        if of:
                            output_function_counter[of] += 1
                        if oc.get("monitored_output_mode") is True:
                            monitored_output_true += 1
                    
                    # Alarm thresholds
                    for t in _safe_list(a.get("alarm_thresholds")):
                        name = t.get("name")
                        value = t.get("value")
                        if name is not None and value is not None:
                            thresh_values_by_name[name].append(value)
    
    # Aggregate counts
    out["n_loop_controllers"] = n_loop_controllers
    out["n_loops_total"] = n_loops_total
    out["addresses_total"] = addresses_total
    
    out["addresses_per_loop_mean"] = (
        float(np.mean(addresses_per_loop)) if addresses_per_loop else 0.0
    )
    out["addresses_per_loop_max"] = (
        int(np.max(addresses_per_loop)) if addresses_per_loop else 0
    )
    
    out["n_unique_device_types"] = len(type_counter)
    out["n_unique_protocols"] = len(protocol_counter)
    out["n_zones_used"] = len(zone_counter)
    
    out["manual_call_points"] = manual_call_points
    out["addresses_with_zone_disables"] = has_zone_disables
    
    out["monitored_outputs_true"] = monitored_output_true
    
    # Expand counters into feature columns
    for k, v in type_counter.items():
        out[f"count_type__{k}"] = int(v)
    
    for k, v in protocol_counter.items():
        out[f"count_protocol__{k}"] = int(v)
    
    # Zone aggregates
    if zone_counter:
        devices_per_zone = list(zone_counter.values())
        out["devices_per_zone_mean"] = float(np.mean(devices_per_zone))
        out["devices_per_zone_max"] = int(np.max(devices_per_zone))
    else:
        out["devices_per_zone_mean"] = 0.0
        out["devices_per_zone_max"] = 0
    
    # Output functions
    for k, v in output_function_counter.items():
        out[f"count_output_function__{k}"] = int(v)
    
    # Threshold statistics
    for name, vals in thresh_values_by_name.items():
        if vals:
            out[f"alarm_threshold_mean__{name}"] = float(np.mean(vals))
            out[f"alarm_threshold_std__{name}"] = float(np.std(vals))
            out[f"alarm_threshold_n__{name}"] = int(len(vals))
        else:
            out[f"alarm_threshold_mean__{name}"] = np.nan
            out[f"alarm_threshold_std__{name}"] = np.nan
            out[f"alarm_threshold_n__{name}"] = 0
    
    # Complexity index (weighted combination of features)
    out["complexity_index_v1"] = (
        1.0 * addresses_total
        + 5.0 * n_loop_controllers
        + 2.0 * n_loops_total
        + 3.0 * manual_call_points
        + 2.0 * monitored_output_true
        + 1.0 * out["n_unique_device_types"]
    )
    
    return out


def load_configs_from_directory(
    config_dir: str | Path,
    price_mapping: Optional[Dict[str, float]] = None,
    price_column: str = "price",
) -> pd.DataFrame:
    """
    Load all JSON configuration files from a directory and extract features.
    
    Args:
        config_dir: Path to directory containing JSON config files
        price_mapping: Optional dictionary mapping site_id to price values
        price_column: Name of the price column (default: "price")
        
    Returns:
        DataFrame with extracted features and optional price column
        
    Raises:
        FileNotFoundError: If config directory doesn't exist
        ValueError: If no JSON files found in directory
    """
    config_path = Path(config_dir)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration directory not found: {config_dir}")
    
    if not config_path.is_dir():
        raise ValueError(f"Path is not a directory: {config_dir}")
    
    # Find all JSON files
    json_files = list(config_path.glob("*.json"))
    
    if not json_files:
        raise ValueError(f"No JSON files found in directory: {config_dir}")
    
    logger.info(f"Found {len(json_files)} JSON configuration files")
    
    rows = []
    failed_files = []
    
    for json_file in json_files:
        try:
            # Extract site_id from filename (remove extension)
            site_id = json_file.stem
            
            # Load JSON
            with open(json_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            
            # Extract features
            row = extract_site_features(cfg, site_id=site_id)
            
            # Add price if mapping provided
            if price_mapping and site_id in price_mapping:
                row[price_column] = price_mapping[site_id]
            elif price_mapping is not None:
                logger.warning(f"No price mapping found for site_id: {site_id}")
            
            rows.append(row)
            
        except Exception as e:
            logger.error(f"Failed to process {json_file.name}: {e}")
            failed_files.append(json_file.name)
    
    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files: {failed_files}")
    
    if not rows:
        raise ValueError("No valid configuration files could be processed")
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Fill NaN values with 0 for numeric columns, None for object columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(0)
    
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].fillna("")
    
    logger.info(f"Successfully loaded {len(df)} configurations with {len(df.columns)} features")
    
    return df


def load_price_mapping(price_file: str | Path) -> Dict[str, float]:
    """
    Load price mapping from CSV file.
    
    Expected CSV format: site_id,price
    
    Args:
        price_file: Path to CSV file with site_id and price columns
        
    Returns:
        Dictionary mapping site_id to price
    """
    price_path = Path(price_file)
    
    if not price_path.exists():
        logger.warning(f"Price mapping file not found: {price_file}")
        return {}
    
    try:
        df = pd.read_csv(price_path)
        
        # Try to find site_id and price columns (case-insensitive)
        site_col = None
        price_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if "site" in col_lower and "id" in col_lower:
                site_col = col
            elif col_lower == "price":
                price_col = col
        
        if site_col is None or price_col is None:
            raise ValueError(
                f"CSV must contain 'site_id' and 'price' columns. Found: {df.columns.tolist()}"
            )
        
        mapping = dict(zip(df[site_col], df[price_col]))
        logger.info(f"Loaded {len(mapping)} price mappings from {price_file}")
        
        return mapping
        
    except Exception as e:
        logger.error(f"Failed to load price mapping: {e}")
        return {}
