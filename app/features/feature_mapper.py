"""
Feature mapping from simple UI inputs to full JSON config feature set.
Converts simple prediction request features to the comprehensive feature set
extracted from JSON configurations.
"""

from typing import Any, Dict


def map_simple_to_full_features(simple_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map simple UI features to full JSON config feature set.
    
    This function converts the simple features from the UI/API (antal_sektioner,
    antal_detektorer, etc.) to the full feature set expected by models trained
    on JSON config data.
    
    Args:
        simple_features: Dictionary with simple features:
            - antal_sektioner: int
            - antal_detektorer: int
            - antal_larmdon: int
            - dörrhållarmagneter: int
            - ventilation: int (0 or 1)
            - stad: str
            - kvartalsvis: int (0 or 1)
            - månadsvis: int (0 or 1)
            - årsvis: int (0 or 1)
    
    Returns:
        Dictionary with full feature set matching JSON config extraction
    """
    # Start with basic metadata (use defaults)
    full_features: Dict[str, Any] = {
        "site_id": None,
        "created_by_name": None,
        "version_number": None,
        "version_schema": None,
        "version_date": None,
    }
    
    # Map simple features to JSON config features
    # Use heuristics based on typical relationships
    
    # Basic counts - map directly where possible
    n_detectors = simple_features.get("antal_detektorer", 0)
    n_alarm_devices = simple_features.get("antal_larmdon", 0)
    n_sections = simple_features.get("antal_sektioner", 0)
    
    # Estimate addresses_total (detectors + alarm devices + some overhead)
    addresses_total = n_detectors + n_alarm_devices + max(0, n_sections - 1) * 2
    
    # Estimate loop controllers (typically 1 per section, min 1)
    n_loop_controllers = max(1, n_sections)
    
    # Estimate loops (typically 1-2 per controller)
    n_loops_total = n_loop_controllers * 2
    
    # Calculate addresses per loop
    addresses_per_loop = addresses_total / max(1, n_loops_total)
    
    # Map to full features
    full_features["n_panels"] = 1  # Assume single panel for simple input
    full_features["n_loop_controllers"] = n_loop_controllers
    full_features["n_loops_total"] = n_loops_total
    full_features["addresses_total"] = addresses_total
    full_features["addresses_per_loop_mean"] = float(addresses_per_loop)
    full_features["addresses_per_loop_max"] = int(addresses_per_loop * 1.5)  # Estimate
    
    # Device types - estimate based on inputs
    # Assume most detectors are optical smoke detectors
    n_optical = int(n_detectors * 0.7)  # ~70% optical
    n_thermal = int(n_detectors * 0.2)  # ~20% thermal
    n_beam = int(n_detectors * 0.1)  # ~10% beam
    
    # Set device type counts (only set non-zero values to match training data)
    if n_optical > 0:
        full_features["count_type__Optical smoke detector"] = n_optical
    if n_thermal > 0:
        full_features["count_type__Static thermal detector"] = n_thermal
    if n_beam > 0:
        full_features["count_type__Beam detector"] = n_beam
    
    # Input/output modules
    n_input_modules = max(1, int(n_alarm_devices * 0.3))
    n_output_modules = max(1, int(n_alarm_devices * 0.2))
    
    if n_input_modules > 0:
        full_features["count_type__Single input module"] = n_input_modules
    if n_output_modules > 0:
        full_features["count_type__Single output module"] = n_output_modules
    
    # Manual call points (from dörrhållarmagneter - door holders often have call points)
    manual_call_points = max(0, int(simple_features.get("dörrhållarmagneter", 0) * 0.5))
    full_features["manual_call_points"] = manual_call_points
    
    # Protocol (assume SySe S200 for most devices)
    if addresses_total > 0:
        full_features["count_protocol__SySe S200"] = int(addresses_total * 0.8)
        if addresses_total > 20:
            full_features["count_protocol__SySe AP200"] = int(addresses_total * 0.2)
    
    # Zones - estimate based on sections
    n_zones = max(1, n_sections)
    devices_per_zone = addresses_total / max(1, n_zones)
    full_features["n_zones_used"] = n_zones
    full_features["devices_per_zone_mean"] = float(devices_per_zone)
    full_features["devices_per_zone_max"] = int(devices_per_zone * 1.3)
    
    # Unique counts
    full_features["n_unique_device_types"] = len([k for k in full_features.keys() if k.startswith("count_type__")])
    full_features["n_unique_protocols"] = len([k for k in full_features.keys() if k.startswith("count_protocol__")])
    
    # Zone disables (assume some addresses have zone disables)
    full_features["addresses_with_zone_disables"] = max(0, int(addresses_total * 0.1))
    
    # Output functions - estimate based on alarm devices
    if n_output_modules > 0:
        full_features["count_output_function__Fire alarm output"] = int(n_output_modules * 0.4)
        full_features["count_output_function__Fire alarm device output"] = int(n_output_modules * 0.3)
        full_features["count_output_function__Internal logic output"] = int(n_output_modules * 0.2)
        full_features["count_output_function__Fire door output"] = int(n_output_modules * 0.1)
    
    # Monitored outputs
    full_features["monitored_outputs_true"] = max(0, int(n_output_modules * 0.5))
    
    # Panel metadata
    full_features["panel_name_first"] = None
    full_features["primary_language_first"] = "sv"  # Swedish
    full_features["secondary_language_first"] = None
    
    # Alarm thresholds - set defaults based on typical values
    # Fire threshold
    full_features["alarm_threshold_mean__Fire"] = 50.0
    full_features["alarm_threshold_std__Fire"] = 5.0
    full_features["alarm_threshold_n__Fire"] = n_detectors
    
    # Prealarm threshold
    full_features["alarm_threshold_mean__Prealarm"] = 30.0
    full_features["alarm_threshold_std__Prealarm"] = 3.0
    full_features["alarm_threshold_n__Prealarm"] = n_detectors
    
    # Day mode thresholds (if applicable)
    if simple_features.get("kvartalsvis", 0) == 0:  # Not quarterly = might have day mode
        full_features["alarm_threshold_mean__Fire, day mode"] = 60.0
        full_features["alarm_threshold_std__Fire, day mode"] = 6.0
        full_features["alarm_threshold_n__Fire, day mode"] = int(n_detectors * 0.5)
        
        full_features["alarm_threshold_mean__Prealarm, day mode"] = 35.0
        full_features["alarm_threshold_std__Prealarm, day mode"] = 4.0
        full_features["alarm_threshold_n__Prealarm, day mode"] = int(n_detectors * 0.5)
    
    # Complexity index
    full_features["complexity_index_v1"] = (
        1.0 * addresses_total
        + 5.0 * n_loop_controllers
        + 2.0 * n_loops_total
        + 3.0 * manual_call_points
        + 2.0 * full_features["monitored_outputs_true"]
        + 1.0 * full_features["n_unique_device_types"]
    )
    
    # Add all other device types that might be in training data (set to 0 if not present)
    # These are from the error message - ensure they exist even if 0
    additional_device_types = [
        "Manual call point, indoor",
        "Conventional zone module",
        "Detector base sounder",
        "Dual input, single output module",
        "Input channel",
        "Output channel",
        "Three-criteria detector",
        "Detector base sounder strobe",
        "Conventional zone module, CZR",
        "Multi-criteria detector",
    ]
    
    for device_type in additional_device_types:
        key = f"count_type__{device_type}"
        if key not in full_features:
            full_features[key] = 0
    
    # Add all output functions that might be in training data
    additional_output_functions = [
        "Fire alarm device output, non-silenceable",
        "Not in use",
    ]
    
    for output_func in additional_output_functions:
        key = f"count_output_function__{output_func}"
        if key not in full_features:
            full_features[key] = 0
    
    # Ensure day mode thresholds always exist (even if 0)
    if "alarm_threshold_mean__Fire, day mode" not in full_features:
        full_features["alarm_threshold_mean__Fire, day mode"] = 0.0
        full_features["alarm_threshold_std__Fire, day mode"] = 0.0
        full_features["alarm_threshold_n__Fire, day mode"] = 0
    
    if "alarm_threshold_mean__Prealarm, day mode" not in full_features:
        full_features["alarm_threshold_mean__Prealarm, day mode"] = 0.0
        full_features["alarm_threshold_std__Prealarm, day mode"] = 0.0
        full_features["alarm_threshold_n__Prealarm, day mode"] = 0
    
    return full_features
