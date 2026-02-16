# Migration Guide: CSV to JSON Config-Based Training

This guide explains the migration from CSV-based training data to JSON configuration file-based training.

## Overview

The ML system now supports loading training data from JSON configuration files in addition to CSV files. This allows you to use the rich feature extraction from FireXpert configuration JSONs directly.

## Key Changes

### 1. New Data Loader Module

A new module `app.data.config_loader` provides:
- `extract_site_features()`: Extracts features from a single JSON config
- `load_configs_from_directory()`: Loads all JSON configs from a directory
- `load_price_mapping()`: Loads price mappings from CSV

### 2. Updated DataPreprocessor

The `DataPreprocessor` class now supports:
- Loading from JSON config directory (preferred)
- Loading from CSV file (legacy support)
- Optional price mapping file for target values

### 3. Updated Training Scripts

Both training scripts now support JSON configs:
- `scripts/train_model.py`: Added `--config-dir` and `--price-file` arguments
- `app/models/train_with_mlflow.py`: Updated to support JSON config loading

## Usage

### Training with JSON Configs

**Basic usage (without price mapping):**
```bash
python scripts/train_model.py --config-dir data/configs
```

**With price mapping file:**
```bash
python scripts/train_model.py --config-dir data/configs --price-file data/price_mapping.csv
```

**Using MLflow:**
```bash
python -m app.models.train_with_mlflow --config-dir data/configs --price-file data/price_mapping.csv
```

### Creating a Price Mapping File

If you have existing CSV training data with prices, create a mapping file:

```bash
python scripts/create_price_mapping.py --csv data/training_data.csv --output data/price_mapping.csv
```

Or create a template from JSON configs:

```bash
python scripts/create_price_mapping.py --config-dir data/configs --output data/price_mapping.csv
```

Then manually fill in the prices in the CSV file.

### Price Mapping File Format

The price mapping CSV should have two columns:
```csv
site_id,price
configuration-87a5ed07-6bff-47fe-bb76-b9639c8d7ee0-v52-mcu24.11.0-fc9a87bd-e3ad-4cd8-81ff-9824290038e2,5842
configuration-b1ffd616-2d75-4d69-8d25-62f62e69488b-v91-mcu24.11.0-cc94c9d0-041c-413a-b712-c46be74ff987,42820
...
```

The `site_id` should match the JSON filename (without `.json` extension).

## Feature Extraction

The JSON config loader extracts comprehensive features from FireXpert configurations:

### Basic Features
- `site_id`: Configuration identifier
- `n_panels`: Number of panels
- `n_loop_controllers`: Number of loop controllers
- `n_loops_total`: Total number of loops
- `addresses_total`: Total number of addresses

### Aggregated Features
- `addresses_per_loop_mean`: Average addresses per loop
- `addresses_per_loop_max`: Maximum addresses in any loop
- `n_unique_device_types`: Number of unique device types
- `n_unique_protocols`: Number of unique protocols
- `n_zones_used`: Number of zones used
- `manual_call_points`: Count of manual call points
- `monitored_outputs_true`: Count of monitored outputs

### Device Type Counts
- `count_type__{device_type}`: Count for each device type

### Protocol Counts
- `count_protocol__{protocol}`: Count for each protocol

### Output Function Counts
- `count_output_function__{function}`: Count for each output function

### Alarm Threshold Statistics
- `alarm_threshold_mean__{name}`: Mean threshold value
- `alarm_threshold_std__{name}`: Standard deviation
- `alarm_threshold_n__{name}`: Count of thresholds

### Complexity Index
- `complexity_index_v1`: Weighted complexity score

## Backward Compatibility

The system maintains backward compatibility:
- CSV files can still be used with `--data` argument
- If `--config-dir` is not provided, falls back to CSV
- Existing models and preprocessors continue to work

## Configuration

Update `config.json` to use JSON configs:

```json
{
  "data": {
    "config_dir": "data/configs",
    "price_file": "data/price_mapping.csv",
    "test_size": 0.2,
    "random_state": 42
  }
}
```

## Best Practices

1. **Organize JSON configs**: Keep all configuration JSONs in `data/configs/`
2. **Maintain price mapping**: Keep price mappings in a separate CSV for version control
3. **Version control**: Commit JSON configs and price mappings separately
4. **Feature consistency**: Ensure all JSON configs follow the same schema
5. **Error handling**: The loader logs warnings for failed files but continues processing

## Troubleshooting

### No JSON files found
- Check that `data/configs/` directory exists
- Verify JSON files have `.json` extension
- Check file permissions

### Missing prices
- Ensure price mapping file exists and has correct format
- Verify `site_id` in mapping matches JSON filenames
- Check for encoding issues in CSV file

### Import errors
- Ensure `app.data.config_loader` module is accessible
- Check Python path includes project root
- Verify all dependencies are installed

## Migration Steps

1. **Prepare JSON configs**: Place all configuration JSONs in `data/configs/`
2. **Create price mapping**: Use `create_price_mapping.py` script
3. **Update training commands**: Use `--config-dir` instead of `--data`
4. **Retrain models**: Retrain with new data source
5. **Test predictions**: Verify predictions work with new features
6. **Update documentation**: Update any internal docs referencing CSV training
