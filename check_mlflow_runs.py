"""
Check MLflow runs and experiments.
"""

from pathlib import Path
import yaml

def check_mlflow_runs():
    """Check what's in MLflow directory."""
    mlruns_path = Path("models/mlruns")
    
    if not mlruns_path.exists():
        print("MLruns directory not found!")
        return
    
    print(f"MLruns directory: {mlruns_path.absolute()}\n")
    
    # Check experiments
    experiments = [d for d in mlruns_path.iterdir() if d.is_dir() and d.name.isdigit()]
    
    print(f"Found {len(experiments)} experiment(s):\n")
    
    for exp_dir in experiments:
        meta_file = exp_dir / "meta.yaml"
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                meta = yaml.safe_load(f)
                print(f"Experiment ID: {exp_dir.name}")
                print(f"  Name: {meta.get('name', 'N/A')}")
                print(f"  Artifact Location: {meta.get('artifact_location', 'N/A')}")
        
        # Check for runs
        runs = [d for d in exp_dir.iterdir() if d.is_dir() and len(d.name) == 32]
        print(f"  Runs found: {len(runs)}")
        
        if runs:
            print("  Run IDs:")
            for run_dir in runs[:5]:  # Show first 5
                print(f"    - {run_dir.name}")
            if len(runs) > 5:
                print(f"    ... and {len(runs) - 5} more")
        print()

if __name__ == "__main__":
    try:
        check_mlflow_runs()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
