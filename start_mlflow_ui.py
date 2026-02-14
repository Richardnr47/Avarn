"""
Start MLflow UI for viewing experiments.
Usage: python start_mlflow_ui.py
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    # Get project root
    project_root = Path(__file__).parent
    mlruns_path = project_root / "models" / "mlruns"
    
    # Check if mlruns directory exists
    if not mlruns_path.exists():
        print(f"Error: MLruns directory not found: {mlruns_path}")
        print("Please train models first using train_with_mlflow.py")
        sys.exit(1)
    
    print("Starting MLflow UI...")
    print(f"MLruns directory: {mlruns_path}")
    print("\nOpen in browser: http://localhost:5000")
    print("Press CTRL+C to stop\n")
    
    # Start MLflow UI
    # Use file:// prefix for Windows paths - convert backslashes to forward slashes
    mlruns_uri = f"file:///{str(mlruns_path.absolute()).replace(chr(92), '/')}"
    
    print(f"Using URI: {mlruns_uri}\n")
    
    subprocess.run([
        sys.executable, "-m", "mlflow", "ui",
        "--backend-store-uri", mlruns_uri,
        "--host", "127.0.0.1",
        "--port", "5000"
    ])
