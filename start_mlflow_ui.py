"""
Start MLflow UI for viewing experiments.
Usage: python start_mlflow_ui.py
"""

import subprocess
import sys

from app.config import Config

if __name__ == "__main__":
    # Get MLruns path from config
    mlruns_path = Config.MODELS_DIR / "mlruns"
    
    # Check if mlruns directory exists
    if not mlruns_path.exists():
        print(f"Error: MLruns directory not found: {mlruns_path}")
        print("Please train models first using train_with_mlflow.py")
        sys.exit(1)
    
    print("Starting MLflow UI...")
    print(f"MLruns directory: {mlruns_path}")
    print("\nOpen in browser: http://localhost:5000")
    print("Press CTRL+C to stop\n")
    
    # Get URI from config
    mlruns_uri = Config.get_mlflow_uri()
    
    print(f"Using URI: {mlruns_uri}\n")
    
    subprocess.run([
        sys.executable, "-m", "mlflow", "ui",
        "--backend-store-uri", mlruns_uri,
        "--host", "127.0.0.1",
        "--port", "5000"
    ])
