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
    
    print("Starting MLflow UI...")
    print(f"MLruns directory: {mlruns_path}")
    print("\nOpen in browser: http://localhost:5000")
    print("Press CTRL+C to stop\n")
    
    # Start MLflow UI
    subprocess.run([
        sys.executable, "-m", "mlflow", "ui",
        "--backend-store-uri", str(mlruns_path),
        "--host", "127.0.0.1",
        "--port", "5000"
    ])
