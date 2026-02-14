"""
Run Streamlit UI.
Usage: python run_streamlit.py
Or: streamlit run app/ui/streamlit_app.py
"""

import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "app/ui/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])
