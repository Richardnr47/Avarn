"""
Setup script for Avarn ML System.
Install with: pip install -e .
"""

from setuptools import find_packages, setup

setup(
    name="avarn-ml",
    version="1.0.0",
    description="Production-inspired ML system for fire alarm testing price prediction",
    author="Richard",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.0.0",
        "mlflow>=2.8.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "python-json-logger>=2.0.0",
        "python-dotenv>=1.0.0",
        "streamlit>=1.28.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "httpx>=0.24.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "ruff>=0.1.0",
            "pre-commit>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "avarn-api=app.api.main:main",
            "avarn-train=app.models.train_with_mlflow:main",
        ],
    },
    python_requires=">=3.9",
)
