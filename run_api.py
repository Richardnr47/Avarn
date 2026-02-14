"""
Simple script to run the API.
Usage: python run_api.py
"""

import uvicorn

from app.config import Config

if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=Config.API_RELOAD,
        log_level=Config.LOG_LEVEL.lower()
    )
