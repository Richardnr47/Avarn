"""
Main entry point for the ML API.
Run with: python main.py
Or use: uvicorn app.api.main:app --reload
"""

from app.api.main import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
