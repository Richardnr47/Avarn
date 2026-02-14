"""
Prediction logging and monitoring.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from app.config import Config

logger = logging.getLogger(__name__)


class PredictionLogger:
    """
    Logs predictions for monitoring and analysis.
    """

    def __init__(self, log_dir: Path = None):
        """
        Initialize prediction logger.

        Args:
            log_dir: Directory to store prediction logs (defaults to Config.LOG_DIR)
        """
        if log_dir is None:
            log_dir = Config.LOG_DIR

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # CSV file for predictions
        self.predictions_file = self.log_dir / "predictions.csv"
        self._init_csv()

    def _init_csv(self) -> None:
        """Initialize CSV file with headers if it doesn't exist."""
        if not self.predictions_file.exists():
            df = pd.DataFrame(
                columns=[
                    "prediction_id",
                    "timestamp",
                    "predicted_price",
                    "antal_sektioner",
                    "antal_detektorer",
                    "antal_larmdon",
                    "dörrhållarmagneter",
                    "ventilation",
                    "stad",
                    "kvartalsvis",
                    "månadsvis",
                    "årsvis",
                    "model_version",
                    "pipeline_version",
                ]
            )
            df.to_csv(self.predictions_file, index=False)

    def log_prediction(
        self,
        prediction_id: str,
        request: Dict[str, Any],
        response: Dict[str, Any],
        timestamp: datetime,
    ) -> None:
        """
        Log a prediction to CSV and JSON.

        Args:
            prediction_id: Unique prediction ID
            request: Request data
            response: Response data
            timestamp: Prediction timestamp
        """
        try:
            # Prepare log entry
            log_entry = {
                "prediction_id": prediction_id,
                "timestamp": timestamp.isoformat(),
                "predicted_price": response.get("predicted_price"),
                "antal_sektioner": request.get("antal_sektioner"),
                "antal_detektorer": request.get("antal_detektorer"),
                "antal_larmdon": request.get("antal_larmdon"),
                "dörrhållarmagneter": request.get("dörrhållarmagneter"),
                "ventilation": request.get("ventilation"),
                "stad": request.get("stad"),
                "kvartalsvis": request.get("kvartalsvis"),
                "månadsvis": request.get("månadsvis"),
                "årsvis": request.get("årsvis"),
                "model_version": response.get("model_version"),
                "pipeline_version": response.get("feature_pipeline_version"),
            }

            # Append to CSV
            df = pd.DataFrame([log_entry])
            df.to_csv(self.predictions_file, mode="a", header=False, index=False)

            # Also log to JSON file (daily rotation)
            date_str = timestamp.strftime("%Y-%m-%d")
            json_file = self.log_dir / f"predictions_{date_str}.jsonl"

            with open(json_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            # Log rotation: keep only last 30 days of JSONL files
            self._rotate_logs(days_to_keep=30)

            logger.debug(f"Logged prediction: {prediction_id}")

        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")

    def get_prediction_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        Get statistics on predictions.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with statistics
        """
        try:
            df = pd.read_csv(self.predictions_file)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Filter by date
            cutoff = datetime.now() - pd.Timedelta(days=days)
            df_recent = df[df["timestamp"] >= cutoff]

            if len(df_recent) == 0:
                return {"total_predictions": 0, "period_days": days}

            return {
                "total_predictions": len(df_recent),
                "period_days": days,
                "avg_predicted_price": float(df_recent["predicted_price"].mean()),
                "min_predicted_price": float(df_recent["predicted_price"].min()),
                "max_predicted_price": float(df_recent["predicted_price"].max()),
                "predictions_by_stad": df_recent["stad"].value_counts().to_dict(),
                "predictions_by_frequency": {
                    "kvartalsvis": int((df_recent["kvartalsvis"] == 1).sum()),
                    "månadsvis": int((df_recent["månadsvis"] == 1).sum()),
                    "årsvis": int((df_recent["årsvis"] == 1).sum()),
                },
            }

        except Exception as e:
            logger.error(f"Failed to get prediction stats: {e}")
            return {"error": str(e)}

    def _rotate_logs(self, days_to_keep: int = 30) -> None:
        """
        Rotate log files, keeping only the last N days.

        Args:
            days_to_keep: Number of days of logs to keep
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d")

            for log_file in self.log_dir.glob("predictions_*.jsonl"):
                # Extract date from filename: predictions_YYYY-MM-DD.jsonl
                date_str = log_file.stem.split("_")[-1]
                if date_str < cutoff_str:
                    log_file.unlink()
                    logger.debug(f"Rotated old log file: {log_file.name}")

        except Exception as e:
            logger.warning(f"Failed to rotate logs: {e}")
