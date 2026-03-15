"""Re-entrainement du modele avec backup et validation.

Deploie le nouveau modele si son f1 depasse un seuil minimum.
Le retrain quotidien permet au modele d'apprendre en continu.
"""

import os
import shutil
from datetime import datetime

from loguru import logger

from src.core.database import Database

MIN_F1_THRESHOLD = 0.25


class ModelRetrainer:
    """Gere le re-entrainement du modele avec validation."""

    def __init__(self, db: Database, min_reviews_for_retrain: int = 20):
        self.db = db
        self.min_reviews = min_reviews_for_retrain

    def should_retrain(self) -> bool:
        """Check if enough reviews accumulated since last training."""
        stats = self.db.get_review_stats()
        total = stats["total"]
        active_model = self.db.get_active_model_version()
        last_training_signals = (
            active_model.get("training_signals", 0) if active_model else 0
        )
        new_reviews = total - last_training_signals
        return new_reviews >= self.min_reviews

    def retrain_with_validation(
        self,
        current_model_path: str,
        new_model_dir: str = "data/models",
    ) -> dict:
        """Retrain et deployer si f1 du nouveau modele >= seuil minimum.

        Le nouveau modele est entraine sur les donnees combinees (trades
        historiques + signal reviews). Il est deploye si son f1 >= 0.25.
        Plus besoin de comparer avec l'ancien (qui etait biaise).

        Returns dict with: deployed (bool), new_metrics, backup_path,
        and optionally new_path and version if deployed.
        """
        from src.analysis.feature_engine import FeatureEngine
        from src.model.trainer import Trainer

        backup_path = self._backup_model(current_model_path)

        engine = FeatureEngine(self.db)
        features_df = engine.build_combined_features()

        if len(features_df) < 20:
            return {"deployed": False, "reason": "not_enough_data"}

        # Entrainer le nouveau modele
        new_trainer = Trainer()
        X, y = new_trainer.prepare_data(features_df)
        new_trainer.train(X, y)
        new_results = new_trainer.walk_forward_validate(features_df)
        new_metrics = {
            k: new_results.get(k, 0)
            for k in ["accuracy", "precision", "recall", "f1"]
        }

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stats = self.db.get_review_stats()

        result = {
            "deployed": False,
            "new_metrics": new_metrics,
            "backup_path": backup_path,
        }

        if new_metrics["f1"] >= MIN_F1_THRESHOLD:
            version = self._next_version()
            new_path = os.path.join(new_model_dir, f"nicolas_{version}.joblib")
            new_trainer.save_model(new_path)
            self.db.insert_model_version({
                "version": version,
                "file_path": new_path,
                "trained_at": now,
                "training_signals": stats["total"],
                "accuracy": new_metrics["accuracy"],
                "precision_score": new_metrics["precision"],
                "recall": new_metrics["recall"],
                "f1": new_metrics["f1"],
                "is_active": 0,
                "notes": f"Retrained on {len(features_df)} samples, f1={new_metrics['f1']:.3f}",
            })
            versions = self.db.get_all_model_versions()
            new_v = [v for v in versions if v["version"] == version][0]
            self.db.set_active_model(new_v["id"])
            result.update({"deployed": True, "new_path": new_path, "version": version})
            logger.info(
                f"Modele {version} deploye "
                f"(f1={new_metrics['f1']:.3f}, {len(features_df)} samples)"
            )
        else:
            result["reason"] = "f1_too_low"
            logger.info(
                f"Modele non deploye: f1={new_metrics['f1']:.3f} < {MIN_F1_THRESHOLD}"
            )

        return result

    def _backup_model(self, model_path: str) -> str:
        """Create a timestamped backup of the current model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modele non trouve: {model_path}")
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(model_path)
        backup_path = f"{base}_backup_{date_str}{ext}"
        shutil.copy2(model_path, backup_path)
        logger.debug(f"Backup modele: {backup_path}")
        return backup_path

    def _meets_min_quality(self, metrics: dict) -> bool:
        """Le nouveau modele doit avoir un f1 >= seuil minimum."""
        return metrics.get("f1", 0) >= MIN_F1_THRESHOLD

    def _next_version(self) -> str:
        """Determine next version number (v2, v3, ...)."""
        versions = self.db.get_all_model_versions()
        if not versions:
            return "v2"
        max_num = 1
        for v in versions:
            try:
                num = int(v["version"].replace("v", ""))
                max_num = max(max_num, num)
            except ValueError:
                pass
        return f"v{max_num + 1}"

    def format_retrain_report(self, result: dict) -> str:
        """Format retrain report as HTML for Telegram."""
        new = result.get("new_metrics", {})
        lines = ["<b>Re-entrainement du modele</b>", ""]

        if result.get("deployed"):
            lines.append(
                f"Nouveau modele deploye: {result.get('version', '?')}"
            )
        else:
            reason = result.get("reason", "unknown")
            lines.append(f"Modele non deploye ({reason})")

        lines.extend(["", "Metriques:"])
        for metric in ["accuracy", "precision", "recall", "f1"]:
            val = new.get(metric, 0)
            lines.append(f"  {metric.capitalize()}: {val:.1%}")

        lines.extend(["", f"Backup: {result.get('backup_path', 'N/A')}"])
        return "\n".join(lines)
