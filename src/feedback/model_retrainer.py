"""Re-entrainement du modele avec backup et validation.

Ne remplace le modele actif que si le nouveau est strictement meilleur.
"""

import os
import shutil
from datetime import datetime

from loguru import logger

from src.core.database import Database


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
        """Retrain model, compare with current, deploy only if better.

        Returns dict with: deployed (bool), old_metrics, new_metrics, backup_path,
        and optionally new_path and version if deployed.
        """
        from src.analysis.feature_engine import FeatureEngine
        from src.model.trainer import Trainer

        backup_path = self._backup_model(current_model_path)

        # Load old model and evaluate via walk-forward
        old_trainer = Trainer()
        old_trainer.load_model(current_model_path)

        engine = FeatureEngine(self.db)
        features_df = engine.build_combined_features()

        if len(features_df) < 20:
            return {"deployed": False, "reason": "not_enough_data"}

        old_results = old_trainer.walk_forward_validate(features_df)
        old_metrics = {
            k: old_results.get(k, 0)
            for k in ["accuracy", "precision", "recall", "f1"]
        }

        # Train new model
        new_trainer = Trainer()
        X, y = new_trainer.prepare_data(features_df)
        new_trainer.train(X, y)
        new_results = new_trainer.walk_forward_validate(features_df)
        new_metrics = {
            k: new_results.get(k, 0)
            for k in ["accuracy", "precision", "recall", "f1"]
        }

        is_better = self._is_new_model_better(old_metrics, new_metrics)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stats = self.db.get_review_stats()

        result = {
            "deployed": is_better,
            "old_metrics": old_metrics,
            "new_metrics": new_metrics,
            "backup_path": backup_path,
        }

        if is_better:
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
                "notes": (
                    f"Retrained, f1 {old_metrics['f1']:.3f}"
                    f" -> {new_metrics['f1']:.3f}"
                ),
            })
            versions = self.db.get_all_model_versions()
            new_v = [v for v in versions if v["version"] == version][0]
            self.db.set_active_model(new_v["id"])
            result.update({"new_path": new_path, "version": version})
            logger.info(
                f"Nouveau modele deploye: {version} "
                f"(f1 {old_metrics['f1']:.3f} -> {new_metrics['f1']:.3f})"
            )
        else:
            logger.info(
                f"Ancien modele conserve "
                f"(f1 {old_metrics['f1']:.3f} vs {new_metrics['f1']:.3f})"
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

    def _is_new_model_better(self, old_metrics: dict, new_metrics: dict) -> bool:
        """New model must have >= 1% higher f1 to be considered better."""
        return (new_metrics.get("f1", 0) - old_metrics.get("f1", 0)) >= 0.01

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
        old = result.get("old_metrics", {})
        new = result.get("new_metrics", {})
        lines = ["<b>Re-entrainement du modele</b>", ""]

        if result.get("deployed"):
            lines.append(
                f"Nouveau modele deploye: {result.get('version', '?')}"
            )
        else:
            lines.append(
                "Ancien modele conserve (nouveau pas assez meilleur)"
            )

        lines.extend(["", "Metriques comparees:"])
        for metric in ["accuracy", "precision", "recall", "f1"]:
            old_v = old.get(metric, 0)
            new_v = new.get(metric, 0)
            lines.append(f"  {metric.capitalize()}: {old_v:.1%} -> {new_v:.1%}")

        lines.extend(["", f"Backup: {result.get('backup_path', 'N/A')}"])
        return "\n".join(lines)
