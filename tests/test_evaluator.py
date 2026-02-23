"""Tests pour l'evaluateur du modele."""

import numpy as np
import pandas as pd
import pytest

from src.model.evaluator import Evaluator


class TestFeatureImportance:
    """Tests de feature importance."""

    def setup_method(self):
        self.evaluator = Evaluator()

    def test_returns_dataframe(self):
        """feature_importance retourne un DataFrame."""
        from unittest.mock import MagicMock
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.3, 0.2, 0.1, 0.05, 0.35])
        names = ["range_position_10", "rsi_14", "catalyst_type", "day_of_week", "nb_catalysts"]

        result = self.evaluator.feature_importance(mock_model, names)
        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "importance" in result.columns

    def test_sorted_by_importance(self):
        """Les features sont triees par importance decroissante."""
        from unittest.mock import MagicMock
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.1, 0.3, 0.6])
        names = ["a", "b", "c"]

        result = self.evaluator.feature_importance(mock_model, names)
        assert result.iloc[0]["feature"] == "c"
        assert result.iloc[0]["importance"] == 0.6


class TestErrorAnalysis:
    """Tests d'analyse des erreurs."""

    def setup_method(self):
        self.evaluator = Evaluator()

    def test_identifies_false_positives(self):
        """Identifie les faux positifs (predit gagnant mais perdant)."""
        predictions = [
            {"trade_id": 1, "actual": 0, "predicted": 1, "proba": 0.7},
            {"trade_id": 2, "actual": 1, "predicted": 1, "proba": 0.9},
            {"trade_id": 3, "actual": 1, "predicted": 0, "proba": 0.3},
        ]
        result = self.evaluator.error_analysis(predictions)
        assert len(result["false_positives"]) == 1
        assert result["false_positives"][0]["trade_id"] == 1

    def test_identifies_false_negatives(self):
        """Identifie les faux negatifs (predit perdant mais gagnant)."""
        predictions = [
            {"trade_id": 1, "actual": 0, "predicted": 1, "proba": 0.7},
            {"trade_id": 2, "actual": 1, "predicted": 0, "proba": 0.3},
        ]
        result = self.evaluator.error_analysis(predictions)
        assert len(result["false_negatives"]) == 1
        assert result["false_negatives"][0]["trade_id"] == 2


class TestCompareBaseline:
    """Tests de comparaison avec le baseline."""

    def setup_method(self):
        self.evaluator = Evaluator()

    def test_compare_returns_dict(self):
        """compare_to_baseline retourne un dict."""
        y_true = np.array([1, 1, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 1])
        result = self.evaluator.compare_to_baseline(y_true, y_pred)
        assert "model_accuracy" in result
        assert "baseline_accuracy" in result
        assert "improvement" in result

    def test_improvement_calculation(self):
        """Le calcul de l'amelioration est correct."""
        y_true = np.array([1, 1, 1, 1, 0])  # 80% positifs
        y_pred = np.array([1, 1, 1, 0, 0])  # 80% accuracy
        result = self.evaluator.compare_to_baseline(y_true, y_pred)
        assert result["baseline_accuracy"] == 0.8  # Toujours predire 1
        assert result["model_accuracy"] == 0.8
