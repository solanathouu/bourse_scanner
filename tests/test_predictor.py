"""Tests pour le Predictor — scoring temps reel."""

import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.core.database import Database
from src.model.predictor import Predictor
from src.model.trainer import Trainer


def _seed_db_with_prices(db: Database, ticker: str = "SAN.PA"):
    """Seed la base avec des prix synthetiques."""
    np.random.seed(42)
    dates = pd.bdate_range("2025-05-01", "2025-09-15")
    n = len(dates)
    t = np.linspace(0, 6 * np.pi, n)
    close = 97 + 5 * np.sin(t) + np.random.normal(0, 0.5, n)

    prices = []
    for i, d in enumerate(dates):
        prices.append({
            "ticker": ticker,
            "date": d.strftime("%Y-%m-%d"),
            "open": round(close[i] - 0.5, 4),
            "high": round(close[i] + 1.0, 4),
            "low": round(close[i] - 1.0, 4),
            "close": round(close[i], 4),
            "volume": int(100000 + np.random.randint(-20000, 20000)),
        })
    db.insert_prices_batch(prices)
    return prices


def _create_mock_trainer():
    """Cree un Trainer avec un modele mock."""
    trainer = Trainer()
    trainer.model = MagicMock()
    trainer.model.predict_proba = MagicMock(
        return_value=np.array([[0.2, 0.8]])
    )
    trainer.feature_names = [
        "rsi_14", "macd_histogram", "bollinger_position",
        "range_position_10", "range_position_20",
        "range_amplitude_10", "range_amplitude_20",
        "volume_ratio_20", "atr_14_pct",
        "variation_1j", "variation_5j",
        "distance_sma20", "distance_sma50",
        "catalyst_type", "catalyst_confidence", "news_sentiment",
        "has_clear_catalyst", "buy_reason_length",
        "pe_ratio", "pb_ratio", "target_upside_pct",
        "analyst_count", "days_to_earnings", "recommendation_score",
        "day_of_week", "nb_previous_trades",
        "previous_win_rate", "days_since_last_trade",
    ]
    return trainer


class TestScoreTicker:
    """Tests pour score_ticker."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        _seed_db_with_prices(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def _make_predictor(self):
        """Cree un Predictor avec un trainer mock."""
        predictor = Predictor.__new__(Predictor)
        predictor.db = self.db
        from src.analysis.feature_engine import FeatureEngine
        predictor.engine = FeatureEngine(self.db)
        predictor.trainer = _create_mock_trainer()
        return predictor

    def test_score_ticker_returns_dict(self):
        """score_ticker retourne un dict signal."""
        predictor = self._make_predictor()
        result = predictor.score_ticker("SAN.PA", 98.0, date="2025-08-15")
        assert isinstance(result, dict)
        assert "score" in result
        assert "ticker" in result

    def test_score_ticker_has_score(self):
        """Le signal contient un score entre 0 et 1."""
        predictor = self._make_predictor()
        result = predictor.score_ticker("SAN.PA", 98.0, date="2025-08-15")
        assert 0 <= result["score"] <= 1

    def test_score_ticker_has_catalyst_type(self):
        """Le signal contient le type de catalyseur."""
        predictor = self._make_predictor()
        result = predictor.score_ticker("SAN.PA", 98.0, date="2025-08-15")
        assert "catalyst_type" in result

    def test_score_ticker_has_technical_summary(self):
        """Le signal contient un resume technique."""
        predictor = self._make_predictor()
        result = predictor.score_ticker("SAN.PA", 98.0, date="2025-08-15")
        assert "technical_summary" in result
        assert isinstance(result["technical_summary"], str)

    def test_score_ticker_no_data_returns_none(self):
        """Retourne None si pas de donnees pour le ticker."""
        predictor = self._make_predictor()
        result = predictor.score_ticker("UNKNOWN.PA", 50.0, date="2025-08-15")
        assert result is None

    def test_score_ticker_features_json(self):
        """Le signal contient features_json serialise."""
        predictor = self._make_predictor()
        result = predictor.score_ticker("SAN.PA", 98.0, date="2025-08-15")
        assert "features_json" in result
        import json
        parsed = json.loads(result["features_json"])
        assert "rsi_14" in parsed


class TestScoreWatchlist:
    """Tests pour score_watchlist."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        _seed_db_with_prices(self.db, "SAN.PA")
        _seed_db_with_prices(self.db, "DG.PA")

    def teardown_method(self):
        self.temp_dir.cleanup()

    def _make_predictor(self):
        predictor = Predictor.__new__(Predictor)
        predictor.db = self.db
        from src.analysis.feature_engine import FeatureEngine
        predictor.engine = FeatureEngine(self.db)
        predictor.trainer = _create_mock_trainer()
        return predictor

    def test_watchlist_skips_etf(self):
        """Les ETFs sont ignores dans le scoring."""
        predictor = self._make_predictor()
        watchlist = [
            {"name": "SANOFI", "ticker": "SAN.PA", "etf": False},
            {"name": "AMUNDI ETF", "ticker": "CW8.PA", "etf": True},
        ]
        results = predictor.score_watchlist(watchlist, date="2025-08-15")
        tickers = [r["ticker"] for r in results]
        assert "SAN.PA" in tickers
        assert "CW8.PA" not in tickers

    def test_watchlist_sorted_by_score(self):
        """Les signaux sont tries par score decroissant."""
        predictor = self._make_predictor()
        watchlist = [
            {"name": "SANOFI", "ticker": "SAN.PA", "etf": False},
            {"name": "VINCI", "ticker": "DG.PA", "etf": False},
        ]
        results = predictor.score_watchlist(watchlist, date="2025-08-15")
        assert len(results) >= 1
        if len(results) >= 2:
            assert results[0]["score"] >= results[1]["score"]

    def test_watchlist_skips_missing_price(self):
        """Les tickers sans prix sont ignores."""
        predictor = self._make_predictor()
        watchlist = [
            {"name": "SANOFI", "ticker": "SAN.PA", "etf": False},
            {"name": "MYSTERY", "ticker": "MYSTERY.PA", "etf": False},
        ]
        results = predictor.score_watchlist(watchlist, date="2025-08-15")
        tickers = [r["ticker"] for r in results]
        assert "MYSTERY.PA" not in tickers

    def test_watchlist_adds_name(self):
        """Chaque signal dans la watchlist a le nom de l'action."""
        predictor = self._make_predictor()
        watchlist = [
            {"name": "SANOFI", "ticker": "SAN.PA", "etf": False},
        ]
        results = predictor.score_watchlist(watchlist, date="2025-08-15")
        assert results[0]["name"] == "SANOFI"
