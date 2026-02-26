"""Tests pour la collecte de prix historiques."""

import os
import tempfile
from datetime import datetime
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.core.database import Database
from src.data_collection.price_collector import PriceCollector


class TestPriceCollector:
    """Tests de collecte de prix (yfinance mocke)."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.collector = PriceCollector(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def _make_mock_df(self):
        """Cree un DataFrame similaire a yfinance.download()."""
        dates = pd.date_range("2025-06-01", periods=3, freq="B")
        return pd.DataFrame({
            "Open": [95.0, 95.5, 96.0],
            "High": [96.0, 96.5, 97.0],
            "Low": [94.5, 95.0, 95.5],
            "Close": [95.5, 96.0, 96.5],
            "Volume": [100000, 120000, 110000],
        }, index=dates)

    @patch("src.data_collection.price_collector.yf.download")
    def test_collect_prices_for_ticker(self, mock_download):
        """Collecte les prix pour un ticker et les insere en base."""
        mock_download.return_value = self._make_mock_df()

        count = self.collector.collect_for_ticker(
            "SAN.PA", "2025-06-01", "2025-06-05"
        )

        assert count == 3
        prices = self.db.get_prices("SAN.PA")
        assert len(prices) == 3
        assert prices[0]["close"] == 95.5

    @patch("src.data_collection.price_collector.yf.download")
    def test_collect_prices_empty_result(self, mock_download):
        """Retourne 0 si yfinance ne retourne rien."""
        mock_download.return_value = pd.DataFrame()

        count = self.collector.collect_for_ticker(
            "INCONNU.PA", "2025-06-01", "2025-06-05"
        )

        assert count == 0

    @patch("src.data_collection.price_collector.yf.download")
    def test_collect_deduplication(self, mock_download):
        """Les doublons sont ignores lors d'une 2e collecte."""
        mock_download.return_value = self._make_mock_df()

        self.collector.collect_for_ticker("SAN.PA", "2025-06-01", "2025-06-05")
        self.collector.collect_for_ticker("SAN.PA", "2025-06-01", "2025-06-05")

        prices = self.db.get_prices("SAN.PA")
        assert len(prices) == 3  # pas de doublons

    def test_compute_date_ranges_from_trades(self):
        """Calcule les plages de dates a partir des trades."""
        trades = [
            {"nom_action": "SANOFI", "isin": "FR123",
             "date_achat": "2025-06-15 10:00:00", "date_vente": "2025-06-25 14:00:00",
             "statut": "CLOTURE"},
            {"nom_action": "SANOFI", "isin": "FR123",
             "date_achat": "2025-07-01 09:00:00", "date_vente": "2025-07-10 15:00:00",
             "statut": "CLOTURE"},
        ]
        ranges = self.collector.compute_date_ranges(trades)
        # SANOFI: min_achat=2025-06-15, max_vente=2025-07-10
        # Avec 30j avant: 2025-05-16, fin: 2025-07-10
        assert "SANOFI" in ranges
        assert ranges["SANOFI"]["start"] == "2025-05-16"
        assert ranges["SANOFI"]["end"] == "2025-07-10"

    @patch("src.data_collection.price_collector.yf.download")
    def test_collect_recent(self, mock_download):
        """collect_recent calcule les dates et delegue a collect_for_ticker."""
        mock_download.return_value = self._make_mock_df()

        count = self.collector.collect_recent("SAN.PA", days=5)

        assert count == 3
        mock_download.assert_called_once()
        call_args = mock_download.call_args
        assert call_args[1]["start"] is not None
        assert call_args[1]["end"] is not None

    def test_compute_date_ranges_trade_ouvert(self):
        """Un trade ouvert (sans date_vente) utilise la date du jour."""
        trades = [
            {"nom_action": "ADOCIA", "isin": "FR456",
             "date_achat": "2025-11-04 09:00:00", "date_vente": None,
             "statut": "OUVERT"},
        ]
        ranges = self.collector.compute_date_ranges(trades)
        assert "ADOCIA" in ranges
        # La date de fin doit etre aujourd'hui ou apres
        assert ranges["ADOCIA"]["end"] >= "2026-02-22"
