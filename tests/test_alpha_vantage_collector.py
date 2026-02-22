"""Tests pour la collecte de news via Alpha Vantage."""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from src.core.database import Database
from src.data_collection.alpha_vantage_collector import AlphaVantageCollector


class TestAlphaVantageCollector:
    """Tests de collecte Alpha Vantage (API mockee)."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.collector = AlphaVantageCollector(self.db)
        self.collector.api_key = "test_key"

    def teardown_method(self):
        self.temp_dir.cleanup()

    def _make_mock_response(self):
        """Cree une reponse similaire a Alpha Vantage."""
        return {
            "feed": [
                {
                    "title": "Sanofi Q2 results beat expectations",
                    "url": "https://example.com/av-sanofi-1",
                    "time_published": "20250615T083000",
                    "source": "Reuters",
                    "summary": "Strong pharma results...",
                    "overall_sentiment_score": 0.425,
                    "ticker_sentiment": [
                        {"ticker": "SNY", "relevance_score": "0.95",
                         "ticker_sentiment_score": "0.38"}
                    ],
                },
                {
                    "title": "Sanofi pipeline update",
                    "url": "https://example.com/av-sanofi-2",
                    "time_published": "20250616T100000",
                    "source": "Bloomberg",
                    "summary": "New drug approval...",
                    "overall_sentiment_score": 0.612,
                    "ticker_sentiment": [],
                },
            ]
        }

    @patch("src.data_collection.alpha_vantage_collector.requests.get")
    def test_collect_for_action(self, mock_get):
        """Collecte les news pour une action et insere en base."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._make_mock_response()
        mock_get.return_value = mock_resp

        count = self.collector.collect_for_action(
            "SANOFI", "SAN.PA", "20250601T0000", "20250620T2359"
        )

        assert count == 2
        news = self.db.get_news("SAN.PA")
        assert len(news) == 2
        assert news[0]["sentiment"] == 0.425
        assert news[0]["source_api"] == "alpha_vantage"

    @patch("src.data_collection.alpha_vantage_collector.requests.get")
    def test_collect_empty_result(self, mock_get):
        """Retourne 0 si Alpha Vantage ne retourne rien."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"feed": []}
        mock_get.return_value = mock_resp

        count = self.collector.collect_for_action(
            "INCONNU", "INC.PA", "20250601T0000", "20250620T2359"
        )
        assert count == 0

    @patch("src.data_collection.alpha_vantage_collector.requests.get")
    def test_collect_handles_api_error(self, mock_get):
        """Gere les erreurs API proprement."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"Information": "Rate limit exceeded"}
        mock_get.return_value = mock_resp

        count = self.collector.collect_for_action(
            "SANOFI", "SAN.PA", "20250601T0000", "20250620T2359"
        )
        assert count == 0

    @patch("src.data_collection.alpha_vantage_collector.requests.get")
    def test_parse_date_format(self, mock_get):
        """Verifie que le format de date Alpha Vantage est bien parse."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._make_mock_response()
        mock_get.return_value = mock_resp

        self.collector.collect_for_action(
            "SANOFI", "SAN.PA", "20250601T0000", "20250620T2359"
        )
        news = self.db.get_news("SAN.PA")
        assert news[0]["published_at"] == "2025-06-15 08:30:00"
