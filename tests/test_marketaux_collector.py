"""Tests pour la collecte de news via Marketaux."""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from src.core.database import Database
from src.data_collection.marketaux_collector import MarketauxCollector


class TestMarketauxCollector:
    """Tests de collecte Marketaux (API mockee)."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.collector = MarketauxCollector(self.db)
        self.collector.api_key = "test_key"

    def teardown_method(self):
        self.temp_dir.cleanup()

    def _make_mock_response(self, page=1):
        """Cree une reponse similaire a Marketaux."""
        return {
            "meta": {"found": 2, "returned": 2, "limit": 50, "page": page},
            "data": [
                {
                    "title": "Sanofi names new CEO",
                    "url": "https://example.com/mx-sanofi-1",
                    "published_at": "2025-06-15T08:30:00.000000Z",
                    "source": "seekingalpha.com",
                    "description": "CEO announcement...",
                    "entities": [
                        {"symbol": "SNY", "sentiment_score": 0.673},
                    ],
                },
                {
                    "title": "Sanofi stock analysis",
                    "url": "https://example.com/mx-sanofi-2",
                    "published_at": "2025-06-16T10:00:00.000000Z",
                    "source": "finance.yahoo.com",
                    "description": "Stock outlook...",
                    "entities": [
                        {"symbol": "SNY", "sentiment_score": 0.432},
                    ],
                },
            ],
        }

    @patch("src.data_collection.marketaux_collector.requests.get")
    def test_collect_for_action(self, mock_get):
        """Collecte les news et insere en base avec sentiment."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._make_mock_response()
        mock_get.return_value = mock_resp

        count = self.collector.collect_for_action(
            "SANOFI", "SAN.PA", "2025-06-01", "2025-06-20"
        )

        assert count == 2
        news = self.db.get_news("SAN.PA")
        assert len(news) == 2
        assert news[0]["sentiment"] == 0.673
        assert news[0]["source_api"] == "marketaux"

    @patch("src.data_collection.marketaux_collector.requests.get")
    def test_collect_empty_result(self, mock_get):
        """Retourne 0 si Marketaux ne retourne rien."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"meta": {"found": 0, "returned": 0}, "data": []}
        mock_get.return_value = mock_resp

        count = self.collector.collect_for_action(
            "INCONNU", "INC.PA", "2025-06-01", "2025-06-20"
        )
        assert count == 0

    @patch("src.data_collection.marketaux_collector.requests.get")
    def test_collect_handles_api_error(self, mock_get):
        """Gere les erreurs API proprement."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": {"message": "Unauthorized"}}
        mock_get.return_value = mock_resp

        count = self.collector.collect_for_action(
            "SANOFI", "SAN.PA", "2025-06-01", "2025-06-20"
        )
        assert count == 0

    @patch("src.data_collection.marketaux_collector.requests.get")
    def test_parse_date_format(self, mock_get):
        """Verifie que le format de date Marketaux est bien parse."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._make_mock_response()
        mock_get.return_value = mock_resp

        self.collector.collect_for_action(
            "SANOFI", "SAN.PA", "2025-06-01", "2025-06-20"
        )
        news = self.db.get_news("SAN.PA")
        assert news[0]["published_at"] == "2025-06-15 08:30:00"

    @patch("src.data_collection.marketaux_collector.requests.get")
    def test_deduplication_by_url(self, mock_get):
        """Les doublons URL sont ignores."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._make_mock_response()
        mock_get.return_value = mock_resp

        self.collector.collect_for_action("SANOFI", "SAN.PA", "2025-06-01", "2025-06-20")
        self.collector.collect_for_action("SANOFI", "SAN.PA", "2025-06-01", "2025-06-20")

        news = self.db.get_news("SAN.PA")
        assert len(news) == 2
