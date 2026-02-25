"""Tests pour la collecte de news via Newsdata.io."""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from src.core.database import Database
from src.data_collection.newsdata_collector import NewsdataCollector


class TestNewsdataCollector:
    """Tests de collecte Newsdata.io (requests mocke)."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()

    def teardown_method(self):
        self.temp_dir.cleanup()

    def _make_collector(self):
        """Cree un collector avec une fausse API key."""
        with patch.dict(os.environ, {"NEWSDATA_API_KEY": "test_key_123"}):
            return NewsdataCollector(self.db)

    def _make_api_response(self, articles):
        """Cree une reponse API simulee."""
        return {
            "status": "success",
            "totalResults": len(articles),
            "results": articles,
        }

    def test_parse_article_valid(self):
        """Parse un article Newsdata.io valide."""
        collector = self._make_collector()
        article = {
            "title": "Sanofi publie des resultats record",
            "link": "https://newsdata.io/sanofi-1",
            "pubDate": "2026-02-20 10:30:00",
            "description": "Le groupe pharmaceutique annonce...",
            "source_id": "lefigaro",
            "sentiment": {"positive": 0.8, "negative": 0.1, "neutral": 0.1},
        }
        result = collector._parse_article(article, "SAN.PA")

        assert result is not None
        assert result["ticker"] == "SAN.PA"
        assert result["title"] == "Sanofi publie des resultats record"
        assert result["source_api"] == "newsdata"
        assert result["sentiment"] == 0.7  # 0.8 - 0.1

    def test_parse_article_no_sentiment(self):
        """Parse un article sans sentiment."""
        collector = self._make_collector()
        article = {
            "title": "Sanofi news",
            "link": "https://newsdata.io/2",
            "pubDate": "2026-02-20",
            "description": "desc",
        }
        result = collector._parse_article(article, "SAN.PA")

        assert result is not None
        assert result["sentiment"] is None

    def test_parse_article_missing_url(self):
        """Retourne None si pas d'URL."""
        collector = self._make_collector()
        result = collector._parse_article({"title": "test"}, "SAN.PA")
        assert result is None

    @patch("src.data_collection.newsdata_collector.requests.get")
    def test_collect_for_action(self, mock_get):
        """Collecte les articles pour une action."""
        collector = self._make_collector()

        mock_response = MagicMock()
        mock_response.json.return_value = self._make_api_response([
            {"title": "Sanofi news", "link": "https://newsdata.io/3",
             "pubDate": "2026-02-20", "description": "desc", "source_id": "bfm"},
        ])
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        count = collector.collect_for_action("SANOFI", "SAN.PA")
        assert count == 1
        assert self.db.count_news() == 1

    @patch("src.data_collection.newsdata_collector.requests.get")
    def test_collect_for_action_empty(self, mock_get):
        """Retourne 0 si pas de resultats."""
        collector = self._make_collector()

        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success", "results": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        count = collector.collect_for_action("SANOFI", "SAN.PA")
        assert count == 0

    @patch("src.data_collection.newsdata_collector.requests.get")
    def test_collect_for_action_api_error(self, mock_get):
        """Gere les erreurs API sans crash."""
        collector = self._make_collector()
        mock_get.side_effect = Exception("Connection timeout")

        count = collector.collect_for_action("SANOFI", "SAN.PA")
        assert count == 0

    def test_missing_api_key_raises(self):
        """Leve une erreur si la cle API est absente."""
        with patch.dict(os.environ, {}, clear=True):
            # S'assurer que NEWSDATA_API_KEY n'est pas dans l'env
            os.environ.pop("NEWSDATA_API_KEY", None)
            with pytest.raises(ValueError, match="NEWSDATA_API_KEY"):
                NewsdataCollector(self.db)
