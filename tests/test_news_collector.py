"""Tests pour la collecte de news historiques."""

import os
import tempfile
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from src.core.database import Database
from src.data_collection.news_collector import NewsCollector


class TestNewsCollector:
    """Tests de collecte de news (GNews mocke)."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.collector = NewsCollector(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def _make_mock_articles(self):
        """Cree des articles similaires a GNews.get_news()."""
        return [
            {
                "title": "Sanofi: resultats T2 solides",
                "description": "Le labo publie des resultats...",
                "url": "https://example.com/sanofi-1",
                "published date": "Mon, 15 Jun 2025 08:30:00 GMT",
                "publisher": {"title": "BFM Bourse"},
            },
            {
                "title": "Sanofi lance un nouveau medicament",
                "description": "Approbation FDA pour...",
                "url": "https://example.com/sanofi-2",
                "published date": "Tue, 16 Jun 2025 10:00:00 GMT",
                "publisher": {"title": "Les Echos"},
            },
        ]

    @patch("src.data_collection.news_collector.GNews")
    def test_collect_news_for_action(self, MockGNews):
        """Collecte les news pour une action et les insere en base."""
        mock_gn = MagicMock()
        mock_gn.get_news.return_value = self._make_mock_articles()
        MockGNews.return_value = mock_gn

        count = self.collector.collect_for_action(
            "SANOFI", "SAN.PA", "2025-06-08", "2025-06-18"
        )

        assert count == 2
        news = self.db.get_news("SAN.PA")
        assert len(news) == 2

    @patch("src.data_collection.news_collector.GNews")
    def test_collect_news_empty(self, MockGNews):
        """Retourne 0 si GNews ne retourne rien."""
        mock_gn = MagicMock()
        mock_gn.get_news.return_value = []
        MockGNews.return_value = mock_gn

        count = self.collector.collect_for_action(
            "INCONNU", "INC.PA", "2025-06-08", "2025-06-18"
        )

        assert count == 0

    @patch("src.data_collection.news_collector.GNews")
    def test_collect_news_deduplication(self, MockGNews):
        """Les doublons (meme URL) sont ignores."""
        mock_gn = MagicMock()
        mock_gn.get_news.return_value = self._make_mock_articles()
        MockGNews.return_value = mock_gn

        self.collector.collect_for_action("SANOFI", "SAN.PA", "2025-06-08", "2025-06-18")
        self.collector.collect_for_action("SANOFI", "SAN.PA", "2025-06-08", "2025-06-18")

        news = self.db.get_news("SAN.PA")
        assert len(news) == 2  # pas de doublons

    def test_compute_news_windows(self):
        """Calcule les fenetres de recherche de news autour des trades."""
        trades = [
            {"nom_action": "SANOFI", "date_achat": "2025-06-15 10:00:00",
             "date_vente": "2025-06-25 14:00:00", "statut": "CLOTURE"},
            {"nom_action": "SANOFI", "date_achat": "2025-07-01 09:00:00",
             "date_vente": "2025-07-10 15:00:00", "statut": "CLOTURE"},
        ]
        windows = self.collector.compute_news_windows(trades)
        # 2 trades SANOFI -> 2 fenetres distinctes
        sanofi_windows = [w for w in windows if w["nom_action"] == "SANOFI"]
        assert len(sanofi_windows) == 2
        # Premiere fenetre: 2025-06-08 a 2025-06-18
        assert sanofi_windows[0]["start"] == "2025-06-08"
        assert sanofi_windows[0]["end"] == "2025-06-18"

    @patch("src.data_collection.news_collector.GNews")
    def test_parse_article_handles_missing_fields(self, MockGNews):
        """Les articles avec des champs manquants sont geres proprement."""
        mock_gn = MagicMock()
        mock_gn.get_news.return_value = [
            {
                "title": "Article sans description",
                "url": "https://example.com/no-desc",
                "published date": "Mon, 15 Jun 2025 08:30:00 GMT",
                "publisher": {"title": "Source"},
            },
        ]
        MockGNews.return_value = mock_gn

        count = self.collector.collect_for_action(
            "SANOFI", "SAN.PA", "2025-06-08", "2025-06-18"
        )

        assert count == 1
        news = self.db.get_news("SAN.PA")
        assert news[0]["description"] is None or news[0]["description"] == ""
