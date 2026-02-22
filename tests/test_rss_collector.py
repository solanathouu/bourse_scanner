"""Tests pour la collecte de news via flux RSS."""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from src.core.database import Database
from src.data_collection.rss_collector import RSSCollector


class TestRSSCollector:
    """Tests de collecte RSS (feedparser mocke)."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.collector = RSSCollector(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def _make_mock_feed(self, entries):
        """Cree un feed RSS simule."""
        feed = MagicMock()
        feed.bozo = False
        feed.entries = entries
        return feed

    def _make_entry(self, title, link, summary="", published="2026-02-22"):
        """Cree une entree RSS simulee."""
        entry = {
            "title": title,
            "link": link,
            "summary": summary,
            "published": published,
            "published_parsed": (2026, 2, 22, 10, 0, 0, 0, 0, 0),
        }
        return entry

    @patch("src.data_collection.rss_collector.feedparser.parse")
    def test_collect_feed_matches_our_actions(self, mock_parse):
        """Les articles mentionnant nos actions sont collectes."""
        entries = [
            self._make_entry(
                "Sanofi publie ses resultats trimestriels",
                "https://example.com/rss-sanofi-1",
                "Le groupe pharmaceutique Sanofi..."
            ),
            self._make_entry(
                "CAC 40: la bourse de Paris en hausse",
                "https://example.com/rss-cac40",
                "Article generique sur le marche"
            ),
        ]
        mock_parse.return_value = self._make_mock_feed(entries)

        count = self.collector.collect_feed("test_feed", "https://example.com/rss")

        assert count == 1  # Seul l'article Sanofi matche
        news = self.db.get_news("SAN.PA")
        assert len(news) == 1
        assert "Sanofi" in news[0]["title"]
        assert news[0]["source_api"] == "rss_test_feed"

    @patch("src.data_collection.rss_collector.feedparser.parse")
    def test_collect_feed_empty(self, mock_parse):
        """Un flux sans articles pertinents retourne 0."""
        entries = [
            self._make_entry(
                "Apple lance un nouveau produit",
                "https://example.com/rss-apple",
            ),
        ]
        mock_parse.return_value = self._make_mock_feed(entries)

        count = self.collector.collect_feed("test_feed", "https://example.com/rss")
        assert count == 0

    @patch("src.data_collection.rss_collector.feedparser.parse")
    def test_collect_feed_broken_feed(self, mock_parse):
        """Un flux inaccessible retourne 0 sans crash."""
        feed = MagicMock()
        feed.bozo = True
        feed.entries = []
        mock_parse.return_value = feed

        count = self.collector.collect_feed("broken", "https://broken.com/rss")
        assert count == 0

    @patch("src.data_collection.rss_collector.feedparser.parse")
    def test_match_ticker_by_first_word(self, mock_parse):
        """Match par le premier mot significatif du nom d'action."""
        entries = [
            self._make_entry(
                "Schneider annonce un partenariat strategique",
                "https://example.com/rss-schneider",
            ),
        ]
        mock_parse.return_value = self._make_mock_feed(entries)

        count = self.collector.collect_feed("test_feed", "https://example.com/rss")
        assert count == 1
        news = self.db.get_news("SU.PA")
        assert len(news) == 1

    @patch("src.data_collection.rss_collector.feedparser.parse")
    def test_deduplication_across_feeds(self, mock_parse):
        """Les articles avec la meme URL ne sont pas dupliques."""
        entries = [
            self._make_entry(
                "Sanofi news",
                "https://example.com/rss-same-url",
            ),
        ]
        mock_parse.return_value = self._make_mock_feed(entries)

        self.collector.collect_feed("feed1", "https://example.com/rss")
        self.collector.collect_feed("feed2", "https://example.com/rss")

        news = self.db.get_news("SAN.PA")
        assert len(news) == 1
