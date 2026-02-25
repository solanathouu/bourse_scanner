"""Tests pour le scraper Boursorama (prix tickers delistes)."""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from src.core.database import Database
from src.data_collection.scrapers.boursorama_scraper import (
    BoursoramaPriceScraper, BOURSORAMA_SYMBOLS, DELISTED_TICKERS,
)


class TestBoursoramaPriceScraper:
    """Tests du scraper Boursorama (requests mocke)."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.scraper = BoursoramaPriceScraper(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_get_boursorama_symbol(self):
        """Mappe les tickers Yahoo vers Boursorama."""
        assert self.scraper._get_boursorama_symbol("2CRSI.PA") == "1rP2CRSI"
        assert self.scraper._get_boursorama_symbol("ALTBG.PA") == "1rPALTBG"
        assert self.scraper._get_boursorama_symbol("UNKNOWN.PA") is None

    @patch("src.data_collection.scrapers.boursorama_scraper.requests.get")
    def test_fetch_prices_parses_response(self, mock_get):
        """Parse correctement la reponse JSON Boursorama."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "d": {
                "QuoteTab": [
                    {"d": "/Date(1706745600000)/", "o": 5.0, "h": 5.5,
                     "l": 4.8, "c": 5.2, "v": 10000},
                    {"d": "/Date(1706832000000)/", "o": 5.2, "h": 5.6,
                     "l": 5.0, "c": 5.4, "v": 12000},
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        prices = self.scraper._fetch_prices("1rP2CRSI")

        assert len(prices) == 2
        assert prices[0]["close"] == 5.2
        assert prices[0]["date"]  # Doit avoir une date valide

    @patch("src.data_collection.scrapers.boursorama_scraper.requests.get")
    def test_collect_for_ticker_inserts_in_db(self, mock_get):
        """collect_for_ticker insere les prix en base."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "d": {
                "QuoteTab": [
                    {"d": "/Date(1706745600000)/", "o": 5.0, "h": 5.5,
                     "l": 4.8, "c": 5.2, "v": 10000},
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        count = self.scraper.collect_for_ticker("2CRSI.PA")

        assert count == 1
        assert self.db.count_prices() == 1
        prices = self.db.get_prices("2CRSI.PA")
        assert len(prices) == 1
        assert prices[0]["close"] == 5.2

    def test_collect_for_ticker_unknown_symbol(self):
        """Retourne 0 pour un ticker sans mapping Boursorama."""
        count = self.scraper.collect_for_ticker("UNKNOWN.PA")
        assert count == 0

    @patch("src.data_collection.scrapers.boursorama_scraper.requests.get")
    def test_fetch_prices_api_error(self, mock_get):
        """Gere les erreurs API sans crash."""
        mock_get.side_effect = Exception("Connection refused")

        prices = self.scraper._fetch_prices("1rP2CRSI")
        assert prices == []

    def test_delisted_tickers_list(self):
        """Verifie que les tickers delistes sont configures."""
        assert "2CRSI.PA" in DELISTED_TICKERS
        assert "ALTBG.PA" in DELISTED_TICKERS

    def test_boursorama_symbols_match_delisted(self):
        """Chaque ticker deliste a un symbole Boursorama."""
        for ticker in DELISTED_TICKERS:
            assert ticker in BOURSORAMA_SYMBOLS
