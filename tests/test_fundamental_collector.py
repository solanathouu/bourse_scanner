"""Tests pour la collecte de donnees fondamentales yfinance."""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from src.core.database import Database
from src.data_collection.fundamental_collector import (
    FundamentalCollector, RECOMMENDATION_MAP,
)


class TestFundamentalCollector:
    """Tests de collecte fondamentale (yfinance mocke)."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.collector = FundamentalCollector(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def _make_info(self, **overrides):
        """Cree un dict info yfinance complet."""
        base = {
            "regularMarketPrice": 100.0,
            "trailingPE": 15.2,
            "priceToBook": 2.1,
            "marketCap": 120_000_000_000,
            "dividendYield": 0.035,
            "targetMeanPrice": 115.0,
            "numberOfAnalystOpinions": 28,
            "recommendationKey": "buy",
        }
        base.update(overrides)
        return base

    def test_extract_fundamentals_complete(self):
        """Extrait tous les champs d'un info complet."""
        info = self._make_info()
        result = self.collector._extract_fundamentals(info, "SAN.PA")

        assert result["ticker"] == "SAN.PA"
        assert result["pe_ratio"] == 15.2
        assert result["pb_ratio"] == 2.1
        assert result["market_cap"] == 120_000_000_000
        assert result["dividend_yield"] == 3.5  # 0.035 * 100
        assert result["target_price"] == 115.0
        assert result["analyst_count"] == 28
        assert result["recommendation"] == "buy"

    def test_extract_fundamentals_partial(self):
        """Gere les champs manquants (small caps)."""
        info = {
            "regularMarketPrice": 5.0,
            "trailingPE": None,
            "priceToBook": None,
        }
        result = self.collector._extract_fundamentals(info, "MEMS.PA")

        assert result["pe_ratio"] is None
        assert result["pb_ratio"] is None
        assert result["target_price"] is None
        assert result["analyst_count"] is None

    def test_extract_fundamentals_etf(self):
        """Les ETF n'ont pas de PER ni consensus."""
        info = {
            "regularMarketPrice": 500.0,
        }
        result = self.collector._extract_fundamentals(info, "CW8.PA")

        assert result["pe_ratio"] is None
        assert result["recommendation"] is None

    @patch("src.data_collection.fundamental_collector.yf.Ticker")
    def test_collect_for_ticker_inserts_in_db(self, mock_ticker_cls):
        """collect_for_ticker insere les donnees en base."""
        mock_ticker = MagicMock()
        mock_ticker.get_info.return_value = self._make_info()
        mock_ticker.get_calendar.return_value = {"Earnings Date": ["2026-04-25"]}
        mock_ticker_cls.return_value = mock_ticker

        result = self.collector.collect_for_ticker("SAN.PA")

        assert result is not None
        assert result["pe_ratio"] == 15.2
        assert self.db.count_fundamentals() == 1

        db_result = self.db.get_fundamentals("SAN.PA")
        assert len(db_result) == 1
        assert db_result[0]["pe_ratio"] == 15.2
        assert db_result[0]["earnings_date"] == "2026-04-25"

    @patch("src.data_collection.fundamental_collector.yf.Ticker")
    def test_collect_for_ticker_empty_info(self, mock_ticker_cls):
        """Retourne None si yfinance ne renvoie rien."""
        mock_ticker = MagicMock()
        mock_ticker.get_info.return_value = {}
        mock_ticker_cls.return_value = mock_ticker

        result = self.collector.collect_for_ticker("DELISTED.PA")
        assert result is None
        assert self.db.count_fundamentals() == 0

    @patch("src.data_collection.fundamental_collector.yf.Ticker")
    def test_collect_all_stats(self, mock_ticker_cls):
        """collect_all retourne les stats correctes."""
        mock_ticker = MagicMock()
        mock_ticker.get_info.return_value = self._make_info()
        mock_ticker.get_calendar.return_value = {}
        mock_ticker_cls.return_value = mock_ticker

        result = self.collector.collect_all()

        assert result["total"] > 0
        assert result["collected"] > 0
        assert result["errors"] >= 0

    def test_recommendation_map_values(self):
        """Verifie les valeurs du mapping recommendation."""
        assert RECOMMENDATION_MAP["strongBuy"] == 5
        assert RECOMMENDATION_MAP["buy"] == 4
        assert RECOMMENDATION_MAP["hold"] == 3
        assert RECOMMENDATION_MAP["sell"] == 2
        assert RECOMMENDATION_MAP["strongSell"] == 1
