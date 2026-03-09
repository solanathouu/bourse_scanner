"""Tests pour le collecteur de carnets d'ordres Boursorama."""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from src.core.database import Database
from src.data_collection.orderbook_collector import (
    OrderBookCollector,
    BOURSORAMA_ORDERBOOK_SYMBOLS,
)


class TestGetBoursoramaSymbol:
    """Tests du mapping ticker -> symbole Boursorama."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.collector = OrderBookCollector(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_get_boursorama_symbol_known(self):
        """Mapping correct pour un ticker connu."""
        assert self.collector._get_boursorama_symbol("DBV.PA") == "1rPDBV"
        assert self.collector._get_boursorama_symbol("SAN.PA") == "1rPSAN"

    def test_get_boursorama_symbol_unknown(self):
        """Retourne None pour un ticker inconnu."""
        assert self.collector._get_boursorama_symbol("UNKNOWN.PA") is None


class TestParseOrderbook:
    """Tests du parsing du carnet d'ordres."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.collector = OrderBookCollector(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_parse_orderbook_calculates_metrics(self):
        """Calcul correct du spread, ratios et concentration."""
        data = {
            "bids": [
                {"price": 10.0, "quantity": 500, "orders": 3},
                {"price": 9.9, "quantity": 300, "orders": 2},
                {"price": 9.8, "quantity": 200, "orders": 1},
            ],
            "asks": [
                {"price": 10.1, "quantity": 400, "orders": 2},
                {"price": 10.2, "quantity": 200, "orders": 1},
            ],
        }
        result = self.collector._parse_orderbook(data)

        assert result["best_bid"] == 10.0
        assert result["best_ask"] == 10.1
        assert result["bid_volume_total"] == 1000
        assert result["ask_volume_total"] == 600
        assert result["bid_orders_total"] == 6
        assert result["ask_orders_total"] == 3
        # spread = (10.1 - 10.0) / 10.0 * 100 = 1.0%
        assert abs(result["spread_pct"] - 1.0) < 0.01
        # ratio = 1000 / 600 = 1.6667
        assert abs(result["bid_ask_volume_ratio"] - 1.6667) < 0.01
        # concentration: top3 = all = 1000/1000 = 1.0
        assert result["bid_depth_concentration"] == 1.0

    def test_parse_orderbook_empty(self):
        """Carnet vide retourne des zeros."""
        data = {"bids": [], "asks": []}
        result = self.collector._parse_orderbook(data)
        assert result["best_bid"] == 0
        assert result["best_ask"] == 0
        assert result["spread_pct"] == 0.0
        assert result["bid_ask_volume_ratio"] == 0.0


class TestFetchOrderbook:
    """Tests du fetch HTTP avec mock."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.collector = OrderBookCollector(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    @patch("src.data_collection.orderbook_collector.requests.get")
    def test_fetch_orderbook_parses_html(self, mock_get):
        """Parse correctement le HTML avec data-ist-orderbook."""
        orderbook_data = {
            "bids": [{"price": 5.0, "quantity": 100, "orders": 1}],
            "asks": [{"price": 5.1, "quantity": 80, "orders": 1}],
        }
        html_content = (
            '<div data-ist-orderbook data-ist-init="'
            + json.dumps(orderbook_data).replace('"', '&quot;')
            + '"></div>'
        )
        mock_response = MagicMock()
        mock_response.text = html_content
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = self.collector._fetch_orderbook("1rPDBV")
        assert result is not None
        assert "bids" in result
        assert result["bids"][0]["price"] == 5.0

    @patch("src.data_collection.orderbook_collector.requests.get")
    def test_fetch_orderbook_api_error(self, mock_get):
        """Erreur HTTP retourne None sans crash."""
        mock_get.side_effect = Exception("Connection error")
        result = self.collector._fetch_orderbook("1rPDBV")
        assert result is None


class TestCollectOrderbook:
    """Tests de la collecte complete avec stockage en DB."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.collector = OrderBookCollector(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    @patch("src.data_collection.orderbook_collector.requests.get")
    def test_collect_orderbook_stores_in_db(self, mock_get):
        """Collecte un carnet et verifie l'insertion en DB."""
        orderbook_data = {
            "bids": [{"price": 10.0, "quantity": 500, "orders": 3}],
            "asks": [{"price": 10.1, "quantity": 400, "orders": 2}],
        }
        html_content = (
            '<div data-ist-orderbook data-ist-init="'
            + json.dumps(orderbook_data).replace('"', '&quot;')
            + '"></div>'
        )
        mock_response = MagicMock()
        mock_response.text = html_content
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = self.collector.collect_orderbook("SAN.PA")
        assert result is not None
        assert result["ticker"] == "SAN.PA"

        # Verifie en DB
        latest = self.db.get_latest_orderbook("SAN.PA")
        assert latest is not None
        assert latest["best_bid"] == 10.0

    @patch.object(OrderBookCollector, "collect_orderbook")
    def test_collect_all_watchlist(self, mock_collect):
        """Itere sur la watchlist et collecte."""
        mock_collect.return_value = {"ticker": "SAN.PA"}
        watchlist = [
            {"ticker": "SAN.PA", "name": "SANOFI"},
            {"ticker": "DBV.PA", "name": "DBV TECHNOLOGIES"},
            {"ticker": "CW8.PA", "name": "ETF", "etf": True},
        ]
        result = self.collector.collect_all_watchlist(watchlist)
        # 2 actions (pas l'ETF)
        assert mock_collect.call_count == 2
        assert result["collected"] == 2
