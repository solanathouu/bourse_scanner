"""Tests pour le filtrage des signaux."""

import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from src.core.database import Database
from src.alerts.signal_filter import SignalFilter


class TestSignalFilter:
    """Tests pour SignalFilter."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.config = {
            "threshold": 0.75,
            "cooldown_hours": 24,
            "market_open": "09:00",
            "market_close": "17:30",
        }
        self.filter = SignalFilter(self.db, self.config)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_passes_threshold_high_score(self):
        """Un signal avec score >= seuil passe."""
        signal = {"ticker": "SAN.PA", "score": 0.85}
        assert self.filter._passes_threshold(signal) is True

    def test_rejects_threshold_low_score(self):
        """Un signal avec score < seuil est rejete."""
        signal = {"ticker": "SAN.PA", "score": 0.60}
        assert self.filter._passes_threshold(signal) is False

    def test_passes_cooldown_no_previous(self):
        """Un ticker sans signal precedent passe le cooldown."""
        signal = {"ticker": "SAN.PA", "score": 0.85}
        assert self.filter._passes_cooldown(signal) is True

    def test_rejects_cooldown_recent_signal(self):
        """Un ticker avec signal recent est en cooldown."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.db.insert_signal({
            "ticker": "SAN.PA", "date": "2026-02-26",
            "score": 0.80, "sent_at": now,
        })
        signal = {"ticker": "SAN.PA", "score": 0.85}
        assert self.filter._passes_cooldown(signal) is False

    def test_passes_cooldown_old_signal(self):
        """Un ticker avec signal ancien (>24h) passe le cooldown."""
        old = (datetime.now() - timedelta(hours=25)).strftime("%Y-%m-%d %H:%M:%S")
        self.db.insert_signal({
            "ticker": "SAN.PA", "date": "2026-02-25",
            "score": 0.80, "sent_at": old,
        })
        signal = {"ticker": "SAN.PA", "score": 0.85}
        assert self.filter._passes_cooldown(signal) is True

    def test_filter_signals_combined(self):
        """filter_signals applique seuil + cooldown."""
        signals = [
            {"ticker": "SAN.PA", "score": 0.85},  # passe
            {"ticker": "AI.PA", "score": 0.60},    # score trop bas
            {"ticker": "DG.PA", "score": 0.90},    # passe
        ]
        result = self.filter.filter_signals(signals)
        tickers = [s["ticker"] for s in result]
        assert "SAN.PA" in tickers
        assert "DG.PA" in tickers
        assert "AI.PA" not in tickers

    def test_record_signal(self):
        """record_signal insere le signal en BDD."""
        signal = {
            "ticker": "SAN.PA", "date": "2026-02-26",
            "score": 0.82, "catalyst_type": "EARNINGS",
        }
        self.filter.record_signal(signal)
        assert self.db.count_signals() == 1
        latest = self.db.get_latest_signal("SAN.PA")
        assert latest["score"] == 0.82
        assert latest["sent_at"] is not None

    def test_is_market_hours_weekday(self):
        """Verifie la detection des horaires de marche."""
        # On ne peut pas tester le resultat exact (depend de l'heure)
        # mais on verifie que ca ne plante pas
        result = self.filter.is_market_hours()
        assert isinstance(result, bool)


class TestAdaptiveFiltering:
    """Tests du filtrage adaptatif avec regles dynamiques."""

    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.tmp.name, "test.db"))
        self.db.init_db()
        self.config = {"threshold": 0.75, "cooldown_hours": 24}
        self.sf = SignalFilter(self.db, self.config)

    def teardown_method(self):
        self.tmp.cleanup()

    def test_exclude_catalyst_rule(self):
        """Les signaux avec catalyst exclu sont filtres."""
        self.db.insert_filter_rule({
            "rule_type": "EXCLUDE_CATALYST",
            "rule_json": '{"catalyst_type": "TECHNICAL"}',
            "win_rate": 0.20, "sample_size": 10,
            "created_at": "2026-03-07", "active": 1,
        })
        signals = [
            {"ticker": "SAN.PA", "score": 0.85, "catalyst_type": "EARNINGS"},
            {"ticker": "DBV.PA", "score": 0.82, "catalyst_type": "TECHNICAL"},
        ]
        filtered = self.sf.filter_signals(signals)
        assert len(filtered) == 1
        assert filtered[0]["ticker"] == "SAN.PA"

    def test_max_signals_per_day(self):
        """La limite par jour tronque les signaux excedentaires."""
        self.db.insert_filter_rule({
            "rule_type": "MAX_SIGNALS_PER_DAY",
            "rule_json": '{"max": 2}',
            "win_rate": None, "sample_size": None,
            "created_at": "2026-03-07", "active": 1,
        })
        signals = [
            {"ticker": "SAN.PA", "score": 0.90, "catalyst_type": "EARNINGS"},
            {"ticker": "DBV.PA", "score": 0.88, "catalyst_type": "UPGRADE"},
            {"ticker": "MAU.PA", "score": 0.85, "catalyst_type": "EARNINGS"},
        ]
        filtered = self.sf.filter_signals(signals)
        assert len(filtered) == 2

    def test_adaptive_threshold(self):
        """Le seuil adaptatif remplace le seuil par defaut."""
        self.db.insert_filter_rule({
            "rule_type": "ADAPTIVE_THRESHOLD",
            "rule_json": '{"threshold": 0.85}',
            "win_rate": None, "sample_size": None,
            "created_at": "2026-03-07", "active": 1,
        })
        signals = [
            {"ticker": "SAN.PA", "score": 0.90, "catalyst_type": "EARNINGS"},
            {"ticker": "DBV.PA", "score": 0.82, "catalyst_type": "UPGRADE"},
        ]
        filtered = self.sf.filter_signals(signals)
        assert len(filtered) == 1
        assert filtered[0]["ticker"] == "SAN.PA"

    def test_no_rules_uses_default_threshold(self):
        """Sans regles adaptatives, le seuil par defaut est utilise."""
        signals = [
            {"ticker": "SAN.PA", "score": 0.80, "catalyst_type": "EARNINGS"},
            {"ticker": "DBV.PA", "score": 0.70, "catalyst_type": "UPGRADE"},
        ]
        filtered = self.sf.filter_signals(signals)
        assert len(filtered) == 1
        assert filtered[0]["ticker"] == "SAN.PA"

    def test_record_signal_stores_price(self):
        """record_signal enregistre le signal_price en BDD."""
        signal = {
            "ticker": "SAN.PA", "date": "2026-03-07",
            "score": 0.85, "current_price": 95.50,
            "catalyst_type": "EARNINGS", "features_json": "{}",
        }
        self.sf.record_signal(signal)
        stored = self.db.get_latest_signal("SAN.PA")
        assert stored["signal_price"] == 95.50
