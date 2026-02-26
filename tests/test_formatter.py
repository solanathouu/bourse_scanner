"""Tests pour le formatage des alertes Telegram."""

import pytest

from src.alerts.formatter import AlertFormatter


class TestAlertFormatter:
    """Tests pour AlertFormatter."""

    def setup_method(self):
        self.formatter = AlertFormatter()

    def test_format_signal_contains_ticker(self):
        """Le message contient le ticker et le nom."""
        signal = {
            "ticker": "SAN.PA", "name": "SANOFI", "score": 0.82,
            "catalyst_type": "EARNINGS", "features": {"rsi_14": 38.2},
            "technical_summary": "RSI 38.2 (survendu)",
        }
        msg = self.formatter.format_signal(signal)
        assert "SAN.PA" in msg
        assert "SANOFI" in msg

    def test_format_signal_contains_score(self):
        """Le message contient le score en pourcentage."""
        signal = {
            "ticker": "SAN.PA", "name": "SANOFI", "score": 0.82,
            "features": {},
        }
        msg = self.formatter.format_signal(signal)
        assert "82%" in msg

    def test_format_signal_contains_catalyst(self):
        """Le message contient le type de catalyseur."""
        signal = {
            "ticker": "SAN.PA", "name": "SANOFI", "score": 0.85,
            "catalyst_type": "EARNINGS",
            "catalyst_news_title": "Sanofi: resultats T2",
            "features": {},
        }
        msg = self.formatter.format_signal(signal)
        assert "EARNINGS" in msg
        assert "Sanofi: resultats T2" in msg

    def test_format_signal_contains_disclaimer(self):
        """Le message contient le disclaimer obligatoire."""
        signal = {
            "ticker": "SAN.PA", "name": "SANOFI", "score": 0.80,
            "features": {},
        }
        msg = self.formatter.format_signal(signal)
        assert "Aide a la decision" in msg

    def test_format_signal_with_fundamentals(self):
        """Le message affiche les fondamentaux si disponibles."""
        signal = {
            "ticker": "SAN.PA", "name": "SANOFI", "score": 0.82,
            "features": {"pe_ratio": 15.2, "analyst_count": 28,
                         "recommendation_score": 4},
        }
        msg = self.formatter.format_signal(signal)
        assert "PE 15.2" in msg
        assert "28" in msg

    def test_format_daily_summary(self):
        """Le resume quotidien liste les signaux."""
        signals = [
            {"ticker": "SAN.PA", "name": "SANOFI", "score": 0.85,
             "catalyst_type": "EARNINGS"},
            {"ticker": "DG.PA", "name": "VINCI", "score": 0.78,
             "catalyst_type": "CONTRACT"},
        ]
        msg = self.formatter.format_daily_summary(signals)
        assert "2 signal(s)" in msg
        assert "SANOFI" in msg
        assert "VINCI" in msg
        assert "Aide a la decision" in msg
