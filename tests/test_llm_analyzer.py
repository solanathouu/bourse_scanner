"""Tests pour l'analyseur LLM des trades."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.database import Database
from src.analysis.llm_analyzer import LLMAnalyzer, LLM_ANALYSIS_PROMPT


def _seed_test_db(db: Database):
    """Peuple une base de test."""
    db.insert_trades_batch([
        {
            "isin": "FR0000120578", "nom_action": "SANOFI",
            "date_achat": "2025-07-10", "date_vente": "2025-07-20",
            "prix_achat": 95.0, "prix_vente": 100.0, "quantite": 10,
            "rendement_brut_pct": 5.26, "rendement_net_pct": 5.0,
            "duree_jours": 10, "frais_totaux": 2.5, "statut": "CLOTURE",
        },
    ])
    # Prix
    np.random.seed(42)
    dates = pd.bdate_range("2025-05-01", "2025-09-15")
    close = 97 + 5 * np.sin(np.linspace(0, 6 * np.pi, len(dates)))
    for i, d in enumerate(dates):
        db.insert_price({
            "ticker": "SAN.PA", "date": d.strftime("%Y-%m-%d"),
            "open": round(close[i] - 0.5, 2), "high": round(close[i] + 1, 2),
            "low": round(close[i] - 1, 2), "close": round(close[i], 2),
            "volume": 100000,
        })
    # News
    db.insert_news_batch([
        {
            "ticker": "SAN.PA", "title": "Sanofi: resultats T2 au-dessus des attentes",
            "source": "Reuters", "url": "https://ex.com/san1",
            "published_at": "2025-07-09", "description": "CA en hausse de 8%",
            "sentiment": 0.6, "source_api": "alpha_vantage",
        },
        {
            "ticker": "SAN.PA", "title": "Bourse: le CAC 40 en hausse",
            "source": "BFM", "url": "https://ex.com/san2",
            "published_at": "2025-07-10", "description": "Marche positif",
            "sentiment": 0.3, "source_api": "gnews",
        },
    ])


class TestBuildPrompt:
    """Tests de construction du prompt LLM."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.temp_dir.name, "test.db"))
        self.db.init_db()
        _seed_test_db(self.db)
        self.analyzer = LLMAnalyzer(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_build_prompt_contains_trade_info(self):
        """Le prompt contient les infos du trade."""
        trades = self.db.get_all_trades()
        prompt = self.analyzer.build_prompt(trades[0])
        assert "SANOFI" in prompt
        assert "2025-07-10" in prompt
        assert "95.0" in prompt or "95.00" in prompt

    def test_build_prompt_contains_news(self):
        """Le prompt contient les news de la fenetre."""
        trades = self.db.get_all_trades()
        prompt = self.analyzer.build_prompt(trades[0])
        assert "resultats T2" in prompt

    def test_build_prompt_contains_technical_context(self):
        """Le prompt contient les indicateurs techniques."""
        trades = self.db.get_all_trades()
        prompt = self.analyzer.build_prompt(trades[0])
        assert "RSI" in prompt or "rsi" in prompt


class TestParseResponse:
    """Tests de parsing de la reponse LLM."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.temp_dir.name, "test.db"))
        self.db.init_db()
        self.analyzer = LLMAnalyzer(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_parse_valid_response(self):
        """Parse une reponse JSON valide du LLM."""
        response = json.dumps({
            "primary_news_index": 1,
            "catalyst_type": "EARNINGS",
            "catalyst_confidence": 0.85,
            "catalyst_summary": "Nicolas a achete car resultats T2 solides",
            "news_sentiment": 0.7,
            "buy_reason": "Resultats T2 au-dessus des attentes",
            "sell_reason": "Objectif atteint",
            "trade_quality": "BON",
        })
        news_list = [{"id": 42}, {"id": 43}]
        result = self.analyzer.parse_response(response, trade_id=1,
                                               news_list=news_list)
        assert result["trade_id"] == 1
        assert result["catalyst_type"] == "EARNINGS"
        assert result["primary_news_id"] == 42  # index 1 -> news_list[0]
        assert result["catalyst_confidence"] == 0.85

    def test_parse_no_catalyst_found(self):
        """Parse quand le LLM ne trouve pas de catalyseur."""
        response = json.dumps({
            "primary_news_index": 0,
            "catalyst_type": "TECHNICAL",
            "catalyst_confidence": 0.6,
            "catalyst_summary": "Pas de news declencheuse identifiee",
            "news_sentiment": 0.0,
            "buy_reason": "Signal technique pur",
            "sell_reason": "Stop loss",
            "trade_quality": "MOYEN",
        })
        result = self.analyzer.parse_response(response, trade_id=5,
                                               news_list=[])
        assert result["trade_id"] == 5
        assert result["catalyst_type"] == "TECHNICAL"
        assert result["primary_news_id"] is None

    def test_parse_markdown_wrapped_json(self):
        """Parse un JSON enveloppe dans des backticks markdown."""
        inner = json.dumps({
            "primary_news_index": 1,
            "catalyst_type": "EARNINGS",
            "catalyst_confidence": 0.85,
            "catalyst_summary": "Nicolas a achete car resultats T2 solides",
            "news_sentiment": 0.7,
            "buy_reason": "Resultats T2",
            "sell_reason": "Objectif atteint",
            "trade_quality": "BON",
        })
        response = f"```json\n{inner}\n```"
        result = self.analyzer.parse_response(response, trade_id=1,
                                               news_list=[{"id": 42}])
        assert result["catalyst_type"] == "EARNINGS"
        assert result["catalyst_confidence"] == 0.85

    def test_parse_invalid_json_returns_fallback(self):
        """Retourne un fallback si le JSON est invalide."""
        result = self.analyzer.parse_response("not json", trade_id=1,
                                               news_list=[])
        assert result["trade_id"] == 1
        assert result["catalyst_type"] == "UNKNOWN"
        assert result["catalyst_confidence"] == 0.0

    def test_parse_clamps_confidence(self):
        """La confiance est clampee entre 0 et 1."""
        response = json.dumps({
            "primary_news_index": 0,
            "catalyst_type": "EARNINGS",
            "catalyst_confidence": 1.5,
            "catalyst_summary": "test",
            "news_sentiment": 0.0,
            "buy_reason": "x", "sell_reason": "x",
            "trade_quality": "BON",
        })
        result = self.analyzer.parse_response(response, trade_id=1,
                                               news_list=[])
        assert result["catalyst_confidence"] == 1.0

    def test_parse_validates_catalyst_type(self):
        """Un catalyst_type invalide est remplace par UNKNOWN."""
        response = json.dumps({
            "primary_news_index": 0,
            "catalyst_type": "INVALID_TYPE",
            "catalyst_confidence": 0.5,
            "catalyst_summary": "test",
            "news_sentiment": 0.0,
            "buy_reason": "x", "sell_reason": "x",
            "trade_quality": "BON",
        })
        result = self.analyzer.parse_response(response, trade_id=1,
                                               news_list=[])
        assert result["catalyst_type"] == "UNKNOWN"


class TestAnalyzeTrade:
    """Tests d'analyse d'un trade (avec mock Gemini)."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.temp_dir.name, "test.db"))
        self.db.init_db()
        _seed_test_db(self.db)
        self.analyzer = LLMAnalyzer(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    @patch.object(LLMAnalyzer, "_get_client")
    def test_analyze_trade_calls_gemini(self, mock_client_method):
        """analyze_trade appelle l'API Gemini et sauvegarde le resultat."""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "primary_news_index": 1,
            "catalyst_type": "EARNINGS",
            "catalyst_confidence": 0.9,
            "catalyst_summary": "Resultats T2 solides",
            "news_sentiment": 0.7,
            "buy_reason": "CA en hausse",
            "sell_reason": "Objectif atteint",
            "trade_quality": "BON",
        })
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_method.return_value = mock_client

        trades = self.db.get_all_trades()
        self.analyzer.analyze_trade(trades[0])

        # Verifie que l'API a ete appelee
        mock_client.models.generate_content.assert_called_once()

        # Verifie que le resultat est en base
        result = self.db.get_trade_analysis(trades[0]["id"])
        assert result is not None
        assert result["catalyst_type"] == "EARNINGS"

    def test_analyze_trade_skip_if_exists(self):
        """Skip un trade deja analyse (reprise incrementale)."""
        # Pre-insert an analysis
        self.db.insert_trade_analysis({
            "trade_id": 1, "primary_news_id": None,
            "catalyst_type": "EARNINGS", "catalyst_summary": "deja fait",
            "catalyst_confidence": 0.9, "news_sentiment": 0.7,
            "buy_reason": "x", "sell_reason": "x",
            "trade_quality": "BON", "model_used": "gemini-2.0-flash-lite",
            "analyzed_at": "2026-02-24 19:00:00",
        })
        trades = self.db.get_all_trades()
        # Should return False (skipped) without calling API
        result = self.analyzer.analyze_trade(trades[0])
        assert result is False
