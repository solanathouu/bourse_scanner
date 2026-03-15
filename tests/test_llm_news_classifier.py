"""Tests pour le classificateur LLM de news."""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from src.core.database import Database
from src.analysis.llm_news_classifier import LLMNewsClassifier


class TestBuildPrompt:
    """Tests de construction du prompt."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.temp_dir.name, "test.db"))
        self.db.init_db()
        self.classifier = LLMNewsClassifier(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_prompt_contains_ticker(self):
        """Le prompt contient le ticker et le nom de l'entreprise."""
        news = [{"id": 1, "title": "Test news", "description": "Desc",
                 "published_at": "2026-03-10", "source": "BFM"}]
        prompt = self.classifier._build_prompt("SAN.PA", "SANOFI", news)
        assert "SAN.PA" in prompt
        assert "SANOFI" in prompt

    def test_prompt_contains_all_news(self):
        """Le prompt contient toutes les news numerotees."""
        news = [
            {"id": 1, "title": "News 1", "description": "D1",
             "published_at": "2026-03-10", "source": "BFM"},
            {"id": 2, "title": "News 2", "description": "D2",
             "published_at": "2026-03-11", "source": "Reuters"},
        ]
        prompt = self.classifier._build_prompt("SAN.PA", "SANOFI", news)
        assert "1." in prompt
        assert "News 1" in prompt
        assert "2." in prompt
        assert "News 2" in prompt

    def test_prompt_contains_valid_types(self):
        """Le prompt contient les types de catalyseurs valides."""
        news = [{"id": 1, "title": "T", "description": "",
                 "published_at": "2026-03-10", "source": "X"}]
        prompt = self.classifier._build_prompt("SAN.PA", "SANOFI", news)
        assert "EARNINGS" in prompt
        assert "FDA_REGULATORY" in prompt
        assert "UPGRADE" in prompt

    def test_prompt_truncates_description(self):
        """La description est tronquee a 200 caracteres."""
        long_desc = "X" * 500
        news = [{"id": 1, "title": "T", "description": long_desc,
                 "published_at": "2026-03-10", "source": "X"}]
        prompt = self.classifier._build_prompt("SAN.PA", "SANOFI", news)
        # La description dans le prompt ne doit pas contenir les 500 chars
        assert "X" * 201 not in prompt


class TestParseResponse:
    """Tests de parsing de la reponse LLM."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.temp_dir.name, "test.db"))
        self.db.init_db()
        self.classifier = LLMNewsClassifier(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_parse_valid_response(self):
        """Parse une reponse JSON array valide."""
        news_list = [{"id": 42}, {"id": 43}]
        response = json.dumps([
            {"news_index": 1, "catalyst_type": "EARNINGS",
             "confidence": 0.9, "relevance": 0.95,
             "explanation": "Resultats solides"},
            {"news_index": 2, "catalyst_type": "SECTOR_MACRO",
             "confidence": 0.3, "relevance": 0.2,
             "explanation": "Contexte macro"},
        ])
        results = self.classifier._parse_response(response, news_list)
        assert len(results) == 2
        assert results[0]["news_id"] == 42
        assert results[0]["catalyst_type"] == "EARNINGS"
        assert results[0]["confidence"] == 0.9

    def test_parse_markdown_wrapped(self):
        """Parse un JSON enveloppe dans des backticks markdown."""
        news_list = [{"id": 10}]
        inner = json.dumps([
            {"news_index": 1, "catalyst_type": "UPGRADE",
             "confidence": 0.8, "relevance": 1.0,
             "explanation": "Objectif releve"}
        ])
        response = f"```json\n{inner}\n```"
        results = self.classifier._parse_response(response, news_list)
        assert len(results) == 1
        assert results[0]["catalyst_type"] == "UPGRADE"

    def test_parse_invalid_json_returns_empty(self):
        """Une reponse invalide retourne une liste vide."""
        results = self.classifier._parse_response("pas du json", [{"id": 1}])
        assert results == []

    def test_parse_validates_catalyst_type(self):
        """Un type invalide est remplace par UNKNOWN."""
        news_list = [{"id": 1}]
        response = json.dumps([
            {"news_index": 1, "catalyst_type": "INVALID",
             "confidence": 0.5, "relevance": 0.8,
             "explanation": "test"}
        ])
        results = self.classifier._parse_response(response, news_list)
        assert results[0]["catalyst_type"] == "UNKNOWN"

    def test_parse_clamps_confidence(self):
        """La confiance est clampee entre 0 et 1."""
        news_list = [{"id": 1}]
        response = json.dumps([
            {"news_index": 1, "catalyst_type": "EARNINGS",
             "confidence": 1.5, "relevance": -0.3,
             "explanation": "test"}
        ])
        results = self.classifier._parse_response(response, news_list)
        assert results[0]["confidence"] == 1.0
        assert results[0]["relevance"] == 0.0

    def test_parse_skips_invalid_index(self):
        """Les index hors limites sont ignores."""
        news_list = [{"id": 1}]
        response = json.dumps([
            {"news_index": 5, "catalyst_type": "EARNINGS",
             "confidence": 0.5, "relevance": 0.5,
             "explanation": "test"}
        ])
        results = self.classifier._parse_response(response, news_list)
        assert results == []


class TestSummarize:
    """Tests du resume pour features temps reel."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.temp_dir.name, "test.db"))
        self.db.init_db()
        self.classifier = LLMNewsClassifier(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_summarize_picks_highest_confidence(self):
        """Retourne le catalyseur avec la plus haute confiance."""
        news = [
            {"llm_catalyst_type": "EARNINGS", "llm_catalyst_confidence": 0.9,
             "llm_relevance_score": 0.8, "title": "Resultats",
             "llm_explanation": "Bons resultats"},
            {"llm_catalyst_type": "SECTOR_MACRO", "llm_catalyst_confidence": 0.3,
             "llm_relevance_score": 0.5, "title": "Marche",
             "llm_explanation": "Contexte"},
        ]
        result = self.classifier.summarize_for_realtime(news)
        assert result["catalyst_type"] == "EARNINGS"
        assert result["catalyst_confidence"] == 0.9
        assert result["has_clear_catalyst"] == 1

    def test_summarize_filters_irrelevant(self):
        """Filtre les news avec relevance < 0.3."""
        news = [
            {"llm_catalyst_type": "EARNINGS", "llm_catalyst_confidence": 0.9,
             "llm_relevance_score": 0.1, "title": "Faux positif",
             "llm_explanation": "Pas pertinent"},
        ]
        result = self.classifier.summarize_for_realtime(news)
        assert result["catalyst_type"] == "TECHNICAL"
        assert result["has_clear_catalyst"] == 0

    def test_summarize_empty_returns_technical(self):
        """Liste vide retourne TECHNICAL."""
        result = self.classifier.summarize_for_realtime([])
        assert result["catalyst_type"] == "TECHNICAL"
        assert result["catalyst_confidence"] == 0.0

    def test_summarize_skips_unclassified(self):
        """Ignore les news sans llm_catalyst_type."""
        news = [
            {"llm_catalyst_type": None, "llm_catalyst_confidence": None,
             "llm_relevance_score": None, "title": "Non classifiee"},
        ]
        result = self.classifier.summarize_for_realtime(news)
        assert result["catalyst_type"] == "TECHNICAL"


class TestClassifyAndCache:
    """Tests de classification avec cache DB (Gemini mocke)."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.temp_dir.name, "test.db"))
        self.db.init_db()
        self.classifier = LLMNewsClassifier(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    @patch.object(LLMNewsClassifier, "_get_client")
    def test_classify_calls_gemini_and_caches(self, mock_client_method):
        """classify_and_cache appelle Gemini et sauvegarde en base."""
        mock_response = MagicMock()
        mock_response.text = json.dumps([
            {"news_index": 1, "catalyst_type": "EARNINGS",
             "confidence": 0.85, "relevance": 0.95,
             "explanation": "Bons resultats T3"}
        ])
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_method.return_value = mock_client

        # Inserer une news
        self.db.insert_news({
            "ticker": "SAN.PA", "title": "Sanofi resultats",
            "source": "BFM", "url": "https://example.com/1",
            "published_at": "2026-03-10", "description": "Bons resultats",
        })
        news = self.db.get_news("SAN.PA")

        result = self.classifier.classify_and_cache("SAN.PA", "SANOFI", news)

        # Verifie l'appel API
        mock_client.models.generate_content.assert_called_once()

        # Verifie la mise a jour en base
        updated = self.db.get_news("SAN.PA")
        assert updated[0]["llm_catalyst_type"] == "EARNINGS"
        assert updated[0]["llm_catalyst_confidence"] == 0.85
        assert updated[0]["llm_relevance_score"] == 0.95

    @patch.object(LLMNewsClassifier, "_get_client")
    def test_classify_skips_already_classified(self, mock_client_method):
        """Ne rappelle pas Gemini pour les news deja classifiees."""
        # Inserer une news deja classifiee
        self.db.insert_news({
            "ticker": "SAN.PA", "title": "Deja classifiee",
            "source": "BFM", "url": "https://example.com/2",
            "published_at": "2026-03-10", "description": "Test",
        })
        news = self.db.get_news("SAN.PA")
        # Simuler une classification existante
        self.db.update_news_llm_classification(
            news[0]["id"], "EARNINGS", 0.9, 0.95,
        )
        # Recharger avec les colonnes mises a jour
        news = self.db.get_news("SAN.PA")

        self.classifier.classify_and_cache("SAN.PA", "SANOFI", news)

        # Gemini ne doit PAS avoir ete appele
        mock_client_method.assert_not_called()
