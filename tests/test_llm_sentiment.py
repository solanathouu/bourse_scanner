"""Tests pour le scoring de sentiment LLM des news."""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from src.core.database import Database
from src.analysis.llm_sentiment import LLMSentimentScorer


class TestLLMSentimentScorer:
    """Tests du scoring sentiment (OpenAI mocke)."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.scorer = LLMSentimentScorer(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_build_prompt_contains_title(self):
        """Le prompt contient le titre de la news."""
        prompt = self.scorer._build_prompt("Sanofi resultats T2", "Bons resultats")
        assert "Sanofi resultats T2" in prompt

    def test_build_prompt_contains_description(self):
        """Le prompt contient la description de la news."""
        prompt = self.scorer._build_prompt("News", "Description longue de la news")
        assert "Description longue de la news" in prompt

    def test_parse_response_valid_json(self):
        """Parse une reponse JSON valide."""
        result = self.scorer._parse_response('{"sentiment": 0.75}')
        assert result == 0.75

    def test_parse_response_markdown_wrapped(self):
        """Parse une reponse enveloppee dans ```json ... ```."""
        response = '```json\n{"sentiment": -0.5}\n```'
        result = self.scorer._parse_response(response)
        assert result == -0.5

    def test_parse_response_clamped(self):
        """Le sentiment est clampe entre -1.0 et 1.0."""
        result = self.scorer._parse_response('{"sentiment": 2.5}')
        assert result == 1.0
        result = self.scorer._parse_response('{"sentiment": -3.0}')
        assert result == -1.0

    def test_parse_response_invalid_returns_none(self):
        """Une reponse invalide retourne None."""
        result = self.scorer._parse_response("pas du JSON")
        assert result is None

    @patch.object(LLMSentimentScorer, "_get_openai_client")
    def test_score_news_updates_db(self, mock_client_method):
        """score_news appelle OpenAI et met a jour la base."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"sentiment": 0.6}'
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_method.return_value = mock_client

        # Inserer une news sans sentiment
        self.db.insert_news({
            "ticker": "SAN.PA", "title": "Sanofi resultats",
            "source": "BFM", "url": "https://example.com/1",
            "published_at": "2025-06-15", "description": "Bons resultats",
        })
        news = self.db.get_news_without_sentiment()
        assert len(news) == 1

        success = self.scorer.score_news(news[0])
        assert success is True

        # Verifie la mise a jour en base
        updated = self.db.get_news("SAN.PA")
        assert updated[0]["sentiment"] == 0.6
        assert len(self.db.get_news_without_sentiment()) == 0

    @patch.object(LLMSentimentScorer, "score_news")
    def test_score_all_unscored_skips_already_scored(self, mock_score):
        """score_all_unscored ne traite que les news sans sentiment."""
        # News avec sentiment (deja scoree)
        self.db.insert_news({
            "ticker": "SAN.PA", "title": "News scoree",
            "source": "BFM", "url": "https://example.com/scored",
            "published_at": "2025-06-15", "description": "Deja scoree",
            "sentiment": 0.5, "source_api": "alpha_vantage",
        })
        # News sans sentiment
        self.db.insert_news({
            "ticker": "SAN.PA", "title": "News non scoree",
            "source": "Reuters", "url": "https://example.com/unscored",
            "published_at": "2025-06-16", "description": "A scorer",
        })

        mock_score.return_value = True
        result = self.scorer.score_all_unscored()

        assert result["total"] == 1
        assert mock_score.call_count == 1
