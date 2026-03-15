"""Score le sentiment des news sans sentiment via Gemini Flash Lite.

Pour chaque news avec sentiment IS NULL, envoie le titre et la description
a Gemini et recupere un score de sentiment entre -1.0 et +1.0.
"""

import json
import os
import time

from dotenv import load_dotenv
from loguru import logger
from google import genai
from google.genai import types

from src.core.database import Database

load_dotenv()

SENTIMENT_PROMPT = """Tu es un analyste financier. Score le sentiment de cette news boursiere.

Titre: {title}
Description: {description}

Reponds UNIQUEMENT avec un JSON: {{"sentiment": <float entre -1.0 et 1.0>}}
- -1.0 = tres negatif (profit warning, scandale, faillite)
- -0.5 = negatif (resultats decevants, downgrade)
- 0.0 = neutre (information factuelle, nomination)
- +0.5 = positif (bons resultats, upgrade, partenariat)
- +1.0 = tres positif (acquisition majeure, FDA approval, contrat record)"""

DELAY_BETWEEN_CALLS = 0.5  # secondes


class LLMSentimentScorer:
    """Score le sentiment des news via Gemini."""

    def __init__(self, db: Database, model: str = "gemini-2.0-flash"):
        self.db = db
        self.model = model

    def _get_client(self) -> genai.Client:
        """Cree un client Gemini avec la cle du .env."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY manquante dans .env")
        return genai.Client(api_key=api_key)

    def _build_prompt(self, title: str, description: str) -> str:
        """Construit le prompt de scoring sentiment."""
        desc = (description or "")[:300]
        return SENTIMENT_PROMPT.format(title=title, description=desc)

    def _parse_response(self, response_text: str) -> float | None:
        """Parse la reponse JSON du LLM. Retourne le sentiment ou None."""
        cleaned = response_text.strip()
        # Le LLM enveloppe parfois dans ```json ... ```
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
            sentiment = float(data.get("sentiment", 0.0))
            return max(-1.0, min(1.0, sentiment))
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.error(f"Reponse LLM non parsable: {response_text[:100]}")
            return None

    def score_news(self, news: dict) -> bool:
        """Score une news et met a jour en base.

        Returns:
            True si scoree avec succes, False sinon.
        """
        title = news.get("title", "")
        description = news.get("description", "")
        news_id = news["id"]

        prompt = self._build_prompt(title, description)
        client = self._get_client()

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=50,
            ),
        )

        response_text = response.text
        sentiment = self._parse_response(response_text)

        if sentiment is not None:
            self.db.update_news_sentiment(news_id, sentiment)
            return True

        return False

    def score_all_unscored(self, batch_size: int = 20,
                           max_consecutive_errors: int = 5) -> dict:
        """Score toutes les news sans sentiment, par lots.

        S'arrete automatiquement apres max_consecutive_errors erreurs
        consecutives (ex: cle API invalide, quota depasse).

        Returns:
            Dict {"total": int, "scored": int, "errors": int}.
        """
        unscored = self.db.get_news_without_sentiment()
        total = len(unscored)

        if total == 0:
            logger.info("Aucune news a scorer")
            return {"total": 0, "scored": 0, "errors": 0}

        logger.info(f"Scoring sentiment de {total} news...")

        scored = 0
        errors = 0
        consecutive_errors = 0

        for i, news in enumerate(unscored):
            try:
                success = self.score_news(news)
                if success:
                    scored += 1
                    consecutive_errors = 0
                else:
                    errors += 1
                    consecutive_errors += 1
            except Exception as e:
                logger.error(f"Erreur scoring news #{news['id']}: {e}")
                errors += 1
                consecutive_errors += 1

            if consecutive_errors >= max_consecutive_errors:
                logger.warning(
                    f"Arret apres {max_consecutive_errors} erreurs consecutives "
                    f"({scored} OK, {errors} erreurs sur {i + 1} tentatives)"
                )
                break

            if (i + 1) % batch_size == 0:
                logger.info(f"  Progress: {i + 1}/{total} ({scored} OK, {errors} erreurs)")

            time.sleep(DELAY_BETWEEN_CALLS)

        summary = {"total": total, "scored": scored, "errors": errors}
        logger.info(f"Scoring termine: {summary}")
        return summary
