"""Analyse LLM des trades de Nicolas via Gemini Flash Lite.

Pour chaque trade, envoie le contexte complet (news, indicateurs techniques,
infos du trade) a Gemini et recupere une analyse structuree du catalyseur,
de la raison d'achat/vente, et de la qualite du trade.
"""

import json
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from loguru import logger
from google import genai
from google.genai import types

from src.core.database import Database
from src.data_collection.ticker_mapper import TickerMapper, TickerNotFoundError
from src.analysis.technical_indicators import TechnicalIndicators

load_dotenv()

VALID_CATALYST_TYPES = [
    "EARNINGS", "FDA_REGULATORY", "UPGRADE", "DOWNGRADE", "CONTRACT",
    "DIVIDEND", "RESTRUCTURING", "INSIDER", "SECTOR_MACRO",
    "OTHER_POSITIVE", "OTHER_NEGATIVE", "TECHNICAL", "UNKNOWN",
]

VALID_TRADE_QUALITIES = ["EXCELLENT", "BON", "MOYEN", "MAUVAIS"]

LLM_ANALYSIS_PROMPT = """Tu es un analyste financier expert specialise dans le trading swing court terme sur le marche francais (PEA).

Tu analyses les trades d'un trader nomme Nicolas pour comprendre sa logique de decision.
Nicolas a un style swing court terme (quelques jours), objectif 4-5% par trade, 89% win rate.
Il n'achete JAMAIS au hasard — chaque achat est motive par un catalyseur (news, annonce, signal technique).

Voici un trade a analyser:

## Infos du trade
- Action: {nom_action}
- Achat: {date_achat} a {prix_achat:.2f} EUR
- Duree: {duree_jours} jours

## Indicateurs techniques au moment de l'achat
{technical_context}

## News autour de la date d'achat (J-5 a J+1)
{news_context}

## Instructions

Analyse ce trade et reponds en JSON strict avec ces champs:
- "primary_news_index": numero de la news qui a le plus probablement declenche l'achat (1 = premiere news listee). 0 si aucune news n'est pertinente.
- "catalyst_type": un parmi {valid_types}
- "catalyst_confidence": float 0.0-1.0, ta confiance dans l'identification du catalyseur
- "catalyst_summary": une phrase commencant par "Nicolas a achete parce que..."
- "news_sentiment": float -1.0 a +1.0, sentiment de la news declencheuse (0 si aucune)
- "buy_reason": 2-3 phrases expliquant pourquoi Nicolas a achete a ce moment precis
- "sell_reason": 1-2 phrases expliquant pourquoi il a probablement vendu

Reponds UNIQUEMENT avec le JSON, sans texte autour."""


class LLMAnalyzer:
    """Analyse les trades de Nicolas via Gemini."""

    def __init__(self, db: Database, model: str = "gemini-2.5-flash-lite"):
        self.db = db
        self.model = model
        self.mapper = TickerMapper()
        self.tech = TechnicalIndicators()
        self._price_cache: dict[str, object] = {}

    def _get_client(self) -> genai.Client:
        """Cree un client Gemini avec la cle du .env."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY manquante dans .env")
        return genai.Client(api_key=api_key)

    def _get_enriched_prices(self, ticker: str):
        """Recupere les prix enrichis (avec cache)."""
        if ticker in self._price_cache:
            return self._price_cache[ticker]
        import pandas as pd
        prices = self.db.get_prices(ticker)
        if len(prices) < 20:
            return None
        df = pd.DataFrame(prices)
        enriched = self.tech.compute_all(df)
        self._price_cache[ticker] = enriched
        return enriched

    def _get_technical_context(self, trade: dict) -> str:
        """Construit le contexte technique pour le prompt."""
        nom_action = trade["nom_action"]
        date_achat = trade["date_achat"][:10]
        try:
            ticker = self.mapper.get_ticker(nom_action)
        except TickerNotFoundError:
            return "Indicateurs techniques non disponibles (ticker inconnu)"

        enriched = self._get_enriched_prices(ticker)
        if enriched is None:
            return "Indicateurs techniques non disponibles (pas assez de prix)"

        indicators = self.tech.get_indicators_at_date(enriched, date_achat)
        if indicators is None:
            return "Indicateurs techniques non disponibles (date non trouvee)"

        lines = []
        labels = {
            "rsi_14": "RSI(14)",
            "macd_histogram": "MACD Histogram",
            "bollinger_position": "Position Bollinger (0=bas, 1=haut)",
            "range_position_10": "Position dans range 10j (0=bas, 1=haut)",
            "range_position_20": "Position dans range 20j (0=bas, 1=haut)",
            "volume_ratio_20": "Volume / Moyenne 20j",
            "atr_14_pct": "ATR(14) en %",
            "variation_1j": "Variation J-1 (%)",
            "variation_5j": "Variation 5 jours (%)",
            "distance_sma20": "Distance SMA20 (%)",
            "distance_sma50": "Distance SMA50 (%)",
        }
        for key, label in labels.items():
            val = indicators.get(key)
            if val is not None:
                lines.append(f"- {label}: {val:.2f}")
        return "\n".join(lines) if lines else "Aucun indicateur disponible"

    def _get_news_context(self, trade: dict) -> tuple[str, list[dict]]:
        """Construit le contexte news et retourne (texte, liste_news)."""
        nom_action = trade["nom_action"]
        date_achat = trade["date_achat"][:10]
        try:
            ticker = self.mapper.get_ticker(nom_action)
        except TickerNotFoundError:
            return "Aucune news disponible (ticker inconnu)", []

        date_start = (datetime.strptime(date_achat, "%Y-%m-%d")
                      - timedelta(days=5)).strftime("%Y-%m-%d")
        date_end = (datetime.strptime(date_achat, "%Y-%m-%d")
                    + timedelta(days=1)).strftime("%Y-%m-%d")

        news_list = self.db.get_news_in_window(ticker, date_start, date_end)

        if not news_list:
            return "Aucune news trouvee dans la fenetre J-5 a J+1", []

        lines = []
        for i, news in enumerate(news_list, 1):
            news_date = news["published_at"][:10]
            distance = (datetime.strptime(news_date, "%Y-%m-%d")
                        - datetime.strptime(date_achat, "%Y-%m-%d")).days
            sentiment_str = ""
            if news.get("sentiment") is not None:
                sentiment_str = f" [sentiment: {news['sentiment']:.1f}]"
            desc = news.get("description") or ""
            desc_short = desc[:150] + "..." if len(desc) > 150 else desc
            lines.append(
                f"{i}. [J{distance:+d}] \"{news['title']}\""
                f" (source: {news.get('source', 'inconnue')})"
                f"{sentiment_str}"
                f"\n   {desc_short}"
            )

        return "\n".join(lines), news_list

    def build_prompt(self, trade: dict) -> str:
        """Construit le prompt complet pour analyser un trade."""
        technical_context = self._get_technical_context(trade)
        news_context, _ = self._get_news_context(trade)

        duree = trade["duree_jours"] or 0

        return LLM_ANALYSIS_PROMPT.format(
            nom_action=trade["nom_action"],
            date_achat=trade["date_achat"][:10],
            prix_achat=trade["prix_achat"],
            duree_jours=duree,
            technical_context=technical_context,
            news_context=news_context,
            valid_types=", ".join(VALID_CATALYST_TYPES),
        )

    def parse_response(self, response_text: str, trade_id: int,
                        news_list: list[dict]) -> dict:
        """Parse la reponse JSON du LLM en dict pret pour la base."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fallback = {
            "trade_id": trade_id,
            "primary_news_id": None,
            "catalyst_type": "UNKNOWN",
            "catalyst_summary": "Analyse LLM echouee",
            "catalyst_confidence": 0.0,
            "news_sentiment": 0.0,
            "buy_reason": "",
            "sell_reason": "",
            "trade_quality": "MOYEN",
            "model_used": self.model,
            "analyzed_at": now,
        }

        # Nettoyer le texte (le LLM enveloppe parfois dans ```json ... ```)
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error(f"Trade #{trade_id}: reponse LLM non-JSON: "
                         f"{response_text[:100]}")
            return fallback

        # Resoudre primary_news_id
        news_index = data.get("primary_news_index", 0)
        primary_news_id = None
        if news_index > 0 and news_index <= len(news_list):
            primary_news_id = news_list[news_index - 1].get("id")

        # Valider catalyst_type
        catalyst_type = data.get("catalyst_type", "UNKNOWN")
        if catalyst_type not in VALID_CATALYST_TYPES:
            catalyst_type = "UNKNOWN"

        # Valider trade_quality
        trade_quality = data.get("trade_quality", "MOYEN")
        if trade_quality not in VALID_TRADE_QUALITIES:
            trade_quality = "MOYEN"

        # Clamper confidence
        confidence = float(data.get("catalyst_confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        # Clamper sentiment
        sentiment = float(data.get("news_sentiment", 0.0))
        sentiment = max(-1.0, min(1.0, sentiment))

        return {
            "trade_id": trade_id,
            "primary_news_id": primary_news_id,
            "catalyst_type": catalyst_type,
            "catalyst_summary": data.get("catalyst_summary", "")[:500],
            "catalyst_confidence": confidence,
            "news_sentiment": sentiment,
            "buy_reason": data.get("buy_reason", "")[:1000],
            "sell_reason": data.get("sell_reason", "")[:500],
            "trade_quality": trade_quality,
            "model_used": self.model,
            "analyzed_at": now,
        }

    def analyze_trade(self, trade: dict) -> bool:
        """Analyse un trade via Gemini. Retourne False si skip (deja fait)."""
        trade_id = trade["id"]

        # Reprise incrementale
        existing = self.db.get_trade_analysis(trade_id)
        if existing:
            logger.debug(f"Trade #{trade_id} deja analyse, skip")
            return False

        _, news_list = self._get_news_context(trade)
        prompt = self.build_prompt(trade)

        client = self._get_client()
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=500,
            ),
        )

        response_text = response.text
        analysis = self.parse_response(response_text, trade_id, news_list)
        self.db.insert_trade_analysis(analysis)

        logger.info(f"Trade #{trade_id} ({trade['nom_action']}): "
                     f"{analysis['catalyst_type']} "
                     f"(conf={analysis['catalyst_confidence']:.2f})")
        return True

    def analyze_all_trades(self) -> dict:
        """Analyse tous les trades clotures. Reprise incrementale.

        Returns:
            Resume: {total, analyzed, skipped, errors}.
        """
        trades = self.db.get_all_trades()
        closed = [t for t in trades if t["statut"] == "CLOTURE"]

        analyzed = 0
        skipped = 0
        errors = 0

        logger.info(f"Analyse LLM de {len(closed)} trades clotures...")

        for trade in closed:
            try:
                result = self.analyze_trade(trade)
                if result:
                    analyzed += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.error(f"Erreur trade #{trade['id']} "
                             f"({trade['nom_action']}): {e}")
                errors += 1

        summary = {
            "total": len(closed),
            "analyzed": analyzed,
            "skipped": skipped,
            "errors": errors,
        }
        logger.info(f"Analyse terminee: {summary}")
        return summary
