"""Collecte de news via Alpha Vantage News Sentiment API."""

import os
import time

import requests
from dotenv import load_dotenv
from loguru import logger

from src.core.database import Database
from src.data_collection.ticker_mapper import TickerMapper, TickerNotFoundError

load_dotenv()

API_URL = "https://www.alphavantage.co/query"
DELAY_BETWEEN_REQUESTS = 2  # secondes (limite: 1 req/sec)


class AlphaVantageCollector:
    """Collecte les news avec sentiment via Alpha Vantage."""

    def __init__(self, db: Database):
        self.db = db
        self.mapper = TickerMapper()
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        if not self.api_key:
            logger.warning("ALPHA_VANTAGE_API_KEY non configuree dans .env")

    def _parse_article(self, article: dict, ticker: str) -> dict:
        """Transforme un article Alpha Vantage en dict pour la base."""
        # Date format: 20260222T202400 -> 2026-02-22 20:24:00
        raw_date = article.get("time_published", "")
        published_at = ""
        if raw_date and len(raw_date) >= 15:
            published_at = (
                f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]} "
                f"{raw_date[9:11]}:{raw_date[11:13]}:{raw_date[13:15]}"
            )

        sentiment = article.get("overall_sentiment_score")
        if sentiment is not None:
            sentiment = float(sentiment)

        return {
            "ticker": ticker,
            "title": article.get("title", ""),
            "source": article.get("source", ""),
            "url": article.get("url", ""),
            "published_at": published_at,
            "description": article.get("summary", ""),
            "sentiment": sentiment,
            "source_api": "alpha_vantage",
        }

    def collect_for_action(
        self, nom_action: str, ticker: str, time_from: str, time_to: str
    ) -> int:
        """Collecte les news pour une action sur une periode.

        Args:
            nom_action: Nom pour la recherche keyword
            ticker: Ticker Yahoo pour le stockage
            time_from: Date debut YYYYMMDDTHHMMSS
            time_to: Date fin YYYYMMDDTHHMMSS

        Returns:
            Nombre de news inserees.
        """
        logger.info(f"Alpha Vantage: {nom_action} ({time_from} -> {time_to})")

        params = {
            "function": "NEWS_SENTIMENT",
            "keywords": nom_action,
            "time_from": time_from,
            "time_to": time_to,
            "limit": 200,
            "apikey": self.api_key,
        }

        try:
            resp = requests.get(API_URL, params=params, timeout=30)
            data = resp.json()
        except Exception as e:
            logger.error(f"Alpha Vantage erreur HTTP: {e}")
            return 0

        # Verifier les erreurs API
        if "Error Message" in data or "Information" in data:
            msg = data.get("Error Message", data.get("Information", ""))
            logger.warning(f"Alpha Vantage: {msg[:100]}")
            return 0

        articles = data.get("feed", [])
        if not articles:
            logger.warning(f"Alpha Vantage: 0 articles pour {nom_action}")
            return 0

        news_list = [self._parse_article(a, ticker) for a in articles]
        news_list = [n for n in news_list if n["url"]]

        self.db.insert_news_batch(news_list)
        logger.info(f"Alpha Vantage: {len(news_list)} news pour {nom_action}")
        return len(news_list)

    def collect_all(self) -> dict:
        """Collecte les news pour toutes les actions tradees.

        Returns:
            Dict {"total_news": int, "errors": list}
        """
        if not self.api_key:
            logger.error("ALPHA_VANTAGE_API_KEY manquante, collecte impossible")
            return {"total_news": 0, "errors": [{"action": "ALL", "error": "API key missing"}]}

        trades = self.db.get_all_trades()
        # Regrouper par action: periode globale
        ranges = {}
        today_str = time.strftime("%Y%m%d")
        for trade in trades:
            name = trade["nom_action"].lstrip("* ").strip()
            date_achat = trade["date_achat"][:10].replace("-", "")
            date_vente = (
                trade["date_vente"][:10].replace("-", "")
                if trade["date_vente"] else today_str
            )
            if name not in ranges:
                ranges[name] = {"start": date_achat, "end": date_vente}
            else:
                if date_achat < ranges[name]["start"]:
                    ranges[name]["start"] = date_achat
                if date_vente > ranges[name]["end"]:
                    ranges[name]["end"] = date_vente

        total = 0
        errors = []

        for nom_action, period in ranges.items():
            try:
                ticker = self.mapper.get_ticker(nom_action)
                time_from = period["start"] + "T0000"
                time_to = period["end"] + "T2359"
                count = self.collect_for_action(
                    nom_action, ticker, time_from, time_to
                )
                total += count
                time.sleep(DELAY_BETWEEN_REQUESTS)
            except TickerNotFoundError as e:
                errors.append({"action": nom_action, "error": str(e)})
                logger.warning(f"Ticker inconnu: {nom_action}")
            except Exception as e:
                errors.append({"action": nom_action, "error": str(e)})
                logger.error(f"Alpha Vantage erreur {nom_action}: {e}")

        logger.info(f"Alpha Vantage termine: {total} news, {len(errors)} erreurs")
        return {"total_news": total, "errors": errors}
