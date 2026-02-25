"""Collecte de news via l'API Newsdata.io (5e source).

Newsdata.io fournit des articles en francais avec sentiment integre.
200 credits/jour, 10 resultats par credit.
"""

import os
import time
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv
from loguru import logger

from src.core.database import Database
from src.data_collection.ticker_mapper import TickerMapper

load_dotenv()

API_BASE_URL = "https://newsdata.io/api/1"
DELAY_BETWEEN_CALLS = 1  # secondes


class NewsdataCollector:
    """Collecte les news depuis Newsdata.io."""

    def __init__(self, db: Database):
        self.db = db
        self.mapper = TickerMapper()
        self.api_key = os.getenv("NEWSDATA_API_KEY")
        if not self.api_key:
            raise ValueError("NEWSDATA_API_KEY manquante dans .env")

    def _parse_article(self, article: dict, ticker: str) -> dict | None:
        """Transforme un article Newsdata.io en dict pour la base.

        Returns:
            dict pour insert_news, ou None si article invalide.
        """
        title = article.get("title", "")
        if not title:
            return None

        url = article.get("link", "")
        if not url:
            return None

        published_at = article.get("pubDate", "")
        description = article.get("description") or ""

        # Newsdata.io fournit un champ sentiment (optionnel)
        sentiment = None
        sent_data = article.get("sentiment")
        if sent_data and isinstance(sent_data, dict):
            # Format: {"positive": 0.8, "negative": 0.1, "neutral": 0.1}
            pos = sent_data.get("positive", 0) or 0
            neg = sent_data.get("negative", 0) or 0
            sentiment = round(pos - neg, 2)

        return {
            "ticker": ticker,
            "title": title[:500],
            "source": article.get("source_id", "newsdata"),
            "url": url,
            "published_at": published_at,
            "description": description[:500],
            "sentiment": sentiment,
            "source_api": "newsdata",
        }

    def collect_for_action(self, nom_action: str, ticker: str,
                           from_date: str | None = None,
                           to_date: str | None = None) -> int:
        """Collecte les news pour une action.

        Returns:
            Nombre de news inserees.
        """
        # Dates par defaut: 1 an en arriere
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")
        if not from_date:
            from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        # Preparer le query (nom simplifie)
        query = nom_action.split()[0]  # Premier mot
        if len(nom_action.split()) > 1 and len(nom_action.split()[0]) <= 3:
            query = " ".join(nom_action.split()[:2])  # "AB SCIENCE" -> "AB Science"

        params = {
            "apikey": self.api_key,
            "q": query,
            "language": "fr",
            "from_date": from_date,
            "to_date": to_date,
        }

        logger.info(f"Newsdata.io: collecte '{query}' ({from_date} -> {to_date})")

        try:
            response = requests.get(f"{API_BASE_URL}/archive", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Newsdata.io erreur pour {nom_action}: {e}")
            return 0

        if data.get("status") != "success":
            logger.warning(f"Newsdata.io: status={data.get('status')} pour {nom_action}")
            return 0

        articles = data.get("results") or []
        news_list = []
        for article in articles:
            parsed = self._parse_article(article, ticker)
            if parsed:
                news_list.append(parsed)

        if news_list:
            self.db.insert_news_batch(news_list)
            logger.info(f"Newsdata.io: {nom_action} -> {len(news_list)} articles")

        return len(news_list)

    def collect_all(self) -> dict:
        """Collecte les news pour toutes les actions tradees.

        Returns:
            Dict {"total_news": int, "actions_ok": int, "errors": list}
        """
        mappings = self.mapper.get_all_mappings()
        total = 0
        actions_ok = 0
        errors = []

        for nom_action, ticker in sorted(mappings.items()):
            # Skip les ETF
            if "ETF" in nom_action or "BNPP" in nom_action:
                continue

            try:
                count = self.collect_for_action(nom_action, ticker)
                total += count
                actions_ok += 1
                time.sleep(DELAY_BETWEEN_CALLS)
            except Exception as e:
                errors.append({"action": nom_action, "error": str(e)})
                logger.error(f"Newsdata.io erreur {nom_action}: {e}")

        logger.info(f"Newsdata.io termine: {total} news, {actions_ok} actions, "
                     f"{len(errors)} erreurs")
        return {
            "total_news": total,
            "actions_ok": actions_ok,
            "errors": errors,
        }
