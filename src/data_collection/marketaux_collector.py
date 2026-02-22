"""Collecte de news via Marketaux API."""

import os
import time

import requests
from dotenv import load_dotenv
from loguru import logger

from src.core.database import Database
from src.data_collection.ticker_mapper import TickerMapper, TickerNotFoundError

load_dotenv()

API_URL = "https://api.marketaux.com/v1/news/all"
DELAY_BETWEEN_REQUESTS = 2  # secondes (limite: 30 req/min)


class MarketauxCollector:
    """Collecte les news avec sentiment via Marketaux."""

    def __init__(self, db: Database):
        self.db = db
        self.mapper = TickerMapper()
        self.api_key = os.getenv("MARKETAUX_API_KEY", "")
        if not self.api_key:
            logger.warning("MARKETAUX_API_KEY non configuree dans .env")

    def _parse_article(self, article: dict, ticker: str) -> dict:
        """Transforme un article Marketaux en dict pour la base."""
        # Date format: 2025-12-26T13:50:58.000000Z -> 2025-12-26 13:50:58
        raw_date = article.get("published_at", "")
        published_at = ""
        if raw_date:
            published_at = raw_date[:19].replace("T", " ")

        # Sentiment: chercher dans entities pour notre ticker
        sentiment = None
        for entity in article.get("entities", []):
            sent = entity.get("sentiment_score")
            if sent is not None:
                sentiment = float(sent)
                break  # Prendre le premier score disponible

        return {
            "ticker": ticker,
            "title": article.get("title", ""),
            "source": article.get("source", ""),
            "url": article.get("url", ""),
            "published_at": published_at,
            "description": article.get("description", ""),
            "sentiment": sentiment,
            "source_api": "marketaux",
        }

    def collect_for_action(
        self, nom_action: str, ticker: str, start: str, end: str
    ) -> int:
        """Collecte les news pour une action sur une periode.

        Args:
            nom_action: Nom pour la recherche
            ticker: Ticker Yahoo pour le stockage
            start: Date debut YYYY-MM-DD
            end: Date fin YYYY-MM-DD

        Returns:
            Nombre de news inserees.
        """
        logger.info(f"Marketaux: {nom_action} ({start} -> {end})")

        params = {
            "search": nom_action,
            "language": "fr,en",
            "published_after": start,
            "published_before": end,
            "filter_entities": "true",
            "limit": 50,
            "api_token": self.api_key,
        }

        total_inserted = 0
        page = 1

        while True:
            params["page"] = page
            try:
                resp = requests.get(API_URL, params=params, timeout=30)
                data = resp.json()
            except Exception as e:
                logger.error(f"Marketaux erreur HTTP: {e}")
                break

            if "error" in data:
                logger.warning(f"Marketaux: {data['error'].get('message', data['error'])}")
                break

            articles = data.get("data", [])
            if not articles:
                break

            news_list = [self._parse_article(a, ticker) for a in articles]
            news_list = [n for n in news_list if n["url"]]

            self.db.insert_news_batch(news_list)
            total_inserted += len(news_list)

            # Verifier s'il y a d'autres pages
            meta = data.get("meta", {})
            total_found = meta.get("found", 0)
            returned = meta.get("returned", 0)
            if page * 50 >= total_found or returned == 0:
                break

            page += 1
            time.sleep(DELAY_BETWEEN_REQUESTS)

        if total_inserted:
            logger.info(f"Marketaux: {total_inserted} news pour {nom_action}")
        else:
            logger.warning(f"Marketaux: 0 news pour {nom_action}")

        return total_inserted

    def collect_all(self) -> dict:
        """Collecte les news pour toutes les actions tradees.

        Returns:
            Dict {"total_news": int, "errors": list}
        """
        if not self.api_key:
            logger.error("MARKETAUX_API_KEY manquante, collecte impossible")
            return {"total_news": 0, "errors": [{"action": "ALL", "error": "API key missing"}]}

        trades = self.db.get_all_trades()
        # Regrouper par action: periode globale
        ranges = {}
        today = time.strftime("%Y-%m-%d")
        for trade in trades:
            name = trade["nom_action"].lstrip("* ").strip()
            date_achat = trade["date_achat"][:10]
            date_vente = trade["date_vente"][:10] if trade["date_vente"] else today
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
                count = self.collect_for_action(
                    nom_action, ticker, period["start"], period["end"]
                )
                total += count
                time.sleep(DELAY_BETWEEN_REQUESTS)
            except TickerNotFoundError as e:
                errors.append({"action": nom_action, "error": str(e)})
            except Exception as e:
                errors.append({"action": nom_action, "error": str(e)})
                logger.error(f"Marketaux erreur {nom_action}: {e}")

        logger.info(f"Marketaux termine: {total} news, {len(errors)} erreurs")
        return {"total_news": total, "errors": errors}
