"""Collecte de news via flux RSS de sites financiers francais."""

import time
from datetime import datetime

import feedparser
from loguru import logger

from src.core.database import Database
from src.data_collection.ticker_mapper import TickerMapper

# Flux RSS financiers — sources testees et fonctionnelles (fev 2026)
RSS_FEEDS = {
    # Google News FR — large couverture bourse/actions
    "google_news_bourse": "https://news.google.com/rss/search?q=bourse+paris+actions&hl=fr&gl=FR&ceid=FR:fr",
    "google_news_cac40": "https://news.google.com/rss/search?q=CAC+40&hl=fr&gl=FR&ceid=FR:fr",
    # Recherches ciblees par action (les plus tradees)
    "google_news_sanofi": "https://news.google.com/rss/search?q=Sanofi+bourse&hl=fr&gl=FR&ceid=FR:fr",
    "google_news_schneider": "https://news.google.com/rss/search?q=Schneider+Electric+bourse&hl=fr&gl=FR&ceid=FR:fr",
    "google_news_air_liquide": "https://news.google.com/rss/search?q=Air+Liquide+bourse&hl=fr&gl=FR&ceid=FR:fr",
    "google_news_dbv": "https://news.google.com/rss/search?q=DBV+Technologies+bourse&hl=fr&gl=FR&ceid=FR:fr",
    "google_news_nanobiotix": "https://news.google.com/rss/search?q=Nanobiotix+bourse&hl=fr&gl=FR&ceid=FR:fr",
    "google_news_valneva": "https://news.google.com/rss/search?q=Valneva+bourse&hl=fr&gl=FR&ceid=FR:fr",
    "google_news_inventiva": "https://news.google.com/rss/search?q=Inventiva+bourse&hl=fr&gl=FR&ceid=FR:fr",
    # Sites financiers avec RSS actifs
    "yahoo_fr": "https://fr.finance.yahoo.com/rss/",
    "investing_fr": "https://fr.investing.com/rss/news.rss",
    "la_tribune": "https://www.latribune.fr/rss/rubriques/bourse.html",
    "seeking_alpha": "https://seekingalpha.com/market_currents.xml",
}

DELAY_BETWEEN_FEEDS = 2  # secondes


class RSSCollector:
    """Collecte les news depuis les flux RSS financiers francais."""

    def __init__(self, db: Database):
        self.db = db
        self.mapper = TickerMapper()
        self._action_names = list(self.mapper.get_all_mappings().keys())

    def _match_ticker(self, text: str) -> str | None:
        """Cherche si le texte mentionne une de nos actions tradees.

        Returns:
            Ticker Yahoo si match, None sinon.
        """
        text_upper = text.upper()
        for name in self._action_names:
            # Chercher le nom complet ou les mots significatifs
            if name.upper() in text_upper:
                return self.mapper.get_ticker(name)
            # Pour les noms longs, chercher le premier mot (>3 chars)
            first_word = name.split()[0]
            if len(first_word) > 3 and first_word.upper() in text_upper:
                return self.mapper.get_ticker(name)
        return None

    def _parse_entry(self, entry: dict, feed_name: str) -> dict | None:
        """Transforme une entree RSS en dict pour la base.

        Returns:
            dict si l'article mentionne une de nos actions, None sinon.
        """
        title = entry.get("title", "")
        description = entry.get("summary", entry.get("description", ""))
        text = f"{title} {description}"

        ticker = self._match_ticker(text)
        if not ticker:
            return None

        # Parser la date
        published_at = ""
        raw_date = entry.get("published", entry.get("updated", ""))
        if raw_date:
            parsed = entry.get("published_parsed", entry.get("updated_parsed"))
            if parsed:
                try:
                    published_at = time.strftime("%Y-%m-%d %H:%M:%S", parsed)
                except Exception:
                    published_at = raw_date

        return {
            "ticker": ticker,
            "title": title,
            "source": feed_name,
            "url": entry.get("link", ""),
            "published_at": published_at,
            "description": description[:500] if description else "",
            "sentiment": None,
            "source_api": f"rss_{feed_name}",
        }

    def collect_feed(self, feed_name: str, feed_url: str) -> int:
        """Collecte les articles d'un flux RSS.

        Returns:
            Nombre de news pertinentes inserees.
        """
        logger.info(f"RSS: collecte {feed_name} ({feed_url})")

        try:
            feed = feedparser.parse(feed_url)
        except Exception as e:
            logger.error(f"RSS erreur parsing {feed_name}: {e}")
            return 0

        if feed.bozo and not feed.entries:
            logger.warning(f"RSS: {feed_name} inaccessible ou vide")
            return 0

        news_list = []
        for entry in feed.entries:
            article = self._parse_entry(entry, feed_name)
            if article and article["url"]:
                news_list.append(article)

        if not news_list:
            logger.info(f"RSS: {feed_name} — 0 articles sur nos actions")
            return 0

        self.db.insert_news_batch(news_list)
        logger.info(f"RSS: {feed_name} — {len(news_list)} articles pertinents")
        return len(news_list)

    def collect_all(self) -> dict:
        """Collecte les articles de tous les flux RSS configures.

        Returns:
            Dict {"total_news": int, "feeds_ok": int, "feeds_error": int, "errors": list}
        """
        total = 0
        feeds_ok = 0
        errors = []

        for feed_name, feed_url in RSS_FEEDS.items():
            try:
                count = self.collect_feed(feed_name, feed_url)
                total += count
                feeds_ok += 1
                time.sleep(DELAY_BETWEEN_FEEDS)
            except Exception as e:
                errors.append({"feed": feed_name, "error": str(e)})
                logger.error(f"RSS erreur {feed_name}: {e}")

        logger.info(
            f"RSS termine: {total} news, {feeds_ok}/{len(RSS_FEEDS)} feeds OK, "
            f"{len(errors)} erreurs"
        )
        return {
            "total_news": total,
            "feeds_ok": feeds_ok,
            "feeds_error": len(errors),
            "errors": errors,
        }
