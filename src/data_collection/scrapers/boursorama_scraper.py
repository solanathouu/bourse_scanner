"""Scraper Boursorama pour prix historiques de tickers delistes.

Utilise l'endpoint JSON non-documente de Boursorama pour recuperer
les prix EOD (End of Day) des actions qui ne sont plus sur yfinance.
"""

from datetime import datetime

import requests
from loguru import logger

from src.core.database import Database

BOURSORAMA_EOD_URL = (
    "https://www.boursorama.com/bourse/action/graph/ws/GetTicksEOD"
)

# Mapping ticker Yahoo -> symbole Boursorama
BOURSORAMA_SYMBOLS = {
    "2CRSI.PA": "1rP2CRSI",
    "ALTBG.PA": "1rPALTBG",
    "AFYREN.PA": "1rPAFYREN",
}

# Tickers delistes connus (pas de prix sur yfinance)
DELISTED_TICKERS = ["2CRSI.PA", "ALTBG.PA"]

REQUEST_TIMEOUT = 30
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.boursorama.com/",
}


class BoursoramaPriceScraper:
    """Scrape les prix historiques depuis Boursorama."""

    def __init__(self, db: Database):
        self.db = db

    def _get_boursorama_symbol(self, ticker: str) -> str | None:
        """Mappe un ticker Yahoo vers un symbole Boursorama.

        Returns:
            Symbole Boursorama ou None si pas de mapping.
        """
        return BOURSORAMA_SYMBOLS.get(ticker)

    def _fetch_prices(self, symbol: str, length: int = 730) -> list[dict]:
        """Recupere les prix EOD depuis Boursorama.

        Args:
            symbol: Symbole Boursorama (ex: "1rP2CRSI").
            length: Nombre de jours d'historique (max ~730).

        Returns:
            Liste de dicts {date, open, high, low, close, volume}.
        """
        params = {
            "symbol": symbol,
            "length": length,
            "period": 0,
        }

        try:
            response = requests.get(
                BOURSORAMA_EOD_URL, params=params,
                headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Boursorama erreur pour {symbol}: {e}")
            return []

        # Le format de reponse est {"d": {"QuoteTab": [...]}}
        ticks = data.get("d", {}).get("QuoteTab", [])
        if not ticks:
            logger.warning(f"Boursorama: pas de donnees pour {symbol}")
            return []

        prices = []
        for tick in ticks:
            try:
                # Boursorama retourne les dates en timestamp Unix (ms)
                ts = tick.get("d")
                if ts:
                    # Format: "/Date(1706745600000)/" ou timestamp direct
                    if isinstance(ts, str) and "Date(" in ts:
                        ts_ms = int(ts.split("(")[1].split(")")[0])
                        date_str = datetime.fromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")
                    elif isinstance(ts, (int, float)):
                        date_str = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d")
                    else:
                        date_str = str(ts)[:10]
                else:
                    continue

                prices.append({
                    "date": date_str,
                    "open": tick.get("o"),
                    "high": tick.get("h"),
                    "low": tick.get("l"),
                    "close": tick.get("c"),
                    "volume": tick.get("v", 0),
                })
            except Exception as e:
                logger.debug(f"Skip tick invalide: {e}")
                continue

        return prices

    def collect_for_ticker(self, ticker: str) -> int:
        """Collecte les prix Boursorama pour un ticker.

        Returns:
            Nombre de prix recuperes et inseres.
        """
        symbol = self._get_boursorama_symbol(ticker)
        if not symbol:
            logger.warning(f"Pas de symbole Boursorama pour {ticker}")
            return 0

        raw_prices = self._fetch_prices(symbol)
        if not raw_prices:
            return 0

        # Convertir au format de la table prices
        price_records = []
        for p in raw_prices:
            price_records.append({
                "ticker": ticker,
                "date": p["date"],
                "open": p["open"],
                "high": p["high"],
                "low": p["low"],
                "close": p["close"],
                "volume": p["volume"],
            })

        self.db.insert_prices_batch(price_records)
        logger.info(f"Boursorama: {ticker} -> {len(price_records)} prix")
        return len(price_records)

    def collect_delisted(self) -> dict:
        """Collecte les prix pour tous les tickers delistes connus.

        Returns:
            Dict {"total_prices": int, "tickers": dict, "errors": list}
        """
        total = 0
        tickers_result = {}
        errors = []

        for ticker in DELISTED_TICKERS:
            try:
                count = self.collect_for_ticker(ticker)
                tickers_result[ticker] = count
                total += count
            except Exception as e:
                errors.append({"ticker": ticker, "error": str(e)})
                logger.error(f"Boursorama erreur {ticker}: {e}")

        logger.info(f"Boursorama delistes: {total} prix, "
                     f"{len(tickers_result)} tickers")
        return {
            "total_prices": total,
            "tickers": tickers_result,
            "errors": errors,
        }
