"""Collecteur de carnets d'ordres via Boursorama.

Scrape la page de cotation Boursorama pour extraire le carnet d'ordres
(5 meilleures offres achat/vente) et calcule des metriques de pression.
"""

import html
import json
import re
import time
from datetime import datetime

import requests
from loguru import logger

from src.core.database import Database

REQUEST_TIMEOUT = 15
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Referer": "https://www.boursorama.com/",
}

# Mapping ticker Yahoo -> symbole Boursorama (1rP + base)
BOURSORAMA_ORDERBOOK_SYMBOLS = {
    "2CRSI.PA": "1rP2CRSI",
    "AB.PA": "1rPAB",
    "ADOC.PA": "1rPADOC",
    "AFYREN.PA": "1rPAFYREN",
    "AVT.PA": "1rPAVT",
    "ALBIO.PA": "1rPALBIO",
    "ALCAP.PA": "1rPALCAP",
    "ALCJ.PA": "1rPALCJ",
    "DBV.PA": "1rPDBV",
    "EXA.PA": "1rPEXA",
    "GNFT.PA": "1rPGNFT",
    "IVA.PA": "1rPIVA",
    "ALKAL.PA": "1rPALKAL",
    "MAU.PA": "1rPMAU",
    "ALMDT.PA": "1rPALMDT",
    "MEMS.PA": "1rPMEMS",
    "NANO.PA": "1rPNANO",
    "POXEL.PA": "1rPPOXEL",
    "RXL.PA": "1rPRXL",
    "SAN.PA": "1rPSAN",
    "SU.PA": "1rPSU",
    "ALSEN.PA": "1rPALSEN",
    "SOI.PA": "1rPSOI",
    "UBI.PA": "1rPUBI",
    "ALVAL.PA": "1rPALVAL",
    "VLA.PA": "1rPVLA",
    "DG.PA": "1rPDG",
    "WLN.PA": "1rPWLN",
    "ALTBG.PA": "1rPALTBG",
    "CW8.PA": "1rPCW8",
    "ESE.PA": "1rPESE",
}


class OrderBookCollector:
    """Collecte les carnets d'ordres via Boursorama."""

    def __init__(self, db: Database):
        self.db = db

    def _get_boursorama_symbol(self, ticker: str) -> str | None:
        """Mappe un ticker Yahoo vers un symbole Boursorama."""
        return BOURSORAMA_ORDERBOOK_SYMBOLS.get(ticker)

    def _fetch_orderbook(self, symbol: str) -> dict | None:
        """Recupere le carnet d'ordres depuis la page Boursorama.

        Parse le HTML pour trouver data-ist-orderbook data-ist-init="..."
        qui contient le carnet d'ordres en JSON.
        """
        url = f"https://www.boursorama.com/cours/{symbol}/"

        try:
            response = requests.get(
                url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Boursorama orderbook erreur HTTP {symbol}: {e}")
            return None

        text = response.text
        match = re.search(
            r'data-ist-orderbook\s+data-ist-init="([^"]+)"', text,
        )
        if not match:
            logger.debug(f"Pas de carnet d'ordres pour {symbol}")
            return None

        try:
            raw = html.unescape(match.group(1))
            data = json.loads(raw)
            return data
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Boursorama orderbook JSON invalide {symbol}: {e}")
            return None

    def _parse_orderbook(self, data: dict) -> dict:
        """Extrait les metriques du carnet d'ordres.

        Returns:
            Dict avec best_bid, best_ask, volumes, spread, ratios.
        """
        bids = data.get("bids", [])
        asks = data.get("asks", [])

        best_bid = bids[0].get("price", 0) if bids else 0
        best_ask = asks[0].get("price", 0) if asks else 0

        bid_volumes = [b.get("quantity", 0) or 0 for b in bids]
        ask_volumes = [a.get("quantity", 0) or 0 for a in asks]

        bid_orders = [b.get("orders", 0) or 0 for b in bids]
        ask_orders = [a.get("orders", 0) or 0 for a in asks]

        bid_volume_total = sum(bid_volumes)
        ask_volume_total = sum(ask_volumes)
        bid_orders_total = sum(bid_orders)
        ask_orders_total = sum(ask_orders)

        # Spread en %
        spread_pct = 0.0
        if best_bid > 0 and best_ask > 0:
            spread_pct = round((best_ask - best_bid) / best_bid * 100, 4)

        # Ratio volume bid/ask (>1 = pression acheteuse)
        bid_ask_volume_ratio = 0.0
        if ask_volume_total > 0:
            bid_ask_volume_ratio = round(
                bid_volume_total / ask_volume_total, 4,
            )

        # Concentration top 3 bids
        bid_depth_concentration = 0.0
        if bid_volume_total > 0 and len(bid_volumes) >= 3:
            top3 = sum(bid_volumes[:3])
            bid_depth_concentration = round(top3 / bid_volume_total, 4)
        elif bid_volume_total > 0:
            bid_depth_concentration = 1.0

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bid_volume_total": bid_volume_total,
            "ask_volume_total": ask_volume_total,
            "bid_orders_total": bid_orders_total,
            "ask_orders_total": ask_orders_total,
            "spread_pct": spread_pct,
            "bid_ask_volume_ratio": bid_ask_volume_ratio,
            "bid_depth_concentration": bid_depth_concentration,
        }

    def collect_orderbook(self, ticker: str) -> dict | None:
        """Collecte et stocke le carnet d'ordres pour un ticker.

        Returns:
            Dict du snapshot insere, ou None en cas d'erreur.
        """
        symbol = self._get_boursorama_symbol(ticker)
        if not symbol:
            logger.debug(f"Pas de symbole Boursorama pour {ticker}")
            return None

        data = self._fetch_orderbook(symbol)
        if data is None:
            return None

        metrics = self._parse_orderbook(data)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        snapshot = {
            "ticker": ticker,
            "snapshot_time": now,
            **metrics,
            "raw_json": json.dumps(data),
        }

        self.db.insert_orderbook_snapshot(snapshot)
        logger.debug(
            f"Orderbook {ticker}: spread={metrics['spread_pct']:.2f}% "
            f"ratio={metrics['bid_ask_volume_ratio']:.2f}"
        )
        return snapshot

    def collect_all_watchlist(self, watchlist: list[dict]) -> dict:
        """Collecte le carnet d'ordres pour toute la watchlist.

        Args:
            watchlist: Liste de dicts avec au minimum 'ticker'.

        Returns:
            Dict {"collected": int, "errors": int}
        """
        collected = 0
        errors = 0

        for item in watchlist:
            ticker = item.get("ticker", "")
            if item.get("etf"):
                continue

            try:
                result = self.collect_orderbook(ticker)
                if result:
                    collected += 1
                time.sleep(2)
            except Exception as e:
                errors += 1
                logger.error(f"Orderbook erreur {ticker}: {e}")

        logger.info(
            f"Orderbook watchlist: {collected} collectes, {errors} erreurs"
        )
        return {"collected": collected, "errors": errors}
