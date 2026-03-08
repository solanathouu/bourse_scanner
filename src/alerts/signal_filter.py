"""Filtrage intelligent des signaux avant envoi d'alerte.

Applique un seuil de score, un cooldown par ticker, et
verifie que le marche est ouvert (horaires Paris).
Supporte le filtrage adaptatif via regles dynamiques en BDD.
"""

import json
from datetime import datetime, timedelta

import pytz
from loguru import logger

from src.core.database import Database


PARIS_TZ = pytz.timezone("Europe/Paris")


class SignalFilter:
    """Filtre les signaux selon seuil, cooldown et horaires marche."""

    def __init__(self, db: Database, config: dict):
        self.db = db
        self.threshold = config.get("threshold", 0.75)
        self.cooldown_hours = config.get("cooldown_hours", 24)
        self.market_open = config.get("market_open", "09:00")
        self.market_close = config.get("market_close", "17:30")

    def filter_signals(self, signals: list[dict]) -> list[dict]:
        """Filtre une liste de signaux et retourne ceux a envoyer.

        Applique seuil adaptatif + cooldown ticker.
        Pas de blocage par catalyst type — on veut garder le flux
        pour que le feedback loop ait des donnees.
        """
        rules = self.db.get_active_filter_rules()
        adaptive_threshold = self._get_adaptive_threshold(rules)
        effective_threshold = adaptive_threshold if adaptive_threshold is not None else self.threshold

        filtered = []
        for signal in signals:
            if signal.get("score", 0) < effective_threshold:
                logger.debug(
                    f"Signal {signal['ticker']} rejete: "
                    f"score {signal.get('score', 0):.2f} < {effective_threshold}"
                )
                continue
            if not self._passes_cooldown(signal):
                continue
            filtered.append(signal)

        logger.info(
            f"SignalFilter: {len(filtered)}/{len(signals)} retenus "
            f"(seuil={effective_threshold})"
        )
        return filtered

    def _get_adaptive_threshold(self, rules: list[dict]) -> float | None:
        """Extract adaptive threshold from ADAPTIVE_THRESHOLD rule."""
        for rule in rules:
            if rule["rule_type"] == "ADAPTIVE_THRESHOLD":
                data = json.loads(rule["rule_json"])
                return data.get("threshold")
        return None

    def _passes_threshold(self, signal: dict) -> bool:
        """Verifie que le score depasse le seuil."""
        score = signal.get("score", 0)
        if score < self.threshold:
            logger.debug(
                f"Signal {signal['ticker']} rejete: "
                f"score {score:.2f} < seuil {self.threshold}"
            )
            return False
        return True

    def _passes_cooldown(self, signal: dict) -> bool:
        """Verifie qu'on n'a pas deja envoye un signal pour ce ticker recemment."""
        ticker = signal["ticker"]
        latest = self.db.get_latest_signal(ticker)
        if latest is None:
            return True

        sent_at = latest.get("sent_at")
        if not sent_at:
            return True

        try:
            last_sent = datetime.strptime(sent_at, "%Y-%m-%d %H:%M:%S")
            now = datetime.now()
            delta = now - last_sent
            if delta < timedelta(hours=self.cooldown_hours):
                logger.debug(
                    f"Signal {ticker} en cooldown: "
                    f"dernier signal il y a {delta.total_seconds() / 3600:.1f}h"
                )
                return False
        except ValueError:
            pass

        return True

    def is_market_hours(self) -> bool:
        """Verifie que le marche est ouvert (lun-ven, horaires Paris)."""
        now = datetime.now(PARIS_TZ)

        # Weekend
        if now.weekday() >= 5:
            return False

        # Horaires
        open_h, open_m = map(int, self.market_open.split(":"))
        close_h, close_m = map(int, self.market_close.split(":"))

        market_open = now.replace(hour=open_h, minute=open_m, second=0)
        market_close = now.replace(hour=close_h, minute=close_m, second=0)

        return market_open <= now <= market_close

    def record_signal(self, signal: dict):
        """Enregistre un signal en BDD apres envoi."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.db.insert_signal({
            "ticker": signal["ticker"],
            "date": signal["date"],
            "score": signal["score"],
            "signal_price": signal.get("current_price"),
            "catalyst_type": signal.get("catalyst_type"),
            "catalyst_news_title": signal.get("catalyst_news_title"),
            "features_json": signal.get("features_json"),
            "sent_at": now,
        })
        logger.info(f"Signal enregistre: {signal['ticker']} score={signal['score']:.2f}")
