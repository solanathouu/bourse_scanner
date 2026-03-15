"""Review des signaux emis a J+3 avec systeme de Take Profit.

Si le prix HIGH touche +4.5% a n'importe quel moment entre J+0 et J+3,
c'est un WIN (Nicolas aurait TP dans la realite). Le close a J+3 sert
pour la performance finale si le TP n'a pas ete touche.
"""

import json
from datetime import datetime, timedelta

from loguru import logger

from src.core.database import Database


class SignalReviewer:
    """Review les signaux a J+3 avec detection de Take Profit."""

    def __init__(self, db: Database, win_threshold: float = 4.5):
        self.db = db
        self.win_threshold = win_threshold

    def review_pending(self, current_date: str | None = None) -> list[dict]:
        """Review tous les signaux en attente (emis il y a 3+ jours).

        Returns list of review dicts that were stored.
        """
        if current_date is None:
            current_date = datetime.now().strftime("%Y-%m-%d")

        pending = self.db.get_pending_signal_reviews(current_date)
        if not pending:
            logger.info("Aucun signal en attente de review")
            return []

        reviews = []
        for signal in pending:
            review = self._review_signal(signal, current_date)
            if review:
                self.db.insert_signal_review(review)
                reviews.append(review)
                tp_tag = " (TP touche!)" if review.get("tp_hit") else ""
                logger.info(
                    f"Review {signal['ticker']}: {review['outcome']} "
                    f"({review['performance_pct']:+.2f}%){tp_tag}"
                )

        logger.info(f"{len(reviews)}/{len(pending)} signaux reviewes")
        return reviews

    def _review_signal(self, signal: dict, current_date: str) -> dict | None:
        """Review un signal avec detection de TP.

        Logique:
        1. Regarder le HIGH de chaque jour entre J+0 et J+3
        2. Si un HIGH atteint +4.5% vs signal_price -> WIN (TP touche)
        3. Sinon, utiliser le close a J+3 pour la performance finale
        """
        ticker = signal["ticker"]
        signal_price = signal.get("signal_price")

        if not signal_price or signal_price <= 0:
            logger.warning(f"Signal {ticker} sans prix, skip review")
            return None

        signal_dt = datetime.strptime(signal["date"], "%Y-%m-%d")
        tp_price = signal_price * (1 + self.win_threshold / 100)

        # Chercher si le TP a ete touche (HIGH >= tp_price) entre J+0 et J+5
        tp_hit, tp_date, max_high = self._check_tp_hit(
            ticker, signal["date"], signal_price,
        )

        # Prix de cloture a J+3 pour la perf finale si pas de TP
        review_price = self._get_review_price(ticker, signal["date"])
        if review_price is None and not tp_hit:
            logger.warning(f"Pas de prix pour {ticker}, skip review")
            return None

        if tp_hit:
            # TP touche — on considere qu'on a vendu au TP
            perf = self.win_threshold
            outcome = "WIN"
            effective_review_price = tp_price
        else:
            effective_review_price = review_price
            perf = self._calculate_performance(signal_price, review_price)
            outcome = self._classify_outcome(perf)

        failure_reason = self._analyze_failure(signal, perf) if outcome == "LOSS" else None
        review_date = (signal_dt + timedelta(days=3)).strftime("%Y-%m-%d")

        return {
            "signal_id": signal["id"],
            "ticker": ticker,
            "signal_date": signal["date"],
            "signal_price": signal_price,
            "review_date": review_date,
            "review_price": round(effective_review_price, 4),
            "performance_pct": round(perf, 2),
            "outcome": outcome,
            "failure_reason": failure_reason,
            "catalyst_type": signal.get("catalyst_type"),
            "features_json": signal.get("features_json"),
            "reviewed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tp_hit": tp_hit,
        }

    def _check_tp_hit(self, ticker: str, signal_date: str,
                       signal_price: float) -> tuple[bool, str | None, float]:
        """Verifie si le TP a ete touche entre J+0 et J+5.

        Regarde le HIGH de chaque jour. Si HIGH >= signal_price * (1 + 4.5%),
        le TP est touche.

        Returns:
            (tp_hit, tp_date, max_high_pct)
        """
        signal_dt = datetime.strptime(signal_date, "%Y-%m-%d")
        # Fenetre J+0 a J+5 (pour couvrir les weekends)
        date_start = signal_date
        date_end = (signal_dt + timedelta(days=5)).strftime("%Y-%m-%d")

        tp_target = signal_price * (1 + self.win_threshold / 100)
        prices = self.db.get_prices(ticker)
        candidates = [p for p in prices if date_start <= p["date"] <= date_end]
        candidates.sort(key=lambda p: p["date"])

        max_high = 0.0
        for p in candidates:
            high = p.get("high", 0)
            if high <= 0:
                continue
            high_pct = (high - signal_price) / signal_price * 100
            max_high = max(max_high, high_pct)

            if high >= tp_target:
                return True, p["date"], max_high

        return False, None, max_high

    def _get_review_price(self, ticker: str, signal_date: str) -> float | None:
        """Get closing price around J+3 (window J+2 to J+5 for weekends)."""
        signal_dt = datetime.strptime(signal_date, "%Y-%m-%d")
        target_date = (signal_dt + timedelta(days=3)).strftime("%Y-%m-%d")
        date_start = (signal_dt + timedelta(days=2)).strftime("%Y-%m-%d")
        date_end = (signal_dt + timedelta(days=5)).strftime("%Y-%m-%d")

        prices = self.db.get_prices(ticker)
        candidates = [p for p in prices if date_start <= p["date"] <= date_end]

        if not candidates:
            return None

        candidates.sort(key=lambda p: abs(
            (datetime.strptime(p["date"], "%Y-%m-%d")
             - datetime.strptime(target_date, "%Y-%m-%d")).days
        ))
        return candidates[0]["close"]

    def _calculate_performance(self, signal_price: float,
                                review_price: float) -> float:
        """Calcule la performance en pourcentage."""
        if signal_price <= 0:
            return 0.0
        return round((review_price - signal_price) / signal_price * 100, 2)

    def _classify_outcome(self, performance_pct: float) -> str:
        """Classifie le resultat: WIN, LOSS ou NEUTRAL."""
        if performance_pct >= self.win_threshold:
            return "WIN"
        elif performance_pct < 0:
            return "LOSS"
        return "NEUTRAL"

    def _analyze_failure(self, signal: dict, performance_pct: float) -> str | None:
        """Analyze why a signal failed. Returns None if not a loss."""
        if performance_pct >= 0:
            return None

        parts = []
        cat_type = signal.get("catalyst_type", "UNKNOWN")
        parts.append(f"Catalyseur {cat_type} non confirme par le marche")

        features_json = signal.get("features_json")
        if features_json:
            try:
                features = json.loads(features_json)

                rsi = features.get("rsi_14")
                if rsi and rsi > 60:
                    parts.append(f"RSI eleve ({rsi:.0f}) = surachat potentiel")

                vol = features.get("volume_ratio_20")
                if vol and vol < 0.5:
                    parts.append(f"Volume faible ({vol:.2f}x) = pas de conviction")

                sentiment = features.get("news_sentiment")
                if sentiment is not None and sentiment < 0.2:
                    parts.append(f"Sentiment faible ({sentiment:.2f})")
            except (json.JSONDecodeError, TypeError):
                pass

        return " | ".join(parts)
