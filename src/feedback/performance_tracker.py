"""Tracking des performances et generation de regles adaptatives."""

import json
from datetime import datetime
from loguru import logger

from src.core.database import Database


class PerformanceTracker:
    """Analyse les performances et genere des regles de filtrage."""

    def __init__(self, db: Database, base_threshold: float = 0.75, min_samples: int = 5):
        self.db = db
        self.base_threshold = base_threshold
        self.min_samples = min_samples

    def win_rate_by_catalyst(self) -> dict:
        """Calculate win rate per catalyst type from all reviews.

        Returns: {"EARNINGS": {"wins": 2, "total": 3, "win_rate": 0.67}, ...}
        """
        reviews = self.db.get_signal_reviews()
        by_cat = {}
        for r in reviews:
            cat = r.get("catalyst_type") or "UNKNOWN"
            if cat not in by_cat:
                by_cat[cat] = {"wins": 0, "total": 0}
            by_cat[cat]["total"] += 1
            if r["outcome"] == "WIN":
                by_cat[cat]["wins"] += 1
        for cat, data in by_cat.items():
            data["win_rate"] = data["wins"] / data["total"] if data["total"] > 0 else 0.0
        return by_cat

    def compute_adaptive_threshold(self) -> float:
        """Compute adaptive threshold based on global win rate.

        - Win rate < 40% -> threshold + 0.04
        - Win rate 40-50% -> threshold + 0.02
        - Win rate >= 50% -> threshold unchanged
        - Max 0.95
        """
        stats = self.db.get_review_stats()
        if stats["total"] < self.min_samples:
            return self.base_threshold
        win_rate = stats["wins"] / stats["total"] if stats["total"] > 0 else 0.0
        threshold = self.base_threshold
        if win_rate < 0.40:
            threshold += 0.04
        elif win_rate < 0.50:
            threshold += 0.02
        return min(round(threshold, 2), 0.95)

    def generate_filter_rules(self) -> list[dict]:
        """Generate filter rules based on failure patterns.

        - EXCLUDE_CATALYST if a catalyst type has < 30% win rate on min_samples+ samples
        - MAX_SIGNALS_PER_DAY always set to 2
        Deactivates all existing rules first, then creates new ones.
        """
        rates = self.win_rate_by_catalyst()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_rules = []

        # Deactivate old rules
        existing = self.db.get_active_filter_rules()
        for rule in existing:
            self.db.deactivate_filter_rule(rule["id"])

        # Generate EXCLUDE_CATALYST rules
        for cat, data in rates.items():
            if data["total"] < self.min_samples:
                continue
            if data["win_rate"] < 0.30:
                rule = {
                    "rule_type": "EXCLUDE_CATALYST",
                    "rule_json": json.dumps({"catalyst_type": cat}),
                    "win_rate": round(data["win_rate"], 4),
                    "sample_size": data["total"],
                    "created_at": now,
                    "active": 1,
                }
                self.db.insert_filter_rule(rule)
                new_rules.append(rule)
                logger.info(
                    f"Regle EXCLUDE_CATALYST: {cat} "
                    f"(win_rate={data['win_rate']:.0%}, n={data['total']})"
                )

        # Always add MAX_SIGNALS_PER_DAY
        rule_max = {
            "rule_type": "MAX_SIGNALS_PER_DAY",
            "rule_json": json.dumps({"max": 2}),
            "win_rate": None,
            "sample_size": None,
            "created_at": now,
            "active": 1,
        }
        self.db.insert_filter_rule(rule_max)
        new_rules.append(rule_max)

        logger.info(f"{len(new_rules)} regles de filtrage generees")
        return new_rules

    def get_daily_summary(self, reviews: list[dict]) -> str:
        """Format daily review summary as HTML for Telegram."""
        if not reviews:
            return "<b>Review J+3</b>\nAucun signal a reviewer aujourd'hui."

        date = reviews[0].get("review_date", "")
        lines = [f"<b>Review J+3 -- {date}</b>", ""]
        for r in reviews:
            ticker = r["ticker"]
            perf = r["performance_pct"]
            outcome = r["outcome"]
            if outcome == "WIN":
                icon, detail = "OK", f"{r.get('catalyst_type', '')} confirme"
            elif outcome == "LOSS":
                icon, detail = "X", r.get("failure_reason", "Echec")
            else:
                icon, detail = "-", "Performance insuffisante"
            lines.append(f"[{icon}] {ticker}: {perf:+.1f}% -- {detail}")

        stats = self.db.get_review_stats()
        if stats["total"] > 0:
            wr = stats["wins"] / stats["total"]
            lines.extend(["", f"Win rate global: {wr:.0%} ({stats['wins']}/{stats['total']})"])

        threshold = self.compute_adaptive_threshold()
        lines.append(f"Seuil actuel: {threshold:.2f}")
        return "\n".join(lines)

    def get_weekly_summary(self, date_start: str, date_end: str) -> str:
        """Format weekly summary as HTML for Telegram."""
        reviews = self.db.get_reviews_in_period(date_start, date_end)
        lines = [f"<b>Bilan semaine -- {date_start} au {date_end}</b>", ""]

        if not reviews:
            lines.append("Aucune review cette semaine.")
            return "\n".join(lines)

        wins = [r for r in reviews if r["outcome"] == "WIN"]
        losses = [r for r in reviews if r["outcome"] == "LOSS"]
        neutrals = [r for r in reviews if r["outcome"] == "NEUTRAL"]
        perfs = [r["performance_pct"] for r in reviews]

        lines.append(
            f"Signaux: {len(reviews)} | "
            f"WIN: {len(wins)} | LOSS: {len(losses)} | NEUTRAL: {len(neutrals)}"
        )

        if perfs:
            avg_perf = sum(perfs) / len(perfs)
            best = max(reviews, key=lambda r: r["performance_pct"])
            worst = min(reviews, key=lambda r: r["performance_pct"])
            wr = len(wins) / len(reviews)
            lines.append(f"Win rate: {wr:.0%} | Perf moyenne: {avg_perf:+.1f}%")
            lines.append(
                f"Meilleur: {best['ticker']} {best['performance_pct']:+.1f}% | "
                f"Pire: {worst['ticker']} {worst['performance_pct']:+.1f}%"
            )

        rules = self.db.get_active_filter_rules()
        if rules:
            lines.extend(["", "Regles actives:"])
            for rule in rules:
                data = json.loads(rule["rule_json"])
                if rule["rule_type"] == "EXCLUDE_CATALYST":
                    lines.append(
                        f"  - Exclure {data['catalyst_type']} "
                        f"(win_rate={rule['win_rate']:.0%})"
                    )
                elif rule["rule_type"] == "MAX_SIGNALS_PER_DAY":
                    lines.append(f"  - Max {data['max']} signaux/jour")

        threshold = self.compute_adaptive_threshold()
        lines.append(f"\nSeuil adaptatif: {threshold:.2f}")

        active_model = self.db.get_active_model_version()
        if active_model:
            lines.append(f"Modele actif: {active_model['version']}")

        return "\n".join(lines)
