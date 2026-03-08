"""Formatage des messages d'alerte Telegram.

Produit des messages HTML lisibles avec score, catalyseur,
indicateurs techniques et disclaimer.
"""

from html import escape


class AlertFormatter:
    """Formate les signaux en messages Telegram HTML."""

    def format_signal(self, signal: dict) -> str:
        """Formate un signal en message Telegram HTML."""
        ticker = escape(signal["ticker"])
        name = escape(signal.get("name", ticker))
        score = signal["score"]
        score_pct = int(score * 100)

        features = signal.get("features", {})
        price = signal.get("current_price")
        price_str = f"{price:.2f}" if price else "N/A"

        lines = [
            f"<b>SIGNAL — {name} ({ticker})</b>",
            f"Score: {score_pct}% | Prix: {price_str} EUR",
            "",
        ]

        # Catalyseur
        cat_type = signal.get("catalyst_type", "UNKNOWN")
        if cat_type and cat_type not in ("TECHNICAL", "UNKNOWN"):
            news_title = signal.get("catalyst_news_title") or ""
            lines.append(f"Catalyseur: {escape(cat_type)}")
            if news_title:
                lines.append(f'"{escape(news_title)}"')
            lines.append("")

        # Feedback — win rate historique pour ce type de catalyseur
        catalyst_stats = signal.get("catalyst_stats", {})
        cat_stat = catalyst_stats.get(cat_type)
        if cat_stat and cat_stat.get("total", 0) >= 3:
            wr = cat_stat["win_rate"]
            total = cat_stat["total"]
            wins = cat_stat.get("wins", 0)
            lines.append(
                f"Feedback: {cat_type} = {wr:.0%} WR "
                f"({wins}/{total} signaux passes)"
            )

        # Technique
        tech_summary = signal.get("technical_summary", "")
        if tech_summary and tech_summary != "N/A":
            lines.append(f"Technique: {escape(tech_summary)}")

        # Fondamentaux
        fund_parts = []
        pe = features.get("pe_ratio")
        if pe and pe > 0:
            fund_parts.append(f"PE {pe:.1f}")
        analysts = features.get("analyst_count")
        if analysts and analysts > 0:
            reco_label = self._reco_label(features.get("recommendation_score", 0))
            fund_parts.append(f"Analystes: {analysts} ({reco_label})")
        if fund_parts:
            lines.append(f"Fondamentaux: {' | '.join(fund_parts)}")

        lines.append("")
        lines.append("<i>Aide a la decision, pas un signal d'achat automatique.</i>")

        return "\n".join(lines)

    def format_daily_summary(self, signals: list[dict]) -> str:
        """Formate un resume quotidien des signaux emis."""
        if not signals:
            return "<b>Resume quotidien</b>\nAucun signal emis aujourd'hui."

        lines = [
            f"<b>Resume quotidien — {len(signals)} signal(s)</b>",
            "",
        ]

        for s in signals:
            name = escape(s.get("name", s["ticker"]))
            score_pct = int(s["score"] * 100)
            cat = escape(s.get("catalyst_type", ""))
            lines.append(f"- {name}: {score_pct}% ({cat})")

        lines.append("")
        lines.append("<i>Aide a la decision, pas un signal d'achat automatique.</i>")

        return "\n".join(lines)

    def _reco_label(self, score: int) -> str:
        """Convertit le score recommendation en label lisible."""
        labels = {5: "strong buy", 4: "buy", 3: "hold", 2: "sell", 1: "strong sell"}
        return labels.get(score, "N/A")
