"""Formatage des messages d'alerte Telegram.

Produit des messages Markdown lisibles avec score, catalyseur,
indicateurs techniques et disclaimer.
"""


class AlertFormatter:
    """Formate les signaux en messages Telegram Markdown."""

    def format_signal(self, signal: dict) -> str:
        """Formate un signal en message Telegram."""
        ticker = signal["ticker"]
        name = signal.get("name", ticker)
        score = signal["score"]
        score_pct = int(score * 100)

        features = signal.get("features", {})
        price = features.get("close") or signal.get("current_price", "N/A")

        lines = [
            f"*SIGNAL — {name} ({ticker})*",
            f"Score: {score_pct}% | Prix: {price} EUR",
            "",
        ]

        # Catalyseur
        cat_type = signal.get("catalyst_type", "UNKNOWN")
        if cat_type and cat_type not in ("TECHNICAL", "UNKNOWN"):
            news_title = signal.get("catalyst_news_title") or ""
            lines.append(f"Catalyseur: {cat_type}")
            if news_title:
                lines.append(f'"{news_title}"')
            lines.append("")

        # Technique
        tech_summary = signal.get("technical_summary", "")
        if tech_summary and tech_summary != "N/A":
            lines.append(f"Technique: {tech_summary}")

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
        lines.append("_Aide a la decision, pas un signal d'achat automatique._")

        return "\n".join(lines)

    def format_daily_summary(self, signals: list[dict]) -> str:
        """Formate un resume quotidien des signaux emis."""
        if not signals:
            return "*Resume quotidien*\nAucun signal emis aujourd'hui."

        lines = [
            f"*Resume quotidien — {len(signals)} signal(s)*",
            "",
        ]

        for s in signals:
            name = s.get("name", s["ticker"])
            score_pct = int(s["score"] * 100)
            cat = s.get("catalyst_type", "")
            lines.append(f"- {name}: {score_pct}% ({cat})")

        lines.append("")
        lines.append("_Aide a la decision, pas un signal d'achat automatique._")

        return "\n".join(lines)

    def _reco_label(self, score: int) -> str:
        """Convertit le score recommendation en label lisible."""
        labels = {5: "strong buy", 4: "buy", 3: "hold", 2: "sell", 1: "strong sell"}
        return labels.get(score, "N/A")
