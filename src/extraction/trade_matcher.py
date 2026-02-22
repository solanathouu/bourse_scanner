"""Reconstruction des trades complets à partir des exécutions individuelles.

Logique FIFO : les premières actions achetées sont les premières vendues.
Regroupe par ISIN, puis applique le matching chronologique.
"""

from datetime import datetime
from loguru import logger


class TradeMatcher:
    """Reconstruit les trades complets (achat->vente) à partir des exécutions."""

    def match_trades(self, executions: list[dict]) -> list[dict]:
        """
        À partir d'une liste d'exécutions (potentiellement multi-actions),
        reconstruit les trades complets.

        Approche FIFO par ISIN : les achats sont consommés dans l'ordre chronologique.

        Returns:
            Liste de trades complets (dicts compatibles avec table trades_complets)
        """
        # Regrouper par ISIN
        by_isin: dict[str, list[dict]] = {}
        for ex in executions:
            isin = ex.get("isin") or ex.get("nom_action", "UNKNOWN")
            if isin not in by_isin:
                by_isin[isin] = []
            by_isin[isin].append(ex)

        all_trades = []
        for isin, execs in by_isin.items():
            trades = self._match_single_isin(execs)
            all_trades.extend(trades)

        closed = sum(1 for t in all_trades if t["statut"] == "CLOTURE")
        opened = sum(1 for t in all_trades if t["statut"] == "OUVERT")
        logger.info(
            f"Matching terminé: {len(all_trades)} trades reconstitués "
            f"({closed} clôturés, {opened} ouverts)"
        )

        return all_trades

    def _match_single_isin(self, executions: list[dict]) -> list[dict]:
        """Reconstruit les trades pour un seul ISIN (méthode FIFO)."""
        # Trier par date+heure
        sorted_execs = sorted(
            executions,
            key=lambda e: f"{e['date_execution']} {e['heure_execution']}"
        )

        # File FIFO des achats non encore vendus
        # Chaque élément = [exec_dict, quantite_restante]
        buy_queue: list[list] = []
        trades = []

        for ex in sorted_execs:
            if ex["sens"] == "ACHAT":
                buy_queue.append([ex, ex["quantite"]])
            elif ex["sens"] == "VENTE":
                remaining_to_sell = ex["quantite"]

                while remaining_to_sell > 0 and buy_queue:
                    buy_exec, buy_remaining = buy_queue[0]
                    matched_qty = min(remaining_to_sell, buy_remaining)

                    trade = self._create_trade(
                        buy_exec=buy_exec,
                        sell_exec=ex,
                        quantite=matched_qty,
                    )
                    trades.append(trade)

                    remaining_to_sell -= matched_qty
                    buy_remaining -= matched_qty

                    if buy_remaining == 0:
                        buy_queue.pop(0)
                    else:
                        buy_queue[0] = [buy_exec, buy_remaining]

                if remaining_to_sell > 0:
                    logger.warning(
                        f"Vente de {remaining_to_sell} {ex['nom_action']} "
                        f"sans achat correspondant le {ex['date_execution']}"
                    )

        # Les achats restants dans la queue = trades ouverts
        for buy_exec, remaining_qty in buy_queue:
            trade = self._create_trade(
                buy_exec=buy_exec,
                sell_exec=None,
                quantite=remaining_qty,
            )
            trades.append(trade)

        return trades

    def _create_trade(
        self,
        buy_exec: dict,
        sell_exec: dict | None,
        quantite: int,
    ) -> dict:
        """Crée un dict de trade complet."""
        trade = {
            "isin": buy_exec.get("isin"),
            "nom_action": buy_exec["nom_action"],
            "date_achat": f"{buy_exec['date_execution']} {buy_exec['heure_execution']}",
            "prix_achat": buy_exec["prix_unitaire"],
            "quantite": quantite,
        }

        if sell_exec:
            trade["date_vente"] = f"{sell_exec['date_execution']} {sell_exec['heure_execution']}"
            trade["prix_vente"] = sell_exec["prix_unitaire"]
            trade["statut"] = "CLOTURE"

            # Rendement brut %
            trade["rendement_brut_pct"] = round(
                (sell_exec["prix_unitaire"] - buy_exec["prix_unitaire"])
                / buy_exec["prix_unitaire"] * 100,
                2
            )

            # Frais totaux (proportionnels à la quantité matchée)
            buy_commission = buy_exec.get("commission", 0) * (quantite / buy_exec["quantite"])
            buy_frais = buy_exec.get("frais", 0) * (quantite / buy_exec["quantite"])
            sell_commission = sell_exec.get("commission", 0) * (quantite / sell_exec["quantite"])
            sell_frais = sell_exec.get("frais", 0) * (quantite / sell_exec["quantite"])
            trade["frais_totaux"] = round(
                buy_commission + buy_frais + sell_commission + sell_frais, 2
            )

            # Rendement net %
            montant_achat = buy_exec["prix_unitaire"] * quantite
            montant_vente = sell_exec["prix_unitaire"] * quantite
            profit_net = montant_vente - montant_achat - trade["frais_totaux"]
            trade["rendement_net_pct"] = round(profit_net / montant_achat * 100, 2)

            # Durée en jours
            dt_achat = datetime.strptime(trade["date_achat"], "%Y-%m-%d %H:%M:%S")
            dt_vente = datetime.strptime(trade["date_vente"], "%Y-%m-%d %H:%M:%S")
            trade["duree_jours"] = round(
                (dt_vente - dt_achat).total_seconds() / 86400, 1
            )
        else:
            trade["date_vente"] = None
            trade["prix_vente"] = None
            trade["statut"] = "OUVERT"
            trade["rendement_brut_pct"] = None
            trade["rendement_net_pct"] = None
            trade["duree_jours"] = None
            trade["frais_totaux"] = round(
                buy_exec.get("commission", 0) * (quantite / buy_exec["quantite"])
                + buy_exec.get("frais", 0) * (quantite / buy_exec["quantite"]),
                2
            )

        return trade
