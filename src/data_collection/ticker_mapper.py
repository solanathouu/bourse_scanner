"""Mapping entre noms d'actions (de la base) et tickers Yahoo Finance."""

from loguru import logger


class TickerNotFoundError(Exception):
    """Leve quand un nom d'action n'a pas de ticker connu."""
    pass


# Mapping nom_action (tel que dans trades_complets) -> ticker Yahoo Finance
# Verifie manuellement sur finance.yahoo.com
TICKER_MAP = {
    "2CRSI": "2CRSI.PA",
    "AB SCIENCE": "AB.PA",
    "ADOCIA": "ADOC.PA",
    "AFYREN": "AFYREN.PA",
    "AIR LIQUIDE": "AI.PA",
    "AMUNDI ETF MSCI WR": "CW8.PA",
    "BNPP SP500 EUR C": "ESE.PA",
    "DBV TECHNOLOGIES": "DBV.PA",
    "EXAIL TECHNOLOGIES": "EXA.PA",
    "GENFIT": "GNFT.PA",
    "INVENTIVA": "IVA.PA",
    "KALRAY": "ALKAL.PA",
    "MAUREL ET PROM": "MAU.PA",
    "MEMSCAP REGROUPEES": "MEMS.PA",
    "NANOBIOTIX": "NANO.PA",
    "SANOFI": "SAN.PA",
    "SCHNEIDER ELECTRIC SE": "SU.PA",
    "TECHNIP ENERGIES": "TE.PA",
    "THE BLOKCHAIN GP": "ALTBG.PA",
    "VALNEVA": "VLA.PA",
}


class TickerMapper:
    """Convertit les noms d'actions de la base vers les tickers Yahoo Finance."""

    def __init__(self):
        self._map = TICKER_MAP.copy()

    def _clean_name(self, nom_action: str) -> str:
        """Nettoie le nom (retire le * en prefixe si present)."""
        return nom_action.lstrip("* ").strip()

    def get_ticker(self, nom_action: str) -> str:
        """Retourne le ticker Yahoo Finance pour un nom d'action."""
        clean = self._clean_name(nom_action)
        if clean not in self._map:
            raise TickerNotFoundError(
                f"Pas de ticker connu pour '{nom_action}' (nettoye: '{clean}')"
            )
        return self._map[clean]

    def get_all_mappings(self) -> dict[str, str]:
        """Retourne tous les mappings nom -> ticker."""
        return self._map.copy()
