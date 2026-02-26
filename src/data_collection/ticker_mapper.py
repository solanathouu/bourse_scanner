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
    "AVENIR TELECOM": "AVT.PA",
    "BIOSYNEX": "ALBIO.PA",
    "BNPP SP500 EUR C": "ESE.PA",
    "CAPITAL B.": "ALCAP.PA",
    "CROSSJECT": "ALCJ.PA",
    "DBV TECHNOLOGIES": "DBV.PA",
    "EXAIL TECHNOLOGIES": "EXA.PA",
    "GENFIT": "GNFT.PA",
    "INVENTIVA": "IVA.PA",
    "KALRAY": "ALKAL.PA",
    "MAUREL ET PROM": "MAU.PA",
    "MEDIAN TECHNOLOG.": "ALMDT.PA",
    "MEMSCAP REGROUPEES": "MEMS.PA",
    "NANOBIOTIX": "NANO.PA",
    "POXEL": "POXEL.PA",
    "REXEL": "RXL.PA",
    "SANOFI": "SAN.PA",
    "SCHNEIDER ELECTRIC SE": "SU.PA",
    "SENSORION": "ALSEN.PA",
    "SOITEC": "SOI.PA",
    "TECHNIP ENERGIES": "TE.PA",
    "THE BLOKCHAIN GP": "ALTBG.PA",
    "UBISOFT": "UBI.PA",
    "VALBIOTIS": "ALVAL.PA",
    "VALNEVA": "VLA.PA",
    "VINCI": "DG.PA",
    "WORLDLINE": "WLN.PA",
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

    def get_action_name(self, ticker: str) -> str | None:
        """Retourne le nom d'action pour un ticker Yahoo Finance (reverse lookup).

        Returns:
            Nom d'action si trouve, None sinon.
        """
        for name, t in self._map.items():
            if t == ticker:
                return name
        return None

    def get_all_mappings(self) -> dict[str, str]:
        """Retourne tous les mappings nom -> ticker."""
        return self._map.copy()
