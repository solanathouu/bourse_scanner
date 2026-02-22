"""Tests pour le mapping nom_action -> ticker Yahoo Finance."""

import pytest
from src.data_collection.ticker_mapper import TickerMapper, TickerNotFoundError


class TestTickerMapper:
    """Tests de mapping nom -> ticker."""

    def setup_method(self):
        self.mapper = TickerMapper()

    def test_mapping_sanofi(self):
        assert self.mapper.get_ticker("SANOFI") == "SAN.PA"

    def test_mapping_2crsi(self):
        assert self.mapper.get_ticker("2CRSI") == "2CRSI.PA"

    def test_mapping_schneider(self):
        assert self.mapper.get_ticker("SCHNEIDER ELECTRIC SE") == "SU.PA"

    def test_mapping_etf_amundi(self):
        assert self.mapper.get_ticker("AMUNDI ETF MSCI WR") == "CW8.PA"

    def test_mapping_etf_bnpp(self):
        assert self.mapper.get_ticker("BNPP SP500 EUR C") == "ESE.PA"

    def test_mapping_avec_asterisque(self):
        """Les noms avec * en prefixe sont nettoyes."""
        assert self.mapper.get_ticker("* GENFIT") == "GNFT.PA"
        assert self.mapper.get_ticker("* MAUREL ET PROM") == "MAU.PA"

    def test_mapping_inconnu_leve_erreur(self):
        with pytest.raises(TickerNotFoundError, match="INCONNU"):
            self.mapper.get_ticker("INCONNU")

    def test_get_all_mappings(self):
        """Retourne tous les mappings connus."""
        mappings = self.mapper.get_all_mappings()
        assert isinstance(mappings, dict)
        assert len(mappings) >= 19

    def test_get_ticker_for_all_traded_actions(self):
        """Verifie que toutes les actions tradees ont un mapping."""
        traded = [
            "2CRSI", "AB SCIENCE", "ADOCIA", "AFYREN", "AIR LIQUIDE",
            "AMUNDI ETF MSCI WR", "BNPP SP500 EUR C", "DBV TECHNOLOGIES",
            "EXAIL TECHNOLOGIES", "INVENTIVA", "KALRAY", "MAUREL ET PROM",
            "MEMSCAP REGROUPEES", "NANOBIOTIX", "SANOFI",
            "SCHNEIDER ELECTRIC SE", "TECHNIP ENERGIES",
            "THE BLOKCHAIN GP", "VALNEVA",
        ]
        for name in traded:
            ticker = self.mapper.get_ticker(name)
            assert ticker.endswith(".PA"), f"{name} -> {ticker} ne finit pas par .PA"
