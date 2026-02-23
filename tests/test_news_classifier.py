"""Tests pour la classification des news par type de catalyseur."""

import pytest

from src.analysis.news_classifier import NewsClassifier


class TestClassifySingle:
    """Tests de classification d'une seule news."""

    def setup_method(self):
        self.classifier = NewsClassifier()

    def test_earnings_fr(self):
        """Detecte une annonce de resultats en francais."""
        result = self.classifier.classify(
            "Sanofi: resultats T3 au-dessus du consensus", ""
        )
        assert result == "EARNINGS"

    def test_earnings_en(self):
        """Detecte une annonce de resultats en anglais."""
        result = self.classifier.classify(
            "Sanofi Q3 earnings beat expectations", ""
        )
        assert result == "EARNINGS"

    def test_earnings_chiffre_affaires(self):
        """Detecte 'chiffre d'affaires' comme EARNINGS."""
        result = self.classifier.classify(
            "ADOCIA: chiffre d'affaires en hausse de 15%", ""
        )
        assert result == "EARNINGS"

    def test_fda_approval_fr(self):
        """Detecte une approbation FDA en francais."""
        result = self.classifier.classify(
            "Nanobiotix obtient l'autorisation FDA pour son traitement", ""
        )
        assert result == "FDA_REGULATORY"

    def test_fda_approval_en(self):
        """Detecte FDA approval en anglais."""
        result = self.classifier.classify(
            "FDA approves Sanofi new drug application", ""
        )
        assert result == "FDA_REGULATORY"

    def test_clinical_trial(self):
        """Detecte un essai clinique."""
        result = self.classifier.classify(
            "DBV Technologies: resultats positifs de phase 3", ""
        )
        assert result == "FDA_REGULATORY"

    def test_upgrade_fr(self):
        """Detecte un upgrade analyste en francais."""
        result = self.classifier.classify(
            "Goldman Sachs releve son objectif de cours sur Air Liquide", ""
        )
        assert result == "UPGRADE"

    def test_upgrade_en(self):
        """Detecte un upgrade en anglais."""
        result = self.classifier.classify(
            "JP Morgan upgrades Schneider Electric to Buy", ""
        )
        assert result == "UPGRADE"

    def test_downgrade_fr(self):
        """Detecte un downgrade en francais."""
        result = self.classifier.classify(
            "Morgan Stanley abaisse sa recommandation sur Kalray", ""
        )
        assert result == "DOWNGRADE"

    def test_contract_fr(self):
        """Detecte un contrat/partenariat en francais."""
        result = self.classifier.classify(
            "Technip Energies remporte un contrat de 500M$ au Qatar", ""
        )
        assert result == "CONTRACT"

    def test_contract_en(self):
        """Detecte un partnership en anglais."""
        result = self.classifier.classify(
            "Sanofi announces partnership with Regeneron", ""
        )
        assert result == "CONTRACT"

    def test_dividend_fr(self):
        """Detecte un dividende."""
        result = self.classifier.classify(
            "Air Liquide: detachement du dividende de 3.20 euros", ""
        )
        assert result == "DIVIDEND"

    def test_insider_fr(self):
        """Detecte un mouvement d'insider."""
        result = self.classifier.classify(
            "Declaration de franchissement de seuil sur DBV Technologies", ""
        )
        assert result == "INSIDER"

    def test_sector_macro(self):
        """Detecte une news macro/sectorielle."""
        result = self.classifier.classify(
            "Le CAC 40 gagne 1.5% porte par le secteur du luxe", ""
        )
        assert result == "SECTOR_MACRO"

    def test_positive_generic(self):
        """Detecte une news positive generique."""
        result = self.classifier.classify(
            "ADOCIA en forte hausse apres une seance de rebond", ""
        )
        assert result == "OTHER_POSITIVE"

    def test_negative_generic(self):
        """Detecte une news negative generique."""
        result = self.classifier.classify(
            "Kalray: le titre recule fortement en bourse", ""
        )
        assert result == "OTHER_NEGATIVE"

    def test_unknown(self):
        """News sans mot-cle reconnu retourne UNKNOWN."""
        result = self.classifier.classify(
            "Assemblee generale annuelle convoquee", ""
        )
        assert result == "UNKNOWN"

    def test_match_in_description(self):
        """Le matching cherche aussi dans la description."""
        result = self.classifier.classify(
            "Actualite Sanofi", "Le laboratoire publie ses resultats trimestriels"
        )
        assert result == "EARNINGS"

    def test_none_description(self):
        """Description None ne crashe pas."""
        result = self.classifier.classify("Sanofi resultats Q3", None)
        assert result == "EARNINGS"

    def test_priority_fda_over_earnings(self):
        """FDA est prioritaire sur EARNINGS si les deux matchent."""
        result = self.classifier.classify(
            "Resultats positifs de l'essai clinique phase 3 de Nanobiotix", ""
        )
        assert result == "FDA_REGULATORY"

    def test_priority_specific_over_generic(self):
        """EARNINGS est prioritaire sur OTHER_POSITIVE."""
        result = self.classifier.classify(
            "Sanofi en hausse apres des resultats solides", ""
        )
        assert result == "EARNINGS"


class TestClassifyBatch:
    """Tests de classification en batch."""

    def setup_method(self):
        self.classifier = NewsClassifier()

    def test_classify_batch(self):
        """Classifie une liste de news."""
        news_list = [
            {"title": "Sanofi resultats Q3", "description": ""},
            {"title": "FDA approves drug", "description": ""},
            {"title": "Le marche monte", "description": "CAC en hausse"},
        ]
        results = self.classifier.classify_batch(news_list)
        assert len(results) == 3
        assert results[0]["catalyst_type"] == "EARNINGS"
        assert results[1]["catalyst_type"] == "FDA_REGULATORY"
        assert results[2]["catalyst_type"] == "SECTOR_MACRO"

    def test_classify_news_for_trade(self):
        """Resume les catalyseurs pour un trade."""
        news_with_types = [
            {"catalyst_type": "EARNINGS", "score_pertinence": 0.9},
            {"catalyst_type": "UPGRADE", "score_pertinence": 0.7},
            {"catalyst_type": "SECTOR_MACRO", "score_pertinence": 0.5},
        ]
        result = self.classifier.summarize_for_trade(news_with_types)
        assert result["primary_type"] == "EARNINGS"  # Meilleur score
        assert "EARNINGS" in result["types_found"]
        assert "UPGRADE" in result["types_found"]
        assert result["nb_types"] == 3

    def test_classify_trade_empty(self):
        """Trade sans catalyseurs retourne TECHNICAL."""
        result = self.classifier.summarize_for_trade([])
        assert result["primary_type"] == "TECHNICAL"
        assert result["nb_types"] == 0
