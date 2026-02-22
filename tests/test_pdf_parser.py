"""Tests pour le parser PDF SG."""

import os
import tempfile
import pytest

from src.extraction.pdf_parser import SGPDFParser, PDFParseError, _parse_french_number


class TestParseFrenchNumber:
    """Tests pour la conversion des nombres au format français."""

    def test_simple_number(self):
        assert _parse_french_number("540,00") == 540.0

    def test_number_with_thousands_separator(self):
        assert _parse_french_number("3 384,47") == 3384.47

    def test_number_with_eur_suffix(self):
        assert _parse_french_number("626,25 EUR") == 626.25

    def test_number_with_three_decimals(self):
        assert _parse_french_number("4,085") == 4.085

    def test_empty_string(self):
        assert _parse_french_number("") == 0.0

    def test_number_with_spaces_and_eur(self):
        assert _parse_french_number("42 119,95 EUR") == 42119.95


class TestSGPDFParserValidation:
    """Tests de la logique de validation (sans vrais PDF)."""

    def setup_method(self):
        self.parser = SGPDFParser()

    def test_validate_missing_required_fields(self):
        """Un dict incomplet doit lever PDFParseError."""
        data = {"date_execution": "2024-03-15", "fichier_source": "test.pdf"}
        with pytest.raises(PDFParseError, match="Champs manquants"):
            self.parser._validate(data)

    def test_validate_complete_data(self):
        """Un dict complet ne doit pas lever d'erreur."""
        data = {
            "date_execution": "2025-05-09",
            "heure_execution": "14:22:06",
            "sens": "ACHAT",
            "nom_action": "EXAIL TECHNOLOGIES",
            "isin": "FR0000062671",
            "quantite": 10,
            "prix_unitaire": 54.0,
            "montant_brut": 540.0,
            "commission": 0.0,
            "frais": 0.0,
            "montant_net": 540.0,
            "fichier_source": "test.pdf",
        }
        self.parser._validate(data)

    def test_validate_calculates_net_for_achat_when_missing(self):
        """Pour un ACHAT sans montant_net, il doit être calculé."""
        data = {
            "date_execution": "2025-08-20",
            "sens": "ACHAT",
            "nom_action": "ADOCIA",
            "quantite": 75,
            "prix_unitaire": 8.35,
            "montant_brut": 626.25,
            "commission": 3.13,
            "frais": 0.0,
            "fichier_source": "test.pdf",
        }
        self.parser._validate(data)
        assert data["montant_net"] == 626.25 + 3.13

    def test_validate_calculates_net_for_vente_when_missing(self):
        """Pour une VENTE sans montant_net, il doit être calculé."""
        data = {
            "date_execution": "2025-08-21",
            "sens": "VENTE",
            "nom_action": "ADOCIA",
            "quantite": 379,
            "prix_unitaire": 8.93,
            "montant_brut": 3384.47,
            "commission": 15.23,
            "frais": 0.0,
            "fichier_source": "test.pdf",
        }
        self.parser._validate(data)
        assert data["montant_net"] == 3384.47 - 15.23

    def test_parse_nonexistent_file(self):
        """Un fichier inexistant doit lever FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            self.parser.parse_pdf("/chemin/inexistant.pdf")

    def test_parse_directory_empty(self):
        """Un dossier vide retourne une liste vide."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = self.parser.parse_directory(tmpdir)
            assert results == []


class TestSGPDFParserOnRealPDFs:
    """Tests sur les vrais PDF SG (nécessite data/pdfs/)."""

    PDF_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "pdfs"
    )

    @pytest.fixture
    def parser(self):
        return SGPDFParser()

    def _skip_if_no_pdfs(self):
        if not os.path.exists(self.PDF_DIR) or not os.listdir(self.PDF_DIR):
            pytest.skip("Pas de PDF dans data/pdfs/")

    def test_parse_achat_sans_commission(self, parser):
        """ACHAT avec offre promo (commission = 0)."""
        self._skip_if_no_pdfs()
        pdf = os.path.join(self.PDF_DIR, "AvisOperation_FR0000062671_20250509.pdf")
        if not os.path.exists(pdf):
            pytest.skip("PDF spécifique non trouvé")

        result = parser.parse_pdf(pdf)
        assert result["sens"] == "ACHAT"
        assert result["nom_action"] == "EXAIL TECHNOLOGIES"
        assert result["isin"] == "FR0000062671"
        assert result["quantite"] == 10
        assert result["prix_unitaire"] == 54.0
        assert result["montant_brut"] == 540.0
        assert result["commission"] == 0.0
        assert result["date_execution"] == "2025-05-09"
        assert result["heure_execution"] == "14:22:06"

    def test_parse_achat_avec_commission(self, parser):
        """ACHAT avec commission."""
        self._skip_if_no_pdfs()
        pdf = os.path.join(self.PDF_DIR, "AvisOperation_FR0011184241_20250820.pdf")
        if not os.path.exists(pdf):
            pytest.skip("PDF spécifique non trouvé")

        result = parser.parse_pdf(pdf)
        assert result["sens"] == "ACHAT"
        assert result["nom_action"] == "ADOCIA"
        assert result["quantite"] == 75
        assert result["prix_unitaire"] == 8.35
        assert result["commission"] == 3.13
        assert result["montant_net"] == 629.38

    def test_parse_vente(self, parser):
        """VENTE avec commission."""
        self._skip_if_no_pdfs()
        pdf = os.path.join(self.PDF_DIR, "AvisOperation_FR0011184241_20250821.pdf")
        if not os.path.exists(pdf):
            pytest.skip("PDF spécifique non trouvé")

        result = parser.parse_pdf(pdf)
        assert result["sens"] == "VENTE"
        assert result["nom_action"] == "ADOCIA"
        assert result["quantite"] == 379
        assert result["prix_unitaire"] == 8.93
        assert result["montant_brut"] == 3384.47
        assert result["commission"] == 15.23
        assert result["montant_net"] == 3369.24

    def test_parse_all_pdfs_no_errors(self, parser):
        """Tous les PDF doivent être parsés sans erreur."""
        self._skip_if_no_pdfs()
        results = parser.parse_directory(self.PDF_DIR)
        pdf_count = len([f for f in os.listdir(self.PDF_DIR) if f.endswith(".pdf")])
        # Au moins 95% de réussite
        assert len(results) >= pdf_count * 0.95

    def test_all_required_fields_present(self, parser):
        """Chaque résultat doit avoir tous les champs requis."""
        self._skip_if_no_pdfs()
        results = parser.parse_directory(self.PDF_DIR)
        required = [
            "date_execution", "heure_execution", "sens", "nom_action",
            "quantite", "prix_unitaire", "fichier_source",
        ]
        for result in results:
            for field in required:
                assert field in result, f"Champ {field} manquant dans {result.get('fichier_source')}"
                assert result[field] is not None, f"Champ {field} est None dans {result.get('fichier_source')}"
