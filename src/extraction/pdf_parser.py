"""Parser pour les avis d'exécution PDF Société Générale.

Format détecté (stable sur tous les PDF analysés) :
- Ligne sens : "ACHAT AU COMPTANT" ou "VENTE AU COMPTANT"
- Date/heure : "DD/MM/YYYY" puis "HH:MM:SS" (lignes séparées)
- Quantité + nom action : "QTY NOM_ACTION"
- Code ISIN : "Code ISIN : FRXXXXXXXXXX"
- Cours exécuté : "Cours exécuté : X,XXX EUR"
- Montants : "Montant brut | Commission | Frais | Montant net"
- Format français : virgule décimale, espace séparateur milliers
"""

import re
import os
import pdfplumber
from loguru import logger


class PDFParseError(Exception):
    """Erreur lors du parsing d'un PDF."""
    pass


def _parse_french_number(s: str) -> float:
    """Convertit un nombre au format français en float.

    Exemples: '3 384,47' -> 3384.47, '540,00' -> 540.0, '4,085' -> 4.085
    """
    if not s:
        return 0.0
    # Retirer "EUR", espaces insécables, et espaces normaux utilisés comme séparateurs de milliers
    cleaned = s.replace("EUR", "").replace("\xa0", "").replace(" ", "").strip()
    # Virgule -> point
    cleaned = cleaned.replace(",", ".")
    return float(cleaned)


class SGPDFParser:
    """Parse les avis d'exécution Société Générale au format PDF."""

    def parse_pdf(self, pdf_path: str) -> dict:
        """
        Extrait les données d'un avis d'exécution PDF SG.

        Returns:
            Dict avec: date_execution, heure_execution, sens, nom_action,
            isin, quantite, prix_unitaire, montant_brut, commission, frais,
            montant_net, fichier_source
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF non trouvé: {pdf_path}")

        text = self._extract_text(pdf_path)
        data = self._parse_text(text)
        data["fichier_source"] = os.path.basename(pdf_path)

        self._validate(data)
        return data

    def _extract_text(self, pdf_path: str) -> str:
        """Extrait le texte brut du PDF."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages_text = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append(page_text)
                return "\n".join(pages_text)
        except Exception as e:
            raise PDFParseError(f"Erreur extraction texte de {pdf_path}: {e}")

    def _parse_text(self, text: str) -> dict:
        """Parse le texte extrait pour en extraire les champs."""
        data = {}

        # Sens : ACHAT AU COMPTANT ou VENTE AU COMPTANT
        sens_match = re.search(r"(ACHAT|VENTE)\s+AU\s+COMPTANT", text)
        if sens_match:
            data["sens"] = sens_match.group(1)

        # Date d'exécution : DD/MM/YYYY
        # Apparaît après "locale d'exécution" dans le tableau
        date_match = re.search(
            r"locale d'ex[ée]cution.*?\n(\d{2}/\d{2}/\d{4})",
            text, re.DOTALL
        )
        if date_match:
            d, m, y = date_match.group(1).split("/")
            data["date_execution"] = f"{y}-{m}-{d}"

        # Heure d'exécution : HH:MM:SS (ligne après la date)
        heure_match = re.search(
            r"\d{2}/\d{2}/\d{4}\s+\d+\s+.+?\n(\d{2}:\d{2}:\d{2})",
            text
        )
        if heure_match:
            data["heure_execution"] = heure_match.group(1)
        else:
            data["heure_execution"] = "00:00:00"

        # Quantité et nom de l'action : sur la même ligne que la date
        # Format : "DD/MM/YYYY QTY NOM_ACTION Référence : ..."
        qte_nom_match = re.search(
            r"\d{2}/\d{2}/\d{4}\s+(\d+)\s+(.+?)\s+R[ée]f[ée]rence",
            text
        )
        if qte_nom_match:
            data["quantite"] = int(qte_nom_match.group(1))
            data["nom_action"] = qte_nom_match.group(2).strip()

        # Code ISIN
        isin_match = re.search(r"Code ISIN\s*:\s*([A-Z]{2}\d{10})", text)
        if isin_match:
            data["isin"] = isin_match.group(1)

        # Cours exécuté (= prix unitaire)
        cours_match = re.search(
            r"Cours ex[ée]cut[ée]\s*:\s*([\d\s,.]+?)\s*EUR",
            text
        )
        if cours_match:
            data["prix_unitaire"] = _parse_french_number(cours_match.group(1))

        # Montants : Montant brut | Commission | Frais | Montant net
        # Le format est une ligne avec les montants séparés par des espaces
        # Patterns possibles :
        # "540,00 EUR 540,00 EUR" (sans commission, sans frais)
        # "626,25 EUR 3,13 EUR 629,38 EUR" (avec commission, sans frais)
        # "3 384,47 EUR 15,23 EUR 3 369,24 EUR" (vente avec commission)

        # On cherche la ligne après "Montant net au débit/crédit"
        montants_match = re.search(
            r"Montant net au (?:d[ée]bit|cr[ée]dit) de votre compte\n"
            r"([\d\s,\.EUR]+)",
            text
        )
        if montants_match:
            montants_line = montants_match.group(1).strip()
            # Extraire tous les montants "X XXX,XX EUR" de la ligne
            amounts = re.findall(r"([\d\s]+,\d{2})\s*EUR", montants_line)

            if len(amounts) >= 2:
                data["montant_brut"] = _parse_french_number(amounts[0])
                if len(amounts) == 3:
                    # Montant brut, Commission, Montant net
                    data["commission"] = _parse_french_number(amounts[1])
                    data["frais"] = 0.0
                    data["montant_net"] = _parse_french_number(amounts[2])
                elif len(amounts) == 4:
                    # Montant brut, Commission, Frais, Montant net
                    data["commission"] = _parse_french_number(amounts[1])
                    data["frais"] = _parse_french_number(amounts[2])
                    data["montant_net"] = _parse_french_number(amounts[3])
                else:
                    # 2 montants : brut et net (pas de commission)
                    data["commission"] = 0.0
                    data["frais"] = 0.0
                    data["montant_net"] = _parse_french_number(amounts[1])
            elif len(amounts) == 1:
                # Un seul montant = brut = net (pas de commission)
                data["montant_brut"] = _parse_french_number(amounts[0])
                data["commission"] = 0.0
                data["frais"] = 0.0
                data["montant_net"] = data["montant_brut"]

        # Vérifier si offre promotionnelle (commission offerte)
        if "offre promotionnelle" in text.lower():
            promo_match = re.search(
                r"remise de ([\d,]+)\s*EUR",
                text
            )
            if promo_match and "commission" not in data:
                data["commission"] = 0.0
                data["frais"] = 0.0

        return data

    def _validate(self, data: dict):
        """Valide la cohérence des données extraites."""
        required = ["date_execution", "sens", "nom_action", "quantite", "prix_unitaire"]
        missing = [f for f in required if f not in data or data[f] is None]
        if missing:
            raise PDFParseError(
                f"Champs manquants dans {data.get('fichier_source', '?')}: {missing}"
            )

        # Vérifier cohérence montant_brut vs quantité × prix
        if "montant_brut" in data and data["quantite"] and data["prix_unitaire"]:
            expected = data["quantite"] * data["prix_unitaire"]
            actual = data["montant_brut"]
            if expected > 0 and abs(actual - expected) / expected > 0.02:
                logger.warning(
                    f"Incohérence montant brut dans {data.get('fichier_source')}: "
                    f"attendu {expected:.2f}, trouvé {actual:.2f}"
                )

        # S'assurer que les champs optionnels ont des valeurs par défaut
        data.setdefault("commission", 0.0)
        data.setdefault("frais", 0.0)
        data.setdefault("isin", None)

        # Calculer montant_net si absent
        if "montant_net" not in data and "montant_brut" in data:
            commission = data.get("commission", 0)
            frais = data.get("frais", 0)
            if data["sens"] == "ACHAT":
                data["montant_net"] = data["montant_brut"] + commission + frais
            else:
                data["montant_net"] = data["montant_brut"] - commission - frais

    def parse_directory(self, directory: str) -> list[dict]:
        """
        Parse tous les PDF d'un dossier.

        Returns:
            Liste de dicts (un par PDF parsé avec succès).
            Les erreurs sont loguées mais n'arrêtent pas le batch.
        """
        results = []
        errors = []
        pdf_files = sorted([
            f for f in os.listdir(directory)
            if f.lower().endswith(".pdf")
        ])

        if not pdf_files:
            logger.warning(f"Aucun PDF trouvé dans {directory}")
            return results

        logger.info(f"Parsing de {len(pdf_files)} PDF dans {directory}")

        for filename in pdf_files:
            filepath = os.path.join(directory, filename)
            try:
                data = self.parse_pdf(filepath)
                results.append(data)
                logger.debug(
                    f"OK: {filename} -> {data['sens']} {data.get('quantite', '?')}x "
                    f"{data['nom_action']} @ {data.get('prix_unitaire', '?')}"
                )
            except (PDFParseError, FileNotFoundError) as e:
                errors.append({"file": filename, "error": str(e)})
                logger.error(f"ERREUR: {filename} -> {e}")

        logger.info(
            f"Parsing terminé: {len(results)} succès, {len(errors)} erreurs "
            f"sur {len(pdf_files)} PDF"
        )

        if errors:
            logger.warning(f"PDF en erreur: {[e['file'] for e in errors]}")

        return results
