"""Classification des news par type de catalyseur via regles de mots-cles.

Analyse le CONTENU des news (titre + description) pour comprendre
le type d'evenement qui a declenche un trade de Nicolas.
"""

import re

from loguru import logger


# Types de catalyseurs avec mots-cles FR+EN et priorite
# Priorite haute = plus specifique, gagne en cas de multi-match
CATALYST_RULES = {
    "EARNINGS": {
        "keywords": [
            r"r[ée]sultat", r"chiffre d'affaires", r"b[ée]n[ée]fice", r"CA T[1-4]",
            r"bilan", r"comptes", r"trimestriel", r"semestriel",
            r"croissance du CA", r"marge op[ée]rationnelle",
            r"earnings", r"revenue", r"profit", r"quarterly results", r"EPS",
            r"beat expectations", r"guidance", r"fiscal year",
        ],
        "priority": 10,
    },
    "FDA_REGULATORY": {
        "keywords": [
            r"\bFDA\b", r"\bAMM\b", r"autorisation", r"approbation",
            r"\bEMA\b", r"phase [123]", r"essai clinique", r"r[ée]glementaire",
            r"\bANSM\b", r"mise sur le march[ée]",
            r"approval", r"regulatory", r"clinical trial",
            r"marketing authorization",
        ],
        "priority": 11,
    },
    "UPGRADE": {
        "keywords": [
            r"rel[èe]ve", r"objectif de cours", r"surperformance",
            r"surpond[ée]rer", r"potentiel de hausse", r"recommandation.*achat",
            r"upgrade", r"\bbuy\b", r"outperform", r"price target.*rais",
            r"overweight", r"raises target",
        ],
        "priority": 8,
    },
    "DOWNGRADE": {
        "keywords": [
            r"abaisse", r"d[ée]gradation", r"sous-performance",
            r"sous-pond[ée]rer", r"recommandation.*vente",
            r"downgrade", r"\bsell\b", r"underperform", r"underweight",
            r"cuts target", r"lower.*target",
        ],
        "priority": 8,
    },
    "CONTRACT": {
        "keywords": [
            r"contrat", r"partenariat", r"acquisition", r"accord",
            r"commande", r"alliance", r"collaboration", r"joint.?venture",
            r"contract", r"partnership", r"deal\b", r"agreement",
            r"\border\b",
        ],
        "priority": 7,
    },
    "DIVIDEND": {
        "keywords": [
            r"dividende", r"distribution", r"coupon", r"d[ée]tachement",
            r"dividend", r"payout", r"\byield\b",
        ],
        "priority": 6,
    },
    "RESTRUCTURING": {
        "keywords": [
            r"restructuration", r"plan social", r"cession",
            r"r[ée]organisation", r"licenciement", r"fermeture",
            r"restructuring", r"layoff", r"divestiture", r"cost cutting",
        ],
        "priority": 6,
    },
    "INSIDER": {
        "keywords": [
            r"dirigeant", r"rachat d'actions", r"franchissement de seuil",
            r"participation", r"actionnariat",
            r"insider", r"buyback", r"\bstake\b", r"shareholder",
        ],
        "priority": 5,
    },
    "SECTOR_MACRO": {
        "keywords": [
            r"\bCAC\b", r"march[ée]", r"indice", r"\bBCE\b",
            r"inflation", r"\btaux\b", r"conjoncture", r"secteur",
            r"market", r"index", r"\bECB\b", r"macro",
        ],
        "priority": 2,
    },
    "OTHER_POSITIVE": {
        "keywords": [
            r"hausse", r"progression", r"rebond", r"reprise", r"en forme",
            r"rally", r"surge", r"\bgain\b", r"rise\b", r"recovery",
            r"bullish",
        ],
        "priority": 1,
    },
    "OTHER_NEGATIVE": {
        "keywords": [
            r"baisse", r"chute", r"recul", r"perte", r"recule",
            r"d[ée]croche",
            r"decline", r"\bdrop\b", r"\bfall\b", r"\bloss\b",
            r"bearish", r"plunge",
        ],
        "priority": 1,
    },
}


class NewsClassifier:
    """Classifie les news par type de catalyseur via regles de mots-cles.

    Analyse le titre et la description pour determiner le type d'evenement:
    EARNINGS, FDA_REGULATORY, UPGRADE, DOWNGRADE, CONTRACT, DIVIDEND,
    RESTRUCTURING, INSIDER, SECTOR_MACRO, OTHER_POSITIVE, OTHER_NEGATIVE.
    """

    def __init__(self):
        # Pre-compiler les regex pour chaque type
        self._compiled_rules = {}
        for cat_type, rule in CATALYST_RULES.items():
            patterns = [re.compile(kw, re.IGNORECASE) for kw in rule["keywords"]]
            self._compiled_rules[cat_type] = {
                "patterns": patterns,
                "priority": rule["priority"],
            }

    def classify(self, title: str, description: str | None) -> str:
        """Classifie une news par type de catalyseur.

        Cherche les mots-cles dans le titre puis la description.
        Si multi-match, retourne le type avec la priorite la plus haute.
        Retourne 'UNKNOWN' si aucun match.
        """
        text = (title or "").lower()
        if description:
            text += " " + description.lower()

        matches = []
        for cat_type, rule in self._compiled_rules.items():
            for pattern in rule["patterns"]:
                if pattern.search(text):
                    matches.append((cat_type, rule["priority"]))
                    break  # Un match suffit pour ce type

        if not matches:
            return "UNKNOWN"

        # Trier par priorite decroissante, prendre le plus specifique
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[0][0]

    def classify_batch(self, news_list: list[dict]) -> list[dict]:
        """Classifie une liste de news et ajoute le champ 'catalyst_type'.

        Chaque dict doit avoir 'title' et 'description'.
        Retourne les memes dicts avec 'catalyst_type' ajoute.
        """
        results = []
        for news in news_list:
            cat_type = self.classify(
                news.get("title", ""),
                news.get("description"),
            )
            results.append({**news, "catalyst_type": cat_type})
        return results

    def summarize_for_trade(self, news_with_types: list[dict]) -> dict:
        """Resume les types de catalyseurs pour un trade.

        Args:
            news_with_types: Liste de dicts avec 'catalyst_type' et 'score_pertinence'.

        Returns:
            {
                "primary_type": str,      # Type du catalyseur avec le meilleur score
                "types_found": list[str], # Tous les types distincts
                "nb_types": int,          # Nombre de types differents
            }
        """
        if not news_with_types:
            return {
                "primary_type": "TECHNICAL",
                "types_found": [],
                "nb_types": 0,
            }

        # Trier par score de pertinence decroissant
        sorted_news = sorted(
            news_with_types,
            key=lambda n: n.get("score_pertinence", 0),
            reverse=True,
        )
        primary = sorted_news[0]["catalyst_type"]
        types_found = list({n["catalyst_type"] for n in news_with_types})

        return {
            "primary_type": primary,
            "types_found": types_found,
            "nb_types": len(types_found),
        }
