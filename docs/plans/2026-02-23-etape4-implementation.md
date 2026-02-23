# Etape 4 — ML Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Construire le pipeline ML qui apprend le style de trading de Nicolas (range trading + catalyseurs news) et entraine un XGBoost pour predire ses decisions.

**Architecture:** 3 composants dans `src/analysis/` (NewsClassifier, TechnicalIndicators, FeatureEngine) + 2 composants dans `src/model/` (Trainer, Evaluator) + 2 scripts CLI. Chaque composant est teste independamment en TDD.

**Tech Stack:** pandas, numpy, ta (indicateurs techniques), xgboost, scikit-learn, joblib. SQLite existant via Database class.

**Design doc:** `docs/plans/2026-02-23-etape4-ml-pipeline-design.md`

---

### Task 1: Ajouter les dependances ML

**Files:**
- Modify: `pyproject.toml`

**Step 1: Ajouter ta, xgboost, scikit-learn, joblib aux dependances**

Dans `pyproject.toml`, remplacer le bloc dependencies par:

```toml
dependencies = [
    "loguru>=0.7.3",
    "pandas>=3.0.1",
    "pdfplumber>=0.11.9",
    "python-dotenv>=1.2.1",
    "pyyaml>=6.0.3",
    "yfinance>=0.2.50",
    "gnews>=0.4.3",
    "ta>=0.11.0",
    "xgboost>=2.1.0",
    "scikit-learn>=1.5.0",
    "joblib>=1.4.0",
    "numpy>=2.0.0",
]
```

**Step 2: Installer les dependances**

Run: `uv sync`
Expected: Installation reussie

**Step 3: Verifier l'installation**

Run: `uv run python -c "import ta; import xgboost; import sklearn; import joblib; print('OK')"`
Expected: `OK`

**Step 4: Run all existing tests (regression)**

Run: `uv run pytest tests/ -v`
Expected: 97 PASS (aucune regression)

**Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add ta, xgboost, scikit-learn, joblib, numpy for ML pipeline"
```

---

### Task 2: NewsClassifier — Classification des news par type de catalyseur

**Files:**
- Create: `src/analysis/news_classifier.py`
- Create: `tests/test_news_classifier.py`

**Step 1: Write the failing tests**

Creer `tests/test_news_classifier.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_news_classifier.py -v`
Expected: FAIL (module not found)

**Step 3: Implement NewsClassifier**

Creer `src/analysis/news_classifier.py`:

```python
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
            r"bilan", r"comptes", r"trimestriel", r"semestriel", r"annuel",
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
        "priority": 10,
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
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_news_classifier.py -v`
Expected: 22 PASS

**Step 5: Run all tests (regression)**

Run: `uv run pytest tests/ -v`
Expected: 119 PASS (97 + 22)

**Step 6: Commit**

```bash
git add src/analysis/news_classifier.py tests/test_news_classifier.py
git commit -m "feat(analysis): add NewsClassifier - keyword-based catalyst type detection"
```

---

### Task 3: TechnicalIndicators — Indicateurs techniques + detection de range

**Files:**
- Create: `src/analysis/technical_indicators.py`
- Create: `tests/test_technical_indicators.py`

**Step 1: Write the failing tests**

Creer `tests/test_technical_indicators.py`:

```python
"""Tests pour le calcul des indicateurs techniques."""

import pandas as pd
import numpy as np
import pytest

from src.analysis.technical_indicators import TechnicalIndicators


def _make_price_df(n_days: int = 60) -> pd.DataFrame:
    """Cree un DataFrame de prix synthetiques simulant un range trading.

    Simule une action oscillant entre 10 et 12 (range de 20%).
    """
    np.random.seed(42)
    dates = pd.bdate_range("2025-06-01", periods=n_days)
    # Oscillation sinusoidale + bruit
    t = np.linspace(0, 4 * np.pi, n_days)
    base = 11 + np.sin(t)  # Oscille entre 10 et 12
    noise = np.random.normal(0, 0.1, n_days)
    close = base + noise

    return pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "open": close - np.random.uniform(0, 0.3, n_days),
        "high": close + np.random.uniform(0, 0.5, n_days),
        "low": close - np.random.uniform(0, 0.5, n_days),
        "close": close,
        "volume": np.random.randint(50000, 200000, n_days),
    })


class TestComputeAll:
    """Tests du calcul complet des indicateurs."""

    def setup_method(self):
        self.tech = TechnicalIndicators()
        self.df = _make_price_df(60)

    def test_compute_all_returns_dataframe(self):
        """compute_all retourne un DataFrame."""
        result = self.tech.compute_all(self.df)
        assert isinstance(result, pd.DataFrame)

    def test_compute_all_has_rsi(self):
        """Le resultat contient RSI(14)."""
        result = self.tech.compute_all(self.df)
        assert "rsi_14" in result.columns
        # RSI doit etre entre 0 et 100
        valid = result["rsi_14"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_compute_all_has_macd(self):
        """Le resultat contient MACD histogram."""
        result = self.tech.compute_all(self.df)
        assert "macd_histogram" in result.columns

    def test_compute_all_has_bollinger(self):
        """Le resultat contient bollinger_position."""
        result = self.tech.compute_all(self.df)
        assert "bollinger_position" in result.columns

    def test_compute_all_has_range_position(self):
        """Le resultat contient range_position_10 et range_position_20."""
        result = self.tech.compute_all(self.df)
        assert "range_position_10" in result.columns
        assert "range_position_20" in result.columns

    def test_range_position_between_0_and_1(self):
        """range_position est entre 0 (support) et 1 (resistance)."""
        result = self.tech.compute_all(self.df)
        valid = result["range_position_10"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_compute_all_has_range_amplitude(self):
        """Le resultat contient range_amplitude_10 et range_amplitude_20."""
        result = self.tech.compute_all(self.df)
        assert "range_amplitude_10" in result.columns
        assert "range_amplitude_20" in result.columns

    def test_range_amplitude_positive(self):
        """L'amplitude du range est positive."""
        result = self.tech.compute_all(self.df)
        valid = result["range_amplitude_10"].dropna()
        assert (valid >= 0).all()

    def test_compute_all_has_volume_ratio(self):
        """Le resultat contient volume_ratio_20."""
        result = self.tech.compute_all(self.df)
        assert "volume_ratio_20" in result.columns

    def test_compute_all_has_variations(self):
        """Le resultat contient variation_1j et variation_5j."""
        result = self.tech.compute_all(self.df)
        assert "variation_1j" in result.columns
        assert "variation_5j" in result.columns

    def test_compute_all_has_sma_distances(self):
        """Le resultat contient distance_sma20 et distance_sma50."""
        result = self.tech.compute_all(self.df)
        assert "distance_sma20" in result.columns
        assert "distance_sma50" in result.columns

    def test_compute_all_has_atr_pct(self):
        """Le resultat contient atr_14_pct."""
        result = self.tech.compute_all(self.df)
        assert "atr_14_pct" in result.columns

    def test_compute_all_preserves_original_columns(self):
        """Les colonnes originales sont preservees."""
        result = self.tech.compute_all(self.df)
        assert "close" in result.columns
        assert "date" in result.columns
        assert len(result) == len(self.df)


class TestGetIndicatorsAtDate:
    """Tests de recuperation des indicateurs a une date precise."""

    def setup_method(self):
        self.tech = TechnicalIndicators()
        self.df = _make_price_df(60)

    def test_get_at_date_returns_dict(self):
        """get_indicators_at_date retourne un dict."""
        enriched = self.tech.compute_all(self.df)
        # Prendre une date au milieu (apres warmup des indicateurs)
        target_date = self.df.iloc[50]["date"]
        result = self.tech.get_indicators_at_date(enriched, target_date)
        assert isinstance(result, dict)

    def test_get_at_date_has_all_features(self):
        """Le dict retourne contient toutes les features techniques."""
        enriched = self.tech.compute_all(self.df)
        target_date = self.df.iloc[50]["date"]
        result = self.tech.get_indicators_at_date(enriched, target_date)
        expected_keys = [
            "rsi_14", "macd_histogram", "bollinger_position",
            "range_position_10", "range_position_20",
            "range_amplitude_10", "range_amplitude_20",
            "volume_ratio_20", "atr_14_pct",
            "variation_1j", "variation_5j",
            "distance_sma20", "distance_sma50",
        ]
        for key in expected_keys:
            assert key in result, f"Cle manquante: {key}"

    def test_get_at_date_unknown_returns_none(self):
        """Date inconnue retourne None."""
        enriched = self.tech.compute_all(self.df)
        result = self.tech.get_indicators_at_date(enriched, "1999-01-01")
        assert result is None

    def test_short_dataframe_returns_nans(self):
        """Un DataFrame trop court (<50 jours) a des NaN mais ne crashe pas."""
        short_df = _make_price_df(15)
        enriched = self.tech.compute_all(short_df)
        # distance_sma50 sera NaN car pas assez de donnees
        assert enriched["distance_sma50"].isna().any()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_technical_indicators.py -v`
Expected: FAIL (module not found)

**Step 3: Implement TechnicalIndicators**

Creer `src/analysis/technical_indicators.py`:

```python
"""Calcul des indicateurs techniques pour le style range trading de Nicolas.

Inclut les indicateurs standards (RSI, MACD, Bollinger, ATR) et les features
specifiques au range trading (range_position, range_amplitude).
"""

import pandas as pd
import numpy as np
import ta

from loguru import logger


class TechnicalIndicators:
    """Calcule les indicateurs techniques pour un ticker."""

    def compute_all(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute toutes les colonnes d'indicateurs au DataFrame de prix.

        Le DataFrame doit contenir: date, open, high, low, close, volume.
        Retourne une copie enrichie (ne modifie pas l'original).
        """
        df = prices_df.copy()

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"].astype(float)

        # --- Indicateurs standards via librairie ta ---

        # RSI(14)
        df["rsi_14"] = ta.momentum.RSIIndicator(
            close=close, window=14
        ).rsi()

        # MACD(12, 26, 9)
        macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        df["macd_histogram"] = macd.macd_diff()

        # Bollinger Bands(20, 2)
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_range = bb_upper - bb_lower
        df["bollinger_position"] = np.where(
            bb_range > 0,
            (close - bb_lower) / bb_range,
            0.5,
        )

        # ATR(14)
        atr = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range()
        df["atr_14_pct"] = (atr / close) * 100

        # SMA(20), SMA(50), EMA(9)
        sma_20 = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
        sma_50 = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()

        df["distance_sma20"] = ((close - sma_20) / sma_20) * 100
        df["distance_sma50"] = ((close - sma_50) / sma_50) * 100

        # --- Features range trading (specifiques au style Nicolas) ---

        # Range 10 jours
        range_high_10 = high.rolling(window=10, min_periods=10).max()
        range_low_10 = low.rolling(window=10, min_periods=10).min()
        range_span_10 = range_high_10 - range_low_10
        df["range_position_10"] = np.where(
            range_span_10 > 0,
            (close - range_low_10) / range_span_10,
            0.5,
        )
        df["range_amplitude_10"] = np.where(
            range_low_10 > 0,
            (range_span_10 / range_low_10) * 100,
            0.0,
        )

        # Range 20 jours
        range_high_20 = high.rolling(window=20, min_periods=20).max()
        range_low_20 = low.rolling(window=20, min_periods=20).min()
        range_span_20 = range_high_20 - range_low_20
        df["range_position_20"] = np.where(
            range_span_20 > 0,
            (close - range_low_20) / range_span_20,
            0.5,
        )
        df["range_amplitude_20"] = np.where(
            range_low_20 > 0,
            (range_span_20 / range_low_20) * 100,
            0.0,
        )

        # --- Features derivees ---

        # Volume ratio (volume / moyenne 20j)
        vol_ma_20 = volume.rolling(window=20, min_periods=1).mean()
        df["volume_ratio_20"] = np.where(
            vol_ma_20 > 0,
            volume / vol_ma_20,
            1.0,
        )

        # Variations
        df["variation_1j"] = close.pct_change(periods=1) * 100
        df["variation_5j"] = close.pct_change(periods=5) * 100

        return df

    def get_indicators_at_date(self, enriched_df: pd.DataFrame, date: str) -> dict | None:
        """Retourne les indicateurs techniques pour une date specifique.

        Args:
            enriched_df: DataFrame retourne par compute_all().
            date: Date au format 'YYYY-MM-DD'.

        Returns:
            Dict avec les 13 features techniques, ou None si date introuvable.
        """
        mask = enriched_df["date"] == date
        if not mask.any():
            return None

        row = enriched_df[mask].iloc[0]

        feature_cols = [
            "rsi_14", "macd_histogram", "bollinger_position",
            "range_position_10", "range_position_20",
            "range_amplitude_10", "range_amplitude_20",
            "volume_ratio_20", "atr_14_pct",
            "variation_1j", "variation_5j",
            "distance_sma20", "distance_sma50",
        ]

        result = {}
        for col in feature_cols:
            val = row[col]
            # Convertir numpy types en float Python
            result[col] = float(val) if pd.notna(val) else None

        return result
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_technical_indicators.py -v`
Expected: 17 PASS

**Step 5: Run all tests (regression)**

Run: `uv run pytest tests/ -v`
Expected: 136 PASS (119 + 17)

**Step 6: Commit**

```bash
git add src/analysis/technical_indicators.py tests/test_technical_indicators.py
git commit -m "feat(analysis): add TechnicalIndicators with range trading detection"
```

---

### Task 4: FeatureEngine — Assemblage du vecteur de features

**Files:**
- Create: `src/analysis/feature_engine.py`
- Create: `tests/test_feature_engine.py`

**Step 1: Write the failing tests**

Creer `tests/test_feature_engine.py`:

```python
"""Tests pour le feature engine — assemblage du vecteur de features par trade."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.core.database import Database
from src.analysis.feature_engine import FeatureEngine


def _seed_test_db(db: Database):
    """Peuple une base de test avec des donnees realistes."""
    # 2 trades SANOFI (1 gagnant, 1 perdant)
    db.insert_trades_batch([
        {
            "isin": "FR0000120578", "nom_action": "SANOFI",
            "date_achat": "2025-07-10", "date_vente": "2025-07-20",
            "prix_achat": 95.0, "prix_vente": 100.0, "quantite": 10,
            "rendement_brut_pct": 5.26, "rendement_net_pct": 5.0,
            "duree_jours": 10, "frais_totaux": 2.5, "statut": "CLOTURE",
        },
        {
            "isin": "FR0000120578", "nom_action": "SANOFI",
            "date_achat": "2025-08-15", "date_vente": "2025-08-25",
            "prix_achat": 100.0, "prix_vente": 97.0, "quantite": 10,
            "rendement_brut_pct": -3.0, "rendement_net_pct": -3.5,
            "duree_jours": 10, "frais_totaux": 2.5, "statut": "CLOTURE",
        },
    ])

    # Prix synthetiques pour SAN.PA (60 jours avant + periode des trades)
    np.random.seed(42)
    dates = pd.bdate_range("2025-05-01", "2025-09-15")
    n = len(dates)
    t = np.linspace(0, 6 * np.pi, n)
    close = 97 + 5 * np.sin(t) + np.random.normal(0, 0.5, n)

    prices = []
    for i, d in enumerate(dates):
        prices.append({
            "ticker": "SAN.PA",
            "date": d.strftime("%Y-%m-%d"),
            "open": round(close[i] - 0.5, 4),
            "high": round(close[i] + 1.0, 4),
            "low": round(close[i] - 1.0, 4),
            "close": round(close[i], 4),
            "volume": int(100000 + np.random.randint(-20000, 20000)),
        })
    db.insert_prices_batch(prices)

    # News pour le trade 1
    db.insert_news_batch([
        {
            "ticker": "SAN.PA", "title": "Sanofi: resultats T2 solides",
            "source": "Reuters", "url": "https://ex.com/san1",
            "published_at": "2025-07-09", "description": "Bons resultats",
            "sentiment": 0.6, "source_api": "alpha_vantage",
        },
        {
            "ticker": "SAN.PA", "title": "FDA approves Sanofi new drug",
            "source": "Bloomberg", "url": "https://ex.com/san2",
            "published_at": "2025-07-10", "description": "Drug approved by FDA",
            "sentiment": 0.8, "source_api": "gnews",
        },
    ])
    # Catalyseurs pour trade 1
    db.insert_catalyseurs_batch([
        {"trade_id": 1, "news_id": 1, "score_pertinence": 0.9,
         "distance_jours": -1, "match_texte": 1},
        {"trade_id": 1, "news_id": 2, "score_pertinence": 1.0,
         "distance_jours": 0, "match_texte": 1},
    ])
    # Pas de catalyseurs pour trade 2


class TestBuildTradeFeatures:
    """Tests de construction des features pour un trade."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        _seed_test_db(self.db)
        self.engine = FeatureEngine(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_build_returns_dict(self):
        """build_trade_features retourne un dict."""
        trades = self.db.get_all_trades()
        result = self.engine.build_trade_features(trades[0])
        assert isinstance(result, dict)

    def test_build_has_technical_features(self):
        """Le dict contient les features techniques."""
        trades = self.db.get_all_trades()
        result = self.engine.build_trade_features(trades[0])
        assert "rsi_14" in result
        assert "range_position_10" in result
        assert "range_position_20" in result
        assert "bollinger_position" in result

    def test_build_has_catalyst_features(self):
        """Le dict contient les features catalyseur."""
        trades = self.db.get_all_trades()
        result = self.engine.build_trade_features(trades[0])
        assert "catalyst_type" in result
        assert "nb_catalysts" in result
        assert "best_catalyst_score" in result
        assert "has_text_match" in result

    def test_build_has_context_features(self):
        """Le dict contient les features contexte."""
        trades = self.db.get_all_trades()
        result = self.engine.build_trade_features(trades[0])
        assert "day_of_week" in result
        assert "nb_previous_trades" in result
        assert "previous_win_rate" in result

    def test_build_has_target(self):
        """Le dict contient la target is_winner."""
        trades = self.db.get_all_trades()
        # Trade 1: gagnant (+5.26%)
        result = self.engine.build_trade_features(trades[0])
        assert result["is_winner"] == 1
        # Trade 2: perdant (-3%)
        result = self.engine.build_trade_features(trades[1])
        assert result["is_winner"] == 0

    def test_trade_with_catalysts_has_type(self):
        """Un trade avec catalyseurs a un catalyst_type != TECHNICAL."""
        trades = self.db.get_all_trades()
        result = self.engine.build_trade_features(trades[0])
        # News "resultats T2" = EARNINGS, "FDA approves" = FDA_REGULATORY
        assert result["catalyst_type"] != "TECHNICAL"
        assert result["nb_catalysts"] == 2

    def test_trade_without_catalysts_is_technical(self):
        """Un trade sans catalyseurs a catalyst_type == TECHNICAL."""
        trades = self.db.get_all_trades()
        result = self.engine.build_trade_features(trades[1])
        assert result["catalyst_type"] == "TECHNICAL"
        assert result["nb_catalysts"] == 0

    def test_context_second_trade_has_history(self):
        """Le 2e trade SANOFI connait l'historique (1 trade precedent)."""
        trades = self.db.get_all_trades()
        result = self.engine.build_trade_features(trades[1])
        assert result["nb_previous_trades"] == 1
        assert result["previous_win_rate"] == 1.0  # Le 1er trade etait gagnant


class TestBuildAllFeatures:
    """Tests de construction de la matrice complete."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        _seed_test_db(self.db)
        self.engine = FeatureEngine(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_build_all_returns_dataframe(self):
        """build_all_features retourne un DataFrame."""
        result = self.engine.build_all_features()
        assert isinstance(result, pd.DataFrame)

    def test_build_all_has_correct_rows(self):
        """Le DataFrame a autant de lignes que de trades clotures avec donnees."""
        result = self.engine.build_all_features()
        # 2 trades clotures, les deux devraient avoir assez de donnees
        assert len(result) == 2

    def test_build_all_has_target_column(self):
        """Le DataFrame a la colonne target."""
        result = self.engine.build_all_features()
        assert "is_winner" in result.columns

    def test_build_all_has_trade_id(self):
        """Le DataFrame a trade_id pour identifier chaque trade."""
        result = self.engine.build_all_features()
        assert "trade_id" in result.columns

    def test_feature_names_list(self):
        """get_feature_names retourne la liste des features (sans target)."""
        names = self.engine.get_feature_names()
        assert isinstance(names, list)
        assert "is_winner" not in names
        assert "trade_id" not in names
        assert "rsi_14" in names
        assert "catalyst_type" in names
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_feature_engine.py -v`
Expected: FAIL (module not found)

**Step 3: Implement FeatureEngine**

Creer `src/analysis/feature_engine.py`:

```python
"""Assemblage du vecteur de features pour chaque trade de Nicolas.

Combine les indicateurs techniques (au moment de l'achat), le type de
catalyseur, et le contexte personnel (historique de trading sur l'action).
"""

import pandas as pd
from loguru import logger

from src.core.database import Database
from src.data_collection.ticker_mapper import TickerMapper, TickerNotFoundError
from src.analysis.technical_indicators import TechnicalIndicators
from src.analysis.news_classifier import NewsClassifier


# Colonnes de features techniques retournees par TechnicalIndicators
TECHNICAL_FEATURES = [
    "rsi_14", "macd_histogram", "bollinger_position",
    "range_position_10", "range_position_20",
    "range_amplitude_10", "range_amplitude_20",
    "volume_ratio_20", "atr_14_pct",
    "variation_1j", "variation_5j",
    "distance_sma20", "distance_sma50",
]

# Colonnes de features catalyseur
CATALYST_FEATURES = [
    "catalyst_type", "nb_catalysts", "best_catalyst_score",
    "has_text_match", "sentiment_avg", "nb_news_sources",
]

# Colonnes de features contexte
CONTEXT_FEATURES = [
    "day_of_week", "nb_previous_trades",
    "previous_win_rate", "days_since_last_trade",
]


class FeatureEngine:
    """Assemble le vecteur de features pour chaque trade de Nicolas."""

    def __init__(self, db: Database):
        self.db = db
        self.tech = TechnicalIndicators()
        self.classifier = NewsClassifier()
        self.mapper = TickerMapper()
        # Cache des DataFrames de prix enrichis par ticker
        self._price_cache: dict[str, pd.DataFrame] = {}

    def _get_enriched_prices(self, ticker: str) -> pd.DataFrame | None:
        """Recupere et enrichit les prix pour un ticker (avec cache)."""
        if ticker in self._price_cache:
            return self._price_cache[ticker]

        prices = self.db.get_prices(ticker)
        if len(prices) < 20:
            logger.warning(f"Pas assez de prix pour {ticker}: {len(prices)} jours")
            return None

        df = pd.DataFrame(prices)
        enriched = self.tech.compute_all(df)
        self._price_cache[ticker] = enriched
        return enriched

    def _build_technical_features(self, ticker: str, date_achat: str) -> dict | None:
        """Construit les features techniques pour un trade a la date d'achat."""
        enriched = self._get_enriched_prices(ticker)
        if enriched is None:
            return None
        return self.tech.get_indicators_at_date(enriched, date_achat)

    def _build_catalyst_features(self, trade: dict) -> dict:
        """Construit les features catalyseur pour un trade."""
        trade_id = trade["id"]
        catalyseurs = self.db.get_catalyseurs(trade_id)

        if not catalyseurs:
            return {
                "catalyst_type": "TECHNICAL",
                "nb_catalysts": 0,
                "best_catalyst_score": 0.0,
                "has_text_match": 0,
                "sentiment_avg": 0.0,
                "nb_news_sources": 0,
            }

        # Recuperer les news liees aux catalyseurs pour classifier
        news_with_types = []
        sentiments = []
        sources = set()

        for cat in catalyseurs:
            news_id = cat["news_id"]
            # Recuperer la news depuis la base
            conn = self.db._connect()
            news_row = conn.execute(
                "SELECT * FROM news WHERE id = ?", (news_id,)
            ).fetchone()
            conn.close()

            if news_row:
                news = dict(news_row)
                cat_type = self.classifier.classify(
                    news.get("title", ""), news.get("description")
                )
                news_with_types.append({
                    "catalyst_type": cat_type,
                    "score_pertinence": cat["score_pertinence"],
                })
                if news.get("sentiment") is not None:
                    sentiments.append(news["sentiment"])
                if news.get("source_api"):
                    sources.add(news["source_api"])

        # Resumer les types pour ce trade
        summary = self.classifier.summarize_for_trade(news_with_types)

        return {
            "catalyst_type": summary["primary_type"],
            "nb_catalysts": len(catalyseurs),
            "best_catalyst_score": max(c["score_pertinence"] for c in catalyseurs),
            "has_text_match": 1 if any(c["match_texte"] for c in catalyseurs) else 0,
            "sentiment_avg": sum(sentiments) / len(sentiments) if sentiments else 0.0,
            "nb_news_sources": len(sources),
        }

    def _build_context_features(self, trade: dict, all_trades: list[dict]) -> dict:
        """Construit les features de contexte personnel.

        Calcule l'historique de Nicolas sur cette action AVANT ce trade.
        """
        from datetime import datetime

        nom_action = trade["nom_action"]
        date_achat = trade["date_achat"][:10]

        # Trades precedents sur la meme action, avant ce trade
        previous = [
            t for t in all_trades
            if t["nom_action"] == nom_action
            and t["date_achat"][:10] < date_achat
            and t["statut"] == "CLOTURE"
        ]

        nb_previous = len(previous)
        if nb_previous > 0:
            wins = sum(1 for t in previous if t["rendement_brut_pct"] > 0)
            win_rate = wins / nb_previous
            last_trade_date = max(t["date_vente"][:10] for t in previous if t["date_vente"])
            last_dt = datetime.strptime(last_trade_date, "%Y-%m-%d")
            current_dt = datetime.strptime(date_achat, "%Y-%m-%d")
            days_since = (current_dt - last_dt).days
        else:
            win_rate = 0.0
            days_since = -1  # Pas de trade precedent

        # Jour de la semaine (0=lundi, 4=vendredi)
        day_of_week = datetime.strptime(date_achat, "%Y-%m-%d").weekday()

        return {
            "day_of_week": day_of_week,
            "nb_previous_trades": nb_previous,
            "previous_win_rate": round(win_rate, 4),
            "days_since_last_trade": days_since,
        }

    def build_trade_features(self, trade: dict, all_trades: list[dict] | None = None) -> dict | None:
        """Construit le vecteur de features complet pour UN trade.

        Args:
            trade: Dict du trade (de db.get_all_trades()).
            all_trades: Liste de tous les trades pour le contexte.
                        Si None, les recupere depuis la base.

        Returns:
            Dict avec ~25 features + target, ou None si pas assez de donnees.
        """
        if all_trades is None:
            all_trades = self.db.get_all_trades()

        nom_action = trade["nom_action"]
        date_achat = trade["date_achat"][:10]

        # Ticker Yahoo
        try:
            ticker = self.mapper.get_ticker(nom_action)
        except TickerNotFoundError:
            logger.warning(f"Ticker inconnu pour '{nom_action}', skip")
            return None

        # Features techniques
        tech_features = self._build_technical_features(ticker, date_achat)
        if tech_features is None:
            logger.warning(f"Pas de donnees techniques pour {nom_action} au {date_achat}")
            return None

        # Features catalyseur
        cat_features = self._build_catalyst_features(trade)

        # Features contexte
        ctx_features = self._build_context_features(trade, all_trades)

        # Target
        is_winner = 1 if trade["rendement_brut_pct"] > 0 else 0

        # Assembler
        features = {
            "trade_id": trade["id"],
            **tech_features,
            **cat_features,
            **ctx_features,
            "is_winner": is_winner,
        }

        return features

    def build_all_features(self) -> pd.DataFrame:
        """Construit la matrice de features pour tous les trades clotures.

        Retourne un DataFrame avec ~25 colonnes de features + target + trade_id.
        Les trades sans donnees suffisantes sont exclus (avec warning).
        """
        trades = self.db.get_all_trades()
        closed_trades = [t for t in trades if t["statut"] == "CLOTURE"]

        logger.info(f"Construction features pour {len(closed_trades)} trades clotures")

        rows = []
        skipped = 0

        for trade in closed_trades:
            features = self.build_trade_features(trade, all_trades=trades)
            if features is not None:
                rows.append(features)
            else:
                skipped += 1

        if skipped > 0:
            logger.warning(f"{skipped} trades exclus (donnees insuffisantes)")

        df = pd.DataFrame(rows)
        logger.info(f"Matrice de features: {len(df)} lignes x {len(df.columns)} colonnes")
        return df

    def get_feature_names(self) -> list[str]:
        """Liste ordonnee des noms de features (sans target ni trade_id)."""
        return TECHNICAL_FEATURES + CATALYST_FEATURES + CONTEXT_FEATURES
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_feature_engine.py -v`
Expected: 13 PASS

**Step 5: Run all tests (regression)**

Run: `uv run pytest tests/ -v`
Expected: 149 PASS (136 + 13)

**Step 6: Commit**

```bash
git add src/analysis/feature_engine.py tests/test_feature_engine.py
git commit -m "feat(analysis): add FeatureEngine - assembles trade feature vectors"
```

---

### Task 5: Trainer — Entrainement XGBoost

**Files:**
- Create: `src/model/__init__.py`
- Create: `src/model/trainer.py`
- Create: `tests/test_trainer.py`

**Step 1: Create model package**

Creer `src/model/__init__.py` (fichier vide).

**Step 2: Write the failing tests**

Creer `tests/test_trainer.py`:

```python
"""Tests pour l'entrainement du modele XGBoost."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.model.trainer import Trainer


def _make_feature_df(n_samples: int = 50) -> pd.DataFrame:
    """Cree un DataFrame de features synthetiques.

    Simule le style Nicolas: trades gagnants quand range_position < 0.3 + catalyseur.
    """
    np.random.seed(42)

    # Features techniques
    range_pos = np.random.uniform(0, 1, n_samples)
    rsi = np.random.uniform(20, 80, n_samples)
    macd_hist = np.random.normal(0, 0.5, n_samples)
    bollinger_pos = np.random.uniform(0, 1, n_samples)
    range_amp_10 = np.random.uniform(5, 25, n_samples)
    range_amp_20 = np.random.uniform(5, 30, n_samples)
    vol_ratio = np.random.uniform(0.5, 3, n_samples)
    atr_pct = np.random.uniform(1, 5, n_samples)
    var_1j = np.random.normal(0, 2, n_samples)
    var_5j = np.random.normal(0, 4, n_samples)
    dist_sma20 = np.random.normal(0, 3, n_samples)
    dist_sma50 = np.random.normal(0, 5, n_samples)

    # Features catalyseur
    cat_types = np.random.choice(
        ["EARNINGS", "FDA_REGULATORY", "UPGRADE", "CONTRACT", "TECHNICAL"],
        n_samples,
    )
    nb_cats = np.random.randint(0, 10, n_samples)
    best_score = np.random.uniform(0.3, 1.0, n_samples)
    has_match = np.random.randint(0, 2, n_samples)
    sentiment = np.random.uniform(-0.5, 0.8, n_samples)
    nb_sources = np.random.randint(0, 4, n_samples)

    # Features contexte
    dow = np.random.randint(0, 5, n_samples)
    nb_prev = np.random.randint(0, 15, n_samples)
    prev_wr = np.random.uniform(0.5, 1.0, n_samples)
    days_since = np.random.randint(-1, 60, n_samples)

    # Target: gagnant si range_position bas + catalyseur present
    prob = 1 / (1 + np.exp(3 * (range_pos - 0.4) - 0.5 * (nb_cats > 0).astype(float)))
    is_winner = (np.random.random(n_samples) < prob).astype(int)

    return pd.DataFrame({
        "trade_id": range(1, n_samples + 1),
        "range_position_10": range_pos,
        "range_position_20": range_pos + np.random.normal(0, 0.05, n_samples),
        "range_amplitude_10": range_amp_10,
        "range_amplitude_20": range_amp_20,
        "rsi_14": rsi,
        "macd_histogram": macd_hist,
        "bollinger_position": bollinger_pos,
        "volume_ratio_20": vol_ratio,
        "atr_14_pct": atr_pct,
        "variation_1j": var_1j,
        "variation_5j": var_5j,
        "distance_sma20": dist_sma20,
        "distance_sma50": dist_sma50,
        "catalyst_type": cat_types,
        "nb_catalysts": nb_cats,
        "best_catalyst_score": best_score,
        "has_text_match": has_match,
        "sentiment_avg": sentiment,
        "nb_news_sources": nb_sources,
        "day_of_week": dow,
        "nb_previous_trades": nb_prev,
        "previous_win_rate": prev_wr,
        "days_since_last_trade": days_since,
        "is_winner": is_winner,
    })


class TestPrepareData:
    """Tests de preparation des donnees."""

    def setup_method(self):
        self.trainer = Trainer()

    def test_prepare_data_splits_x_y(self):
        """prepare_data retourne X (features) et y (target)."""
        df = _make_feature_df(50)
        X, y = self.trainer.prepare_data(df)
        assert len(X) == 50
        assert len(y) == 50
        assert "is_winner" not in X.columns
        assert "trade_id" not in X.columns

    def test_prepare_data_encodes_catalyst_type(self):
        """catalyst_type est encode en numerique."""
        df = _make_feature_df(50)
        X, y = self.trainer.prepare_data(df)
        assert X["catalyst_type"].dtype in [np.int64, np.int32, np.float64, int]


class TestTrain:
    """Tests d'entrainement du modele."""

    def setup_method(self):
        self.trainer = Trainer()
        self.df = _make_feature_df(50)

    def test_train_returns_metrics(self):
        """train() retourne un dict avec des metriques."""
        X, y = self.trainer.prepare_data(self.df)
        metrics = self.trainer.train(X, y)
        assert "train_accuracy" in metrics
        assert metrics["train_accuracy"] > 0

    def test_model_is_set_after_train(self):
        """Le modele est accessible apres entrainement."""
        X, y = self.trainer.prepare_data(self.df)
        self.trainer.train(X, y)
        assert self.trainer.model is not None

    def test_predict_after_train(self):
        """Peut predire apres entrainement."""
        X, y = self.trainer.prepare_data(self.df)
        self.trainer.train(X, y)
        predictions = self.trainer.predict(X)
        assert len(predictions) == len(X)
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba_after_train(self):
        """Peut obtenir des probabilites apres entrainement."""
        X, y = self.trainer.prepare_data(self.df)
        self.trainer.train(X, y)
        probas = self.trainer.predict_proba(X)
        assert len(probas) == len(X)
        assert all(0 <= p <= 1 for p in probas)


class TestWalkForward:
    """Tests de validation walk-forward."""

    def setup_method(self):
        self.trainer = Trainer()

    def test_walk_forward_returns_metrics(self):
        """walk_forward_validate retourne des metriques."""
        df = _make_feature_df(50)
        # Simuler des dates pour le split
        dates = pd.bdate_range("2025-06-01", periods=50)
        df["date_achat"] = [d.strftime("%Y-%m-%d") for d in dates]

        results = self.trainer.walk_forward_validate(df, split_date="2025-07-25")
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results
        assert "baseline_accuracy" in results

    def test_walk_forward_has_predictions(self):
        """Les predictions detaillees sont retournees."""
        df = _make_feature_df(50)
        dates = pd.bdate_range("2025-06-01", periods=50)
        df["date_achat"] = [d.strftime("%Y-%m-%d") for d in dates]

        results = self.trainer.walk_forward_validate(df, split_date="2025-07-25")
        assert "predictions" in results
        assert len(results["predictions"]) > 0


class TestSaveLoad:
    """Tests de sauvegarde/chargement du modele."""

    def setup_method(self):
        self.trainer = Trainer()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, "test_model.joblib")

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_save_and_load(self):
        """Sauvegarde et recharge le modele."""
        df = _make_feature_df(50)
        X, y = self.trainer.prepare_data(df)
        self.trainer.train(X, y)
        self.trainer.save_model(self.model_path)

        # Charger dans un nouveau trainer
        trainer2 = Trainer()
        trainer2.load_model(self.model_path)
        preds = trainer2.predict(X)
        assert len(preds) == len(X)

    def test_save_without_train_raises(self):
        """Sauvegarder sans entrainer leve une erreur."""
        with pytest.raises(ValueError, match="pas entraine"):
            self.trainer.save_model(self.model_path)
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_trainer.py -v`
Expected: FAIL (module not found)

**Step 4: Implement Trainer**

Creer `src/model/trainer.py`:

```python
"""Entrainement du modele XGBoost sur les trades de Nicolas.

Apprend a predire si Nicolas gagnerait un trade en se basant sur
la combinaison indicateurs techniques + type de catalyseur + contexte.
"""

import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from loguru import logger


# Mapping catalyst_type -> code numerique
CATALYST_TYPE_ENCODING = {
    "TECHNICAL": 0,
    "UNKNOWN": 1,
    "OTHER_NEGATIVE": 2,
    "OTHER_POSITIVE": 3,
    "SECTOR_MACRO": 4,
    "INSIDER": 5,
    "DIVIDEND": 6,
    "RESTRUCTURING": 7,
    "CONTRACT": 8,
    "DOWNGRADE": 9,
    "UPGRADE": 10,
    "EARNINGS": 11,
    "FDA_REGULATORY": 12,
}


class Trainer:
    """Entraine un XGBoost sur les trades historiques de Nicolas."""

    def __init__(self):
        self.model = None
        self.feature_names: list[str] = []

    def prepare_data(self, features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare X (features) et y (target) a partir du DataFrame de features.

        Encode catalyst_type en numerique. Retire trade_id et is_winner de X.
        """
        df = features_df.copy()

        # Encoder catalyst_type
        df["catalyst_type"] = df["catalyst_type"].map(CATALYST_TYPE_ENCODING).fillna(1)

        # Separer X et y
        y = df["is_winner"]
        drop_cols = ["is_winner", "trade_id"]
        # Aussi retirer date_achat si present (pas une feature)
        if "date_achat" in df.columns:
            drop_cols.append("date_achat")

        X = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Remplacer les NaN restants par 0
        X = X.fillna(0)

        self.feature_names = list(X.columns)
        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Entraine le modele XGBoost.

        Parametres optimises pour un petit dataset (~141 samples):
        - max_depth=4 (eviter overfitting)
        - n_estimators=100
        - learning_rate=0.1
        - scale_pos_weight ajuste au ratio negatifs/positifs
        """
        # Calculer le poids pour gerer le desequilibre
        n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        self.model = xgb.XGBClassifier(
            max_depth=4,
            n_estimators=100,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False,
        )
        self.model.fit(X, y, verbose=False)

        # Metriques d'entrainement
        train_preds = self.model.predict(X)
        train_acc = accuracy_score(y, train_preds)

        logger.info(f"Modele entraine: accuracy={train_acc:.3f}, "
                     f"samples={len(y)}, pos={n_pos}, neg={n_neg}")

        return {"train_accuracy": train_acc}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predit les classes (0/1) pour les features donnees."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Retourne les probabilites de la classe positive (gagnant)."""
        return self.model.predict_proba(X)[:, 1]

    def walk_forward_validate(self, features_df: pd.DataFrame,
                               split_date: str = "2025-12-01") -> dict:
        """Validation walk-forward: train avant split_date, test apres.

        Args:
            features_df: DataFrame complet avec features + target + date_achat.
            split_date: Date de separation train/test (YYYY-MM-DD).

        Returns:
            Dict avec metriques et predictions detaillees.
        """
        df = features_df.copy()

        train_mask = df["date_achat"] < split_date
        test_mask = df["date_achat"] >= split_date

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(test_df) == 0:
            logger.warning("Pas de donnees de test apres le split")
            return {"error": "Pas de donnees de test"}

        # Preparer et entrainer sur le train set
        X_train, y_train = self.prepare_data(train_df)
        self.train(X_train, y_train)

        # Predire sur le test set
        X_test, y_test = self.prepare_data(test_df)
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        # Baseline naif: toujours predire gagnant
        baseline_preds = np.ones(len(y_test))
        baseline_acc = accuracy_score(y_test, baseline_preds)

        # Metriques
        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "baseline_accuracy": baseline_acc,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "predictions": [],
        }

        # Predictions detaillees
        for i, (idx, row) in enumerate(test_df.iterrows()):
            results["predictions"].append({
                "trade_id": int(row.get("trade_id", i)),
                "actual": int(y_test.iloc[i]),
                "predicted": int(y_pred[i]),
                "proba": round(float(y_proba[i]), 3),
            })

        logger.info(f"Walk-forward: accuracy={results['accuracy']:.3f} "
                     f"(baseline={baseline_acc:.3f}), "
                     f"train={results['train_size']}, test={results['test_size']}")

        return results

    def save_model(self, path: str = "data/models/nicolas_v1.joblib"):
        """Sauvegarde le modele entraine."""
        if self.model is None:
            raise ValueError("Le modele n'est pas entraine")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
        }, path)
        logger.info(f"Modele sauvegarde: {path}")

    def load_model(self, path: str = "data/models/nicolas_v1.joblib"):
        """Charge un modele sauvegarde."""
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        logger.info(f"Modele charge: {path}")
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_trainer.py -v`
Expected: 11 PASS

**Step 6: Run all tests (regression)**

Run: `uv run pytest tests/ -v`
Expected: 160 PASS (149 + 11)

**Step 7: Commit**

```bash
git add src/model/__init__.py src/model/trainer.py tests/test_trainer.py
git commit -m "feat(model): add Trainer - XGBoost training with walk-forward validation"
```

---

### Task 6: Evaluator — Analyse du modele

**Files:**
- Create: `src/model/evaluator.py`
- Create: `tests/test_evaluator.py`

**Step 1: Write the failing tests**

Creer `tests/test_evaluator.py`:

```python
"""Tests pour l'evaluateur du modele."""

import numpy as np
import pandas as pd
import pytest

from src.model.evaluator import Evaluator


class TestFeatureImportance:
    """Tests de feature importance."""

    def setup_method(self):
        self.evaluator = Evaluator()

    def test_returns_dataframe(self):
        """feature_importance retourne un DataFrame."""
        # Simuler un modele avec feature_importances_
        from unittest.mock import MagicMock
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.3, 0.2, 0.1, 0.05, 0.35])
        names = ["range_position_10", "rsi_14", "catalyst_type", "day_of_week", "nb_catalysts"]

        result = self.evaluator.feature_importance(mock_model, names)
        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "importance" in result.columns

    def test_sorted_by_importance(self):
        """Les features sont triees par importance decroissante."""
        from unittest.mock import MagicMock
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.1, 0.3, 0.6])
        names = ["a", "b", "c"]

        result = self.evaluator.feature_importance(mock_model, names)
        assert result.iloc[0]["feature"] == "c"
        assert result.iloc[0]["importance"] == 0.6


class TestErrorAnalysis:
    """Tests d'analyse des erreurs."""

    def setup_method(self):
        self.evaluator = Evaluator()

    def test_identifies_false_positives(self):
        """Identifie les faux positifs (predit gagnant mais perdant)."""
        predictions = [
            {"trade_id": 1, "actual": 0, "predicted": 1, "proba": 0.7},
            {"trade_id": 2, "actual": 1, "predicted": 1, "proba": 0.9},
            {"trade_id": 3, "actual": 1, "predicted": 0, "proba": 0.3},
        ]
        result = self.evaluator.error_analysis(predictions)
        assert len(result["false_positives"]) == 1
        assert result["false_positives"][0]["trade_id"] == 1

    def test_identifies_false_negatives(self):
        """Identifie les faux negatifs (predit perdant mais gagnant)."""
        predictions = [
            {"trade_id": 1, "actual": 0, "predicted": 1, "proba": 0.7},
            {"trade_id": 2, "actual": 1, "predicted": 0, "proba": 0.3},
        ]
        result = self.evaluator.error_analysis(predictions)
        assert len(result["false_negatives"]) == 1
        assert result["false_negatives"][0]["trade_id"] == 2


class TestCompareBaseline:
    """Tests de comparaison avec le baseline."""

    def setup_method(self):
        self.evaluator = Evaluator()

    def test_compare_returns_dict(self):
        """compare_to_baseline retourne un dict."""
        y_true = np.array([1, 1, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 1])
        result = self.evaluator.compare_to_baseline(y_true, y_pred)
        assert "model_accuracy" in result
        assert "baseline_accuracy" in result
        assert "improvement" in result

    def test_improvement_calculation(self):
        """Le calcul de l'amelioration est correct."""
        y_true = np.array([1, 1, 1, 1, 0])  # 80% positifs
        y_pred = np.array([1, 1, 1, 0, 0])  # 80% accuracy
        result = self.evaluator.compare_to_baseline(y_true, y_pred)
        assert result["baseline_accuracy"] == 0.8  # Toujours predire 1
        assert result["model_accuracy"] == 0.8
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluator.py -v`
Expected: FAIL (module not found)

**Step 3: Implement Evaluator**

Creer `src/model/evaluator.py`:

```python
"""Analyse des performances et interpretabilite du modele.

Repond aux questions cles:
- Est-ce que le modele bat le baseline naif?
- Quelles features comptent le plus?
- Sur quoi le modele se trompe?
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from loguru import logger


class Evaluator:
    """Analyse les performances et l'interpretabilite du modele."""

    def feature_importance(self, model, feature_names: list[str]) -> pd.DataFrame:
        """Retourne les features triees par importance.

        Si range_position et catalyst_type sont dans le top 5, le modele
        a bien appris le style Nicolas (range trading + catalyseurs).
        """
        importances = model.feature_importances_
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        })
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def error_analysis(self, predictions: list[dict]) -> dict:
        """Analyse les trades mal predits.

        Returns:
            {
                "false_positives": list — Predits gagnants mais perdants (DANGER: on perd de l'argent)
                "false_negatives": list — Predits perdants mais gagnants (on rate une opportunite)
                "total_errors": int
            }
        """
        false_positives = [
            p for p in predictions
            if p["predicted"] == 1 and p["actual"] == 0
        ]
        false_negatives = [
            p for p in predictions
            if p["predicted"] == 0 and p["actual"] == 1
        ]

        return {
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "total_errors": len(false_positives) + len(false_negatives),
        }

    def compare_to_baseline(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compare le modele au baseline naif (toujours predire gagnant).

        Le baseline reflette le win rate de Nicolas (88%).
        Le modele doit faire mieux que simplement dire "achete tout".
        """
        baseline_preds = np.ones(len(y_true))
        baseline_acc = accuracy_score(y_true, baseline_preds)
        model_acc = accuracy_score(y_true, y_pred)

        improvement = model_acc - baseline_acc

        return {
            "model_accuracy": round(model_acc, 4),
            "baseline_accuracy": round(baseline_acc, 4),
            "improvement": round(improvement, 4),
            "beats_baseline": improvement > 0,
        }

    def print_report(self, walk_forward_results: dict,
                     importance_df: pd.DataFrame | None = None):
        """Affiche un rapport complet en console."""
        r = walk_forward_results

        print("=" * 60)
        print("RAPPORT D'EVALUATION — Modele Nicolas v1")
        print("=" * 60)

        print(f"\nDonnees: {r.get('train_size', '?')} train, {r.get('test_size', '?')} test")
        print(f"\nMetriques sur le test set:")
        print(f"  Accuracy:   {r.get('accuracy', 0):.1%}")
        print(f"  Precision:  {r.get('precision', 0):.1%}")
        print(f"  Recall:     {r.get('recall', 0):.1%}")
        print(f"  F1-Score:   {r.get('f1', 0):.1%}")
        print(f"  Baseline:   {r.get('baseline_accuracy', 0):.1%} (toujours predire gagnant)")

        if r.get("accuracy", 0) > r.get("baseline_accuracy", 0):
            print(f"\n  >> Le modele BAT le baseline de "
                  f"+{(r['accuracy'] - r['baseline_accuracy']):.1%}")
        else:
            print(f"\n  >> Le modele ne bat PAS le baseline")

        # Matrice de confusion
        cm = r.get("confusion_matrix")
        if cm:
            print(f"\nMatrice de confusion:")
            print(f"  Predit Perdant | Predit Gagnant")
            print(f"  Vrai Perdant:  {cm[0][0]:>6} | {cm[0][1]:>6}")
            print(f"  Vrai Gagnant:  {cm[1][0]:>6} | {cm[1][1]:>6}")

        # Feature importance
        if importance_df is not None:
            print(f"\nTop 10 features les plus importantes:")
            for i, row in importance_df.head(10).iterrows():
                bar = "#" * int(row["importance"] * 50)
                print(f"  {i+1:2}. {row['feature']:25s} {row['importance']:.3f} {bar}")

        # Analyse erreurs
        preds = r.get("predictions", [])
        if preds:
            errors = self.error_analysis(preds)
            print(f"\nAnalyse des erreurs ({errors['total_errors']} erreurs):")
            if errors["false_positives"]:
                print(f"  Faux positifs (DANGEREUX — on perd de l'argent):")
                for fp in errors["false_positives"]:
                    print(f"    Trade #{fp['trade_id']}: predit gagnant (p={fp['proba']:.2f}) "
                          f"mais perdant")
            if errors["false_negatives"]:
                print(f"  Faux negatifs (on rate des opportunites):")
                for fn in errors["false_negatives"]:
                    print(f"    Trade #{fn['trade_id']}: predit perdant (p={fn['proba']:.2f}) "
                          f"mais gagnant")

        print("\n" + "=" * 60)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluator.py -v`
Expected: 6 PASS

**Step 5: Run all tests (regression)**

Run: `uv run pytest tests/ -v`
Expected: 166 PASS (160 + 6)

**Step 6: Commit**

```bash
git add src/model/evaluator.py tests/test_evaluator.py
git commit -m "feat(model): add Evaluator - feature importance and error analysis"
```

---

### Task 7: Scripts CLI — train_model.py et analyze_features.py

**Files:**
- Create: `scripts/train_model.py`
- Create: `scripts/analyze_features.py`

**Step 1: Creer train_model.py**

```python
"""Script d'entrainement du modele Nicolas.

Usage:
    uv run python scripts/train_model.py                # Entrainer + evaluer
    uv run python scripts/train_model.py --features      # Voir les features seulement
    uv run python scripts/train_model.py --importance     # Feature importance
"""

import argparse

from src.core.database import Database
from src.analysis.feature_engine import FeatureEngine
from src.model.trainer import Trainer
from src.model.evaluator import Evaluator

DB_PATH = "data/trades.db"
MODEL_PATH = "data/models/nicolas_v1.joblib"


def main():
    parser = argparse.ArgumentParser(description="Entrainement du modele Nicolas")
    parser.add_argument("--features", action="store_true",
                        help="Afficher les features seulement (pas d'entrainement)")
    parser.add_argument("--importance", action="store_true",
                        help="Afficher le feature importance d'un modele existant")
    parser.add_argument("--split-date", default="2025-12-01",
                        help="Date de split walk-forward (defaut: 2025-12-01)")
    args = parser.parse_args()

    db = Database(DB_PATH)
    db.init_db()
    engine = FeatureEngine(db)

    # Construire les features
    print("Construction des features...")
    features_df = engine.build_all_features()
    print(f"Matrice: {len(features_df)} trades x {len(features_df.columns)} colonnes")
    print(f"Gagnants: {(features_df['is_winner'] == 1).sum()}")
    print(f"Perdants: {(features_df['is_winner'] == 0).sum()}")

    if args.features:
        print(f"\nFeatures disponibles:")
        for name in engine.get_feature_names():
            print(f"  - {name}")
        print(f"\nApercu des 5 premiers trades:")
        print(features_df.head().to_string())
        return

    # Entrainer et evaluer
    trainer = Trainer()
    evaluator = Evaluator()

    # Ajouter date_achat pour le split walk-forward
    trades = db.get_all_trades()
    closed = [t for t in trades if t["statut"] == "CLOTURE"]
    # Mapper trade_id -> date_achat
    date_map = {t["id"]: t["date_achat"][:10] for t in closed}
    features_df["date_achat"] = features_df["trade_id"].map(date_map)

    print(f"\nWalk-forward validation (split: {args.split_date})...")
    results = trainer.walk_forward_validate(features_df, split_date=args.split_date)

    # Feature importance
    importance_df = None
    if trainer.model is not None:
        importance_df = evaluator.feature_importance(
            trainer.model, trainer.feature_names
        )

    # Rapport complet
    evaluator.print_report(results, importance_df)

    # Sauvegarder le modele
    if args.importance:
        # Juste afficher l'importance
        return

    # Re-entrainer sur TOUTES les donnees pour le modele final
    print(f"\nEntrainement final sur toutes les donnees...")
    X, y = trainer.prepare_data(features_df)
    trainer.train(X, y)
    trainer.save_model(MODEL_PATH)
    print(f"Modele sauvegarde: {MODEL_PATH}")


if __name__ == "__main__":
    main()
```

**Step 2: Creer analyze_features.py**

```python
"""Script d'exploration des features pour comprendre les patterns de Nicolas.

Usage:
    uv run python scripts/analyze_features.py             # Stats descriptives
    uv run python scripts/analyze_features.py --trade 42   # Features d'un trade specifique
"""

import argparse

from src.core.database import Database
from src.analysis.feature_engine import FeatureEngine

DB_PATH = "data/trades.db"


def print_descriptive_stats(features_df):
    """Affiche les stats descriptives des features."""
    winners = features_df[features_df["is_winner"] == 1]
    losers = features_df[features_df["is_winner"] == 0]

    print("=" * 70)
    print("ANALYSE DES FEATURES — Profil de trading Nicolas")
    print("=" * 70)

    print(f"\nTrades: {len(features_df)} ({len(winners)} gagnants, {len(losers)} perdants)")

    # Comparer gagnants vs perdants sur les features cles
    key_features = [
        "range_position_10", "range_position_20",
        "rsi_14", "volume_ratio_20",
        "nb_catalysts", "best_catalyst_score",
    ]

    print(f"\n{'Feature':30s} {'Gagnants (moy)':>15s} {'Perdants (moy)':>15s} {'Diff':>10s}")
    print("-" * 70)

    for feat in key_features:
        if feat in features_df.columns:
            w_mean = winners[feat].mean() if len(winners) > 0 else 0
            l_mean = losers[feat].mean() if len(losers) > 0 else 0
            diff = w_mean - l_mean
            print(f"{feat:30s} {w_mean:>15.3f} {l_mean:>15.3f} {diff:>+10.3f}")

    # Distribution des types de catalyseurs
    print(f"\nTypes de catalyseurs (gagnants):")
    if "catalyst_type" in winners.columns:
        for cat, count in winners["catalyst_type"].value_counts().items():
            print(f"  {cat:25s}: {count:3d} ({100*count/len(winners):.1f}%)")

    print(f"\nTypes de catalyseurs (perdants):")
    if "catalyst_type" in losers.columns:
        for cat, count in losers["catalyst_type"].value_counts().items():
            print(f"  {cat:25s}: {count:3d} ({100*count/len(losers):.1f}%)")


def print_trade_features(features_df, trade_id: int):
    """Affiche les features d'un trade specifique."""
    row = features_df[features_df["trade_id"] == trade_id]
    if len(row) == 0:
        print(f"Trade #{trade_id} non trouve dans les features")
        return

    row = row.iloc[0]
    result = "GAGNANT" if row["is_winner"] == 1 else "PERDANT"

    print(f"\n=== Trade #{trade_id} — {result} ===")
    print(f"\nFeatures techniques (au moment de l'achat):")
    for feat in ["range_position_10", "range_position_20", "rsi_14",
                 "macd_histogram", "bollinger_position", "volume_ratio_20",
                 "atr_14_pct", "variation_1j", "variation_5j",
                 "distance_sma20", "distance_sma50"]:
        if feat in row.index:
            print(f"  {feat:25s}: {row[feat]:.4f}")

    print(f"\nFeatures catalyseur:")
    for feat in ["catalyst_type", "nb_catalysts", "best_catalyst_score",
                 "has_text_match", "sentiment_avg", "nb_news_sources"]:
        if feat in row.index:
            print(f"  {feat:25s}: {row[feat]}")

    print(f"\nFeatures contexte:")
    for feat in ["day_of_week", "nb_previous_trades",
                 "previous_win_rate", "days_since_last_trade"]:
        if feat in row.index:
            print(f"  {feat:25s}: {row[feat]}")


def main():
    parser = argparse.ArgumentParser(description="Exploration des features")
    parser.add_argument("--trade", type=int, help="ID du trade a inspecter")
    args = parser.parse_args()

    db = Database(DB_PATH)
    db.init_db()
    engine = FeatureEngine(db)

    print("Construction des features...")
    features_df = engine.build_all_features()

    if args.trade:
        print_trade_features(features_df, args.trade)
    else:
        print_descriptive_stats(features_df)


if __name__ == "__main__":
    main()
```

**Step 3: Run all tests (regression)**

Run: `uv run pytest tests/ -v`
Expected: 166 PASS (pas de changement)

**Step 4: Commit**

```bash
git add scripts/train_model.py scripts/analyze_features.py
git commit -m "feat(scripts): add train_model.py and analyze_features.py CLI"
```

---

### Task 8: Mise a jour CLAUDE.md + test d'integration + commit final

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Ajouter les commandes dans CLAUDE.md**

Dans la section `## Commandes`, ajouter:

```bash
uv run python scripts/train_model.py                # Entrainer le modele + evaluer
uv run python scripts/train_model.py --features      # Explorer les features
uv run python scripts/analyze_features.py            # Stats gagnants vs perdants
uv run python scripts/analyze_features.py --trade 42  # Features d'un trade specifique
```

**Step 2: Mettre a jour le statut de l'etape 4**

Changer la ligne etape 4 de:

```
| 4 | TODO | Feature engineering + entrainement ML |
```

en:

```
| 4 | DONE | ML pipeline — NewsClassifier, TechnicalIndicators, FeatureEngine, Trainer, Evaluator |
```

**Step 3: Mettre a jour Current Project State**

Mettre a jour Tests avec le nouveau total.

**Step 4: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: 166 PASS

**Step 5: Run le pipeline complet sur les vraies donnees**

Run: `uv run python scripts/analyze_features.py`
Expected: Affiche les stats des features pour les ~141 trades

Run: `uv run python scripts/train_model.py`
Expected: Entraine, affiche le rapport, sauvegarde le modele

**Step 6: Commit final**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for etape 4 completion"
```
