# Etape 4bis — LLM Trade Analysis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace shallow regex-based news classification with GPT-4o-mini analysis of each trade to deeply understand WHY Nicolas trades.

**Architecture:** New `LLMAnalyzer` module sends each trade + surrounding news + technical context to GPT-4o-mini, which returns structured JSON analysis. Results stored in `trade_analyses_llm` table. `FeatureEngine` updated to read LLM analyses instead of regex classifications. XGBoost retrained on richer features.

**Tech Stack:** openai Python SDK, GPT-4o-mini, SQLite, existing XGBoost pipeline.

---

### Task 1: Add openai dependency

**Files:**
- Modify: `pyproject.toml:7-19`
- Modify: `.env.example:1-6`

**Step 1: Add openai to pyproject.toml**

In `pyproject.toml`, add `"openai>=1.0.0"` to the dependencies list after `"numpy>=2.0.0"`:

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
    "openai>=1.0.0",
]
```

**Step 2: Update .env.example**

Add at the end:

```
# Etape 4 - Analyse LLM des news
OPENAI_API_KEY=your_key_here
```

**Step 3: Install the dependency**

Run: `uv sync`
Expected: openai installed successfully.

**Step 4: Commit**

```bash
git add pyproject.toml .env.example
git commit -m "feat: add openai dependency for LLM trade analysis"
```

---

### Task 2: Add trade_analyses_llm table to database

**Files:**
- Modify: `src/core/database.py` (add table creation + CRUD methods)
- Test: `tests/test_database.py`

**Step 1: Write the failing tests**

Add to `tests/test_database.py` a new test class:

```python
class TestTradeAnalysesLLM:
    """Tests CRUD pour la table trade_analyses_llm."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        # Insert a trade to reference
        self.db.insert_trade_complet({
            "isin": "FR0000120578", "nom_action": "SANOFI",
            "date_achat": "2025-07-10", "date_vente": "2025-07-20",
            "prix_achat": 95.0, "prix_vente": 100.0, "quantite": 10,
            "rendement_brut_pct": 5.26, "rendement_net_pct": 5.0,
            "duree_jours": 10, "frais_totaux": 2.5, "statut": "CLOTURE",
        })

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_insert_and_get_analysis(self):
        """Insert et recupere une analyse LLM."""
        analysis = {
            "trade_id": 1,
            "primary_news_id": None,
            "catalyst_type": "EARNINGS",
            "catalyst_summary": "Nicolas a achete car resultats T2 solides",
            "catalyst_confidence": 0.85,
            "news_sentiment": 0.6,
            "buy_reason": "Resultats T2 au-dessus des attentes",
            "sell_reason": "Objectif de +5% atteint",
            "trade_quality": "BON",
            "model_used": "gpt-4o-mini",
            "analyzed_at": "2026-02-24 19:00:00",
        }
        self.db.insert_trade_analysis(analysis)
        result = self.db.get_trade_analysis(1)
        assert result is not None
        assert result["catalyst_type"] == "EARNINGS"
        assert result["catalyst_confidence"] == 0.85
        assert result["buy_reason"] == "Resultats T2 au-dessus des attentes"

    def test_get_analysis_missing(self):
        """Retourne None si le trade n'a pas d'analyse."""
        result = self.db.get_trade_analysis(999)
        assert result is None

    def test_upsert_analysis(self):
        """Insert ou replace une analyse existante."""
        analysis1 = {
            "trade_id": 1, "primary_news_id": None,
            "catalyst_type": "UNKNOWN", "catalyst_summary": "test",
            "catalyst_confidence": 0.5, "news_sentiment": 0.0,
            "buy_reason": "v1", "sell_reason": "v1",
            "trade_quality": "MOYEN", "model_used": "gpt-4o-mini",
            "analyzed_at": "2026-02-24 19:00:00",
        }
        self.db.insert_trade_analysis(analysis1)
        # Update with new data
        analysis2 = {**analysis1, "catalyst_type": "EARNINGS",
                      "catalyst_confidence": 0.9, "buy_reason": "v2"}
        self.db.insert_trade_analysis(analysis2)
        result = self.db.get_trade_analysis(1)
        assert result["catalyst_type"] == "EARNINGS"
        assert result["buy_reason"] == "v2"

    def test_count_analyses(self):
        """Compte les analyses."""
        assert self.db.count_trade_analyses() == 0
        self.db.insert_trade_analysis({
            "trade_id": 1, "primary_news_id": None,
            "catalyst_type": "EARNINGS", "catalyst_summary": "test",
            "catalyst_confidence": 0.5, "news_sentiment": 0.0,
            "buy_reason": "x", "sell_reason": "x",
            "trade_quality": "BON", "model_used": "gpt-4o-mini",
            "analyzed_at": "2026-02-24 19:00:00",
        })
        assert self.db.count_trade_analyses() == 1

    def test_get_all_analyses(self):
        """Recupere toutes les analyses."""
        self.db.insert_trade_analysis({
            "trade_id": 1, "primary_news_id": None,
            "catalyst_type": "EARNINGS", "catalyst_summary": "test",
            "catalyst_confidence": 0.85, "news_sentiment": 0.6,
            "buy_reason": "x", "sell_reason": "x",
            "trade_quality": "BON", "model_used": "gpt-4o-mini",
            "analyzed_at": "2026-02-24 19:00:00",
        })
        results = self.db.get_all_trade_analyses()
        assert len(results) == 1
        assert results[0]["trade_id"] == 1
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_database.py::TestTradeAnalysesLLM -v`
Expected: FAIL — `AttributeError: 'Database' object has no attribute 'insert_trade_analysis'`

**Step 3: Add table creation to database.py init_db()**

In `src/core/database.py`, add after the `trade_catalyseurs` table creation block (after line 113):

```python
        # Table trade_analyses_llm (etape 4 bis — analyse LLM)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trade_analyses_llm (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER NOT NULL UNIQUE,
                primary_news_id INTEGER,
                catalyst_type TEXT NOT NULL,
                catalyst_summary TEXT NOT NULL,
                catalyst_confidence REAL NOT NULL,
                news_sentiment REAL,
                buy_reason TEXT,
                sell_reason TEXT,
                trade_quality TEXT,
                model_used TEXT DEFAULT 'gpt-4o-mini',
                analyzed_at TEXT NOT NULL,
                FOREIGN KEY (trade_id) REFERENCES trades_complets(id)
            );
            CREATE INDEX IF NOT EXISTS idx_analyses_trade
                ON trade_analyses_llm(trade_id);
        """)
```

**Step 4: Add CRUD methods to Database class**

Add after the `clear_catalyseurs` method (after line 397):

```python
    # --- Trade Analyses LLM ---

    def insert_trade_analysis(self, analysis: dict):
        """Insere ou remplace une analyse LLM pour un trade."""
        conn = self._connect()
        conn.execute("""
            INSERT OR REPLACE INTO trade_analyses_llm
                (trade_id, primary_news_id, catalyst_type, catalyst_summary,
                 catalyst_confidence, news_sentiment, buy_reason, sell_reason,
                 trade_quality, model_used, analyzed_at)
            VALUES
                (:trade_id, :primary_news_id, :catalyst_type, :catalyst_summary,
                 :catalyst_confidence, :news_sentiment, :buy_reason, :sell_reason,
                 :trade_quality, :model_used, :analyzed_at)
        """, analysis)
        conn.commit()
        conn.close()

    def get_trade_analysis(self, trade_id: int) -> dict | None:
        """Recupere l'analyse LLM pour un trade. None si absente."""
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM trade_analyses_llm WHERE trade_id = ?",
            (trade_id,),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_all_trade_analyses(self) -> list[dict]:
        """Recupere toutes les analyses LLM."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM trade_analyses_llm ORDER BY trade_id"
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def count_trade_analyses(self) -> int:
        """Compte le nombre d'analyses LLM."""
        conn = self._connect()
        count = conn.execute(
            "SELECT COUNT(*) FROM trade_analyses_llm"
        ).fetchone()[0]
        conn.close()
        return count
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_database.py::TestTradeAnalysesLLM -v`
Expected: 5 tests PASS.

Run: `uv run pytest tests/test_database.py -v`
Expected: All existing tests still pass (18 + 5 = 23).

**Step 6: Commit**

```bash
git add src/core/database.py tests/test_database.py
git commit -m "feat(database): add trade_analyses_llm table + CRUD"
```

---

### Task 3: Create LLMAnalyzer module

**Files:**
- Create: `src/analysis/llm_analyzer.py`
- Test: `tests/test_llm_analyzer.py`

**Step 1: Write the failing tests**

Create `tests/test_llm_analyzer.py`:

```python
"""Tests pour l'analyseur LLM des trades."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.database import Database
from src.analysis.llm_analyzer import LLMAnalyzer, LLM_ANALYSIS_PROMPT


def _seed_test_db(db: Database):
    """Peuple une base de test."""
    db.insert_trades_batch([
        {
            "isin": "FR0000120578", "nom_action": "SANOFI",
            "date_achat": "2025-07-10", "date_vente": "2025-07-20",
            "prix_achat": 95.0, "prix_vente": 100.0, "quantite": 10,
            "rendement_brut_pct": 5.26, "rendement_net_pct": 5.0,
            "duree_jours": 10, "frais_totaux": 2.5, "statut": "CLOTURE",
        },
    ])
    # Prix
    np.random.seed(42)
    dates = pd.bdate_range("2025-05-01", "2025-09-15")
    close = 97 + 5 * np.sin(np.linspace(0, 6 * np.pi, len(dates)))
    for i, d in enumerate(dates):
        db.insert_price({
            "ticker": "SAN.PA", "date": d.strftime("%Y-%m-%d"),
            "open": round(close[i] - 0.5, 2), "high": round(close[i] + 1, 2),
            "low": round(close[i] - 1, 2), "close": round(close[i], 2),
            "volume": 100000,
        })
    # News
    db.insert_news_batch([
        {
            "ticker": "SAN.PA", "title": "Sanofi: resultats T2 au-dessus des attentes",
            "source": "Reuters", "url": "https://ex.com/san1",
            "published_at": "2025-07-09", "description": "CA en hausse de 8%",
            "sentiment": 0.6, "source_api": "alpha_vantage",
        },
        {
            "ticker": "SAN.PA", "title": "Bourse: le CAC 40 en hausse",
            "source": "BFM", "url": "https://ex.com/san2",
            "published_at": "2025-07-10", "description": "Marche positif",
            "sentiment": 0.3, "source_api": "gnews",
        },
    ])


class TestBuildPrompt:
    """Tests de construction du prompt LLM."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.temp_dir.name, "test.db"))
        self.db.init_db()
        _seed_test_db(self.db)
        self.analyzer = LLMAnalyzer(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_build_prompt_contains_trade_info(self):
        """Le prompt contient les infos du trade."""
        trades = self.db.get_all_trades()
        prompt = self.analyzer.build_prompt(trades[0])
        assert "SANOFI" in prompt
        assert "2025-07-10" in prompt
        assert "95.0" in prompt or "95.00" in prompt

    def test_build_prompt_contains_news(self):
        """Le prompt contient les news de la fenetre."""
        trades = self.db.get_all_trades()
        prompt = self.analyzer.build_prompt(trades[0])
        assert "resultats T2" in prompt

    def test_build_prompt_contains_technical_context(self):
        """Le prompt contient les indicateurs techniques."""
        trades = self.db.get_all_trades()
        prompt = self.analyzer.build_prompt(trades[0])
        assert "RSI" in prompt or "rsi" in prompt


class TestParseResponse:
    """Tests de parsing de la reponse LLM."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.temp_dir.name, "test.db"))
        self.db.init_db()
        self.analyzer = LLMAnalyzer(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_parse_valid_response(self):
        """Parse une reponse JSON valide du LLM."""
        response = json.dumps({
            "primary_news_index": 1,
            "catalyst_type": "EARNINGS",
            "catalyst_confidence": 0.85,
            "catalyst_summary": "Nicolas a achete car resultats T2 solides",
            "news_sentiment": 0.7,
            "buy_reason": "Resultats T2 au-dessus des attentes",
            "sell_reason": "Objectif atteint",
            "trade_quality": "BON",
        })
        news_list = [{"id": 42}, {"id": 43}]
        result = self.analyzer.parse_response(response, trade_id=1,
                                               news_list=news_list)
        assert result["trade_id"] == 1
        assert result["catalyst_type"] == "EARNINGS"
        assert result["primary_news_id"] == 42  # index 1 -> news_list[0]
        assert result["catalyst_confidence"] == 0.85

    def test_parse_no_catalyst_found(self):
        """Parse quand le LLM ne trouve pas de catalyseur."""
        response = json.dumps({
            "primary_news_index": 0,
            "catalyst_type": "TECHNICAL",
            "catalyst_confidence": 0.6,
            "catalyst_summary": "Pas de news declencheuse identifiee",
            "news_sentiment": 0.0,
            "buy_reason": "Signal technique pur",
            "sell_reason": "Stop loss",
            "trade_quality": "MOYEN",
        })
        result = self.analyzer.parse_response(response, trade_id=5,
                                               news_list=[])
        assert result["trade_id"] == 5
        assert result["catalyst_type"] == "TECHNICAL"
        assert result["primary_news_id"] is None

    def test_parse_invalid_json_returns_fallback(self):
        """Retourne un fallback si le JSON est invalide."""
        result = self.analyzer.parse_response("not json", trade_id=1,
                                               news_list=[])
        assert result["trade_id"] == 1
        assert result["catalyst_type"] == "UNKNOWN"
        assert result["catalyst_confidence"] == 0.0

    def test_parse_clamps_confidence(self):
        """La confiance est clampee entre 0 et 1."""
        response = json.dumps({
            "primary_news_index": 0,
            "catalyst_type": "EARNINGS",
            "catalyst_confidence": 1.5,
            "catalyst_summary": "test",
            "news_sentiment": 0.0,
            "buy_reason": "x", "sell_reason": "x",
            "trade_quality": "BON",
        })
        result = self.analyzer.parse_response(response, trade_id=1,
                                               news_list=[])
        assert result["catalyst_confidence"] == 1.0

    def test_parse_validates_catalyst_type(self):
        """Un catalyst_type invalide est remplace par UNKNOWN."""
        response = json.dumps({
            "primary_news_index": 0,
            "catalyst_type": "INVALID_TYPE",
            "catalyst_confidence": 0.5,
            "catalyst_summary": "test",
            "news_sentiment": 0.0,
            "buy_reason": "x", "sell_reason": "x",
            "trade_quality": "BON",
        })
        result = self.analyzer.parse_response(response, trade_id=1,
                                               news_list=[])
        assert result["catalyst_type"] == "UNKNOWN"


class TestAnalyzeTrade:
    """Tests d'analyse d'un trade (avec mock OpenAI)."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.temp_dir.name, "test.db"))
        self.db.init_db()
        _seed_test_db(self.db)
        self.analyzer = LLMAnalyzer(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    @patch("src.analysis.llm_analyzer.OpenAI")
    def test_analyze_trade_calls_openai(self, mock_openai_cls):
        """analyze_trade appelle l'API OpenAI et sauvegarde le resultat."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps({
                "primary_news_index": 1,
                "catalyst_type": "EARNINGS",
                "catalyst_confidence": 0.9,
                "catalyst_summary": "Resultats T2 solides",
                "news_sentiment": 0.7,
                "buy_reason": "CA en hausse",
                "sell_reason": "Objectif atteint",
                "trade_quality": "BON",
            })))]
        )

        trades = self.db.get_all_trades()
        self.analyzer.analyze_trade(trades[0])

        # Verifie que l'API a ete appelee
        mock_client.chat.completions.create.assert_called_once()

        # Verifie que le resultat est en base
        result = self.db.get_trade_analysis(trades[0]["id"])
        assert result is not None
        assert result["catalyst_type"] == "EARNINGS"

    def test_analyze_trade_skip_if_exists(self):
        """Skip un trade deja analyse (reprise incrementale)."""
        # Pre-insert an analysis
        self.db.insert_trade_analysis({
            "trade_id": 1, "primary_news_id": None,
            "catalyst_type": "EARNINGS", "catalyst_summary": "deja fait",
            "catalyst_confidence": 0.9, "news_sentiment": 0.7,
            "buy_reason": "x", "sell_reason": "x",
            "trade_quality": "BON", "model_used": "gpt-4o-mini",
            "analyzed_at": "2026-02-24 19:00:00",
        })
        trades = self.db.get_all_trades()
        # Should return False (skipped) without calling API
        result = self.analyzer.analyze_trade(trades[0])
        assert result is False
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm_analyzer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.analysis.llm_analyzer'`

**Step 3: Create the llm_analyzer.py module**

Create `src/analysis/llm_analyzer.py`:

```python
"""Analyse LLM des trades de Nicolas via GPT-4o-mini.

Pour chaque trade, envoie le contexte complet (news, indicateurs techniques,
infos du trade) a GPT-4o-mini et recupere une analyse structuree du catalyseur,
de la raison d'achat/vente, et de la qualite du trade.
"""

import json
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

from src.core.database import Database
from src.data_collection.ticker_mapper import TickerMapper, TickerNotFoundError
from src.analysis.technical_indicators import TechnicalIndicators

load_dotenv()

VALID_CATALYST_TYPES = [
    "EARNINGS", "FDA_REGULATORY", "UPGRADE", "DOWNGRADE", "CONTRACT",
    "DIVIDEND", "RESTRUCTURING", "INSIDER", "SECTOR_MACRO",
    "OTHER_POSITIVE", "OTHER_NEGATIVE", "TECHNICAL", "UNKNOWN",
]

VALID_TRADE_QUALITIES = ["EXCELLENT", "BON", "MOYEN", "MAUVAIS"]

LLM_ANALYSIS_PROMPT = """Tu es un analyste financier expert specialise dans le trading swing court terme sur le marche francais (PEA).

Tu analyses les trades d'un trader nomme Nicolas pour comprendre sa logique de decision.
Nicolas a un style swing court terme (quelques jours), objectif 4-5% par trade, 89% win rate.
Il n'achete JAMAIS au hasard — chaque achat est motive par un catalyseur (news, annonce, signal technique).

Voici un trade a analyser:

## Infos du trade
- Action: {nom_action}
- Achat: {date_achat} a {prix_achat:.2f} EUR
- Vente: {date_vente} a {prix_vente:.2f} EUR
- Rendement: {rendement:+.2f}%
- Duree: {duree_jours} jours
- Resultat: {resultat}

## Indicateurs techniques au moment de l'achat
{technical_context}

## News autour de la date d'achat (J-5 a J+1)
{news_context}

## Instructions

Analyse ce trade et reponds en JSON strict avec ces champs:
- "primary_news_index": numero de la news qui a le plus probablement declenche l'achat (1 = premiere news listee). 0 si aucune news n'est pertinente.
- "catalyst_type": un parmi {valid_types}
- "catalyst_confidence": float 0.0-1.0, ta confiance dans l'identification du catalyseur
- "catalyst_summary": une phrase commencant par "Nicolas a achete parce que..."
- "news_sentiment": float -1.0 a +1.0, sentiment de la news declencheuse (0 si aucune)
- "buy_reason": 2-3 phrases expliquant pourquoi Nicolas a achete a ce moment precis
- "sell_reason": 1-2 phrases expliquant pourquoi il a vendu (objectif atteint, stop loss, etc.)
- "trade_quality": un parmi ["EXCELLENT", "BON", "MOYEN", "MAUVAIS"] selon le rendement et la logique

Reponds UNIQUEMENT avec le JSON, sans texte autour."""


class LLMAnalyzer:
    """Analyse les trades de Nicolas via GPT-4o-mini."""

    def __init__(self, db: Database, model: str = "gpt-4o-mini"):
        self.db = db
        self.model = model
        self.mapper = TickerMapper()
        self.tech = TechnicalIndicators()
        self._price_cache: dict[str, object] = {}

    def _get_openai_client(self) -> OpenAI:
        """Cree un client OpenAI avec la cle du .env."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY manquante dans .env")
        return OpenAI(api_key=api_key)

    def _get_enriched_prices(self, ticker: str):
        """Recupere les prix enrichis (avec cache)."""
        if ticker in self._price_cache:
            return self._price_cache[ticker]
        import pandas as pd
        prices = self.db.get_prices(ticker)
        if len(prices) < 20:
            return None
        df = pd.DataFrame(prices)
        enriched = self.tech.compute_all(df)
        self._price_cache[ticker] = enriched
        return enriched

    def _get_technical_context(self, trade: dict) -> str:
        """Construit le contexte technique pour le prompt."""
        nom_action = trade["nom_action"]
        date_achat = trade["date_achat"][:10]
        try:
            ticker = self.mapper.get_ticker(nom_action)
        except TickerNotFoundError:
            return "Indicateurs techniques non disponibles (ticker inconnu)"

        enriched = self._get_enriched_prices(ticker)
        if enriched is None:
            return "Indicateurs techniques non disponibles (pas assez de prix)"

        indicators = self.tech.get_indicators_at_date(enriched, date_achat)
        if indicators is None:
            return "Indicateurs techniques non disponibles (date non trouvee)"

        lines = []
        labels = {
            "rsi_14": "RSI(14)",
            "macd_histogram": "MACD Histogram",
            "bollinger_position": "Position Bollinger (0=bas, 1=haut)",
            "range_position_10": "Position dans range 10j (0=bas, 1=haut)",
            "range_position_20": "Position dans range 20j (0=bas, 1=haut)",
            "volume_ratio_20": "Volume / Moyenne 20j",
            "atr_14_pct": "ATR(14) en %",
            "variation_1j": "Variation J-1 (%)",
            "variation_5j": "Variation 5 jours (%)",
            "distance_sma20": "Distance SMA20 (%)",
            "distance_sma50": "Distance SMA50 (%)",
        }
        for key, label in labels.items():
            val = indicators.get(key)
            if val is not None:
                lines.append(f"- {label}: {val:.2f}")
        return "\n".join(lines) if lines else "Aucun indicateur disponible"

    def _get_news_context(self, trade: dict) -> tuple[str, list[dict]]:
        """Construit le contexte news et retourne (texte, liste_news)."""
        nom_action = trade["nom_action"]
        date_achat = trade["date_achat"][:10]
        try:
            ticker = self.mapper.get_ticker(nom_action)
        except TickerNotFoundError:
            return "Aucune news disponible (ticker inconnu)", []

        date_start = (datetime.strptime(date_achat, "%Y-%m-%d")
                      - timedelta(days=5)).strftime("%Y-%m-%d")
        date_end = (datetime.strptime(date_achat, "%Y-%m-%d")
                    + timedelta(days=1)).strftime("%Y-%m-%d")

        news_list = self.db.get_news_in_window(ticker, date_start, date_end)

        if not news_list:
            return "Aucune news trouvee dans la fenetre J-5 a J+1", []

        lines = []
        for i, news in enumerate(news_list, 1):
            news_date = news["published_at"][:10]
            distance = (datetime.strptime(news_date, "%Y-%m-%d")
                        - datetime.strptime(date_achat, "%Y-%m-%d")).days
            sentiment_str = ""
            if news.get("sentiment") is not None:
                sentiment_str = f" [sentiment: {news['sentiment']:.1f}]"
            desc = news.get("description") or ""
            desc_short = desc[:150] + "..." if len(desc) > 150 else desc
            lines.append(
                f"{i}. [J{distance:+d}] \"{news['title']}\""
                f" (source: {news.get('source', 'inconnue')})"
                f"{sentiment_str}"
                f"\n   {desc_short}"
            )

        return "\n".join(lines), news_list

    def build_prompt(self, trade: dict) -> str:
        """Construit le prompt complet pour analyser un trade."""
        technical_context = self._get_technical_context(trade)
        news_context, _ = self._get_news_context(trade)

        date_vente = trade["date_vente"][:10] if trade["date_vente"] else "OUVERT"
        prix_vente = trade["prix_vente"] or 0
        rendement = trade["rendement_brut_pct"] or 0
        duree = trade["duree_jours"] or 0
        resultat = "GAGNANT" if rendement > 0 else "PERDANT"

        return LLM_ANALYSIS_PROMPT.format(
            nom_action=trade["nom_action"],
            date_achat=trade["date_achat"][:10],
            prix_achat=trade["prix_achat"],
            date_vente=date_vente,
            prix_vente=prix_vente,
            rendement=rendement,
            duree_jours=duree,
            resultat=resultat,
            technical_context=technical_context,
            news_context=news_context,
            valid_types=", ".join(VALID_CATALYST_TYPES),
        )

    def parse_response(self, response_text: str, trade_id: int,
                        news_list: list[dict]) -> dict:
        """Parse la reponse JSON du LLM en dict pret pour la base."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fallback = {
            "trade_id": trade_id,
            "primary_news_id": None,
            "catalyst_type": "UNKNOWN",
            "catalyst_summary": "Analyse LLM echouee",
            "catalyst_confidence": 0.0,
            "news_sentiment": 0.0,
            "buy_reason": "",
            "sell_reason": "",
            "trade_quality": "MOYEN",
            "model_used": self.model,
            "analyzed_at": now,
        }

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            logger.error(f"Trade #{trade_id}: reponse LLM non-JSON")
            return fallback

        # Resoudre primary_news_id
        news_index = data.get("primary_news_index", 0)
        primary_news_id = None
        if news_index > 0 and news_index <= len(news_list):
            primary_news_id = news_list[news_index - 1].get("id")

        # Valider catalyst_type
        catalyst_type = data.get("catalyst_type", "UNKNOWN")
        if catalyst_type not in VALID_CATALYST_TYPES:
            catalyst_type = "UNKNOWN"

        # Valider trade_quality
        trade_quality = data.get("trade_quality", "MOYEN")
        if trade_quality not in VALID_TRADE_QUALITIES:
            trade_quality = "MOYEN"

        # Clamper confidence
        confidence = float(data.get("catalyst_confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        # Clamper sentiment
        sentiment = float(data.get("news_sentiment", 0.0))
        sentiment = max(-1.0, min(1.0, sentiment))

        return {
            "trade_id": trade_id,
            "primary_news_id": primary_news_id,
            "catalyst_type": catalyst_type,
            "catalyst_summary": data.get("catalyst_summary", "")[:500],
            "catalyst_confidence": confidence,
            "news_sentiment": sentiment,
            "buy_reason": data.get("buy_reason", "")[:1000],
            "sell_reason": data.get("sell_reason", "")[:500],
            "trade_quality": trade_quality,
            "model_used": self.model,
            "analyzed_at": now,
        }

    def analyze_trade(self, trade: dict) -> bool:
        """Analyse un trade via GPT-4o-mini. Retourne False si skip (deja fait)."""
        trade_id = trade["id"]

        # Reprise incrementale
        existing = self.db.get_trade_analysis(trade_id)
        if existing:
            logger.debug(f"Trade #{trade_id} deja analyse, skip")
            return False

        _, news_list = self._get_news_context(trade)
        prompt = self.build_prompt(trade)

        client = self._get_openai_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )

        response_text = response.choices[0].message.content
        analysis = self.parse_response(response_text, trade_id, news_list)
        self.db.insert_trade_analysis(analysis)

        logger.info(f"Trade #{trade_id} ({trade['nom_action']}): "
                     f"{analysis['catalyst_type']} "
                     f"(conf={analysis['catalyst_confidence']:.2f})")
        return True

    def analyze_all_trades(self) -> dict:
        """Analyse tous les trades clotures. Reprise incrementale.

        Returns:
            Resume: {total, analyzed, skipped, errors}.
        """
        trades = self.db.get_all_trades()
        closed = [t for t in trades if t["statut"] == "CLOTURE"]

        analyzed = 0
        skipped = 0
        errors = 0

        logger.info(f"Analyse LLM de {len(closed)} trades clotures...")

        for trade in closed:
            try:
                result = self.analyze_trade(trade)
                if result:
                    analyzed += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.error(f"Erreur trade #{trade['id']} "
                             f"({trade['nom_action']}): {e}")
                errors += 1

        summary = {
            "total": len(closed),
            "analyzed": analyzed,
            "skipped": skipped,
            "errors": errors,
        }
        logger.info(f"Analyse terminee: {summary}")
        return summary
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. uv run pytest tests/test_llm_analyzer.py -v`
Expected: All 10 tests PASS.

**Step 5: Run all tests to verify no regression**

Run: `uv run pytest tests/ -v`
Expected: 177+ tests PASS (167 existing + 10 new).

**Step 6: Commit**

```bash
git add src/analysis/llm_analyzer.py tests/test_llm_analyzer.py
git commit -m "feat(analysis): add LLMAnalyzer for GPT-4o-mini trade analysis"
```

---

### Task 4: Create CLI script for LLM analysis

**Files:**
- Create: `scripts/analyze_trades_llm.py`

**Step 1: Create the script**

```python
"""Analyse LLM des trades de Nicolas via GPT-4o-mini.

Usage:
    uv run python scripts/analyze_trades_llm.py           # Analyser tous les trades
    uv run python scripts/analyze_trades_llm.py --trade 42  # Analyser un trade specifique
    uv run python scripts/analyze_trades_llm.py --stats     # Stats des analyses existantes
"""

import argparse

from src.core.database import Database
from src.analysis.llm_analyzer import LLMAnalyzer

DB_PATH = "data/trades.db"


def print_stats(db: Database):
    """Affiche les stats des analyses LLM existantes."""
    analyses = db.get_all_trade_analyses()
    total_trades = db.count_trades()

    print("=" * 60)
    print("ANALYSES LLM — Statistiques")
    print("=" * 60)
    print(f"\nTrades analyses: {len(analyses)}/{total_trades}")

    if not analyses:
        print("Aucune analyse. Lancez sans --stats pour analyser.")
        return

    # Distribution des types
    types = {}
    qualities = {}
    for a in analyses:
        t = a["catalyst_type"]
        types[t] = types.get(t, 0) + 1
        q = a["trade_quality"]
        qualities[q] = qualities.get(q, 0) + 1

    print(f"\nTypes de catalyseurs:")
    for t, count in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  {t:20s}: {count:3d} ({100*count/len(analyses):.1f}%)")

    print(f"\nQualite des trades:")
    for q in ["EXCELLENT", "BON", "MOYEN", "MAUVAIS"]:
        count = qualities.get(q, 0)
        print(f"  {q:20s}: {count:3d} ({100*count/len(analyses):.1f}%)")

    # Confiance moyenne
    avg_conf = sum(a["catalyst_confidence"] for a in analyses) / len(analyses)
    print(f"\nConfiance moyenne: {avg_conf:.2f}")


def print_trade_analysis(db: Database, trade_id: int):
    """Affiche l'analyse LLM d'un trade specifique."""
    analysis = db.get_trade_analysis(trade_id)
    if not analysis:
        print(f"Pas d'analyse pour le trade #{trade_id}")
        return

    print(f"\n=== Trade #{trade_id} — Analyse LLM ===")
    print(f"Type catalyseur: {analysis['catalyst_type']}")
    print(f"Confiance:       {analysis['catalyst_confidence']:.2f}")
    print(f"Sentiment:       {analysis['news_sentiment']:.2f}")
    print(f"Qualite:         {analysis['trade_quality']}")
    print(f"\nResume: {analysis['catalyst_summary']}")
    print(f"\nRaison achat: {analysis['buy_reason']}")
    print(f"Raison vente: {analysis['sell_reason']}")


def main():
    parser = argparse.ArgumentParser(description="Analyse LLM des trades")
    parser.add_argument("--trade", type=int, help="Analyser un trade specifique")
    parser.add_argument("--stats", action="store_true", help="Stats seulement")
    args = parser.parse_args()

    db = Database(DB_PATH)
    db.init_db()

    if args.stats:
        print_stats(db)
        return

    if args.trade:
        # Analyser un seul trade
        analyzer = LLMAnalyzer(db)
        trades = db.get_all_trades()
        trade = next((t for t in trades if t["id"] == args.trade), None)
        if not trade:
            print(f"Trade #{args.trade} non trouve")
            return
        print(f"Analyse du trade #{args.trade}...")
        analyzer.analyze_trade(trade)
        print_trade_analysis(db, args.trade)
        return

    # Analyser tous les trades
    analyzer = LLMAnalyzer(db)
    print("Analyse LLM de tous les trades...")
    print("(reprise incrementale: les trades deja analyses sont ignores)\n")
    summary = analyzer.analyze_all_trades()
    print(f"\nResume: {summary['analyzed']} analyses, "
          f"{summary['skipped']} deja faits, {summary['errors']} erreurs")
    print_stats(db)


if __name__ == "__main__":
    main()
```

**Step 2: Test it manually (dry run on 1 trade)**

Run: `PYTHONPATH=. uv run python scripts/analyze_trades_llm.py --trade 3`
Expected: Trade #3 (EXAIL TECHNOLOGIES) analyzed, output shows catalyst type, summary, etc.

**Step 3: Commit**

```bash
git add scripts/analyze_trades_llm.py
git commit -m "feat(scripts): add CLI for LLM trade analysis"
```

---

### Task 5: Update FeatureEngine to use LLM analyses

**Files:**
- Modify: `src/analysis/feature_engine.py`
- Modify: `tests/test_feature_engine.py`

**Step 1: Update the test for new catalyst features**

In `tests/test_feature_engine.py`, update `_seed_test_db` to also insert LLM analyses, and update the catalyst feature tests.

Add after the catalyseurs insertion in `_seed_test_db`:

```python
    # Analyse LLM pour trade 1
    db.insert_trade_analysis({
        "trade_id": 1, "primary_news_id": 1,
        "catalyst_type": "EARNINGS",
        "catalyst_summary": "Nicolas a achete car resultats T2 solides",
        "catalyst_confidence": 0.85, "news_sentiment": 0.6,
        "buy_reason": "CA en hausse de 8% au T2",
        "sell_reason": "Objectif de +5% atteint",
        "trade_quality": "BON", "model_used": "gpt-4o-mini",
        "analyzed_at": "2026-02-24 19:00:00",
    })
    # Pas d'analyse LLM pour trade 2
```

Update `test_build_has_catalyst_features`:

```python
    def test_build_has_catalyst_features(self):
        """Le dict contient les features catalyseur LLM."""
        trades = self.db.get_all_trades()
        result = self.engine.build_trade_features(trades[0])
        assert "catalyst_type" in result
        assert "catalyst_confidence" in result
        assert "news_sentiment" in result
        assert "trade_quality_score" in result
        assert "has_clear_catalyst" in result
```

Add a new test:

```python
    def test_trade_with_llm_analysis_uses_it(self):
        """Un trade avec analyse LLM utilise les features LLM."""
        trades = self.db.get_all_trades()
        result = self.engine.build_trade_features(trades[0])
        assert result["catalyst_type"] == "EARNINGS"
        assert result["catalyst_confidence"] == 0.85
        assert result["has_clear_catalyst"] == 1
        assert result["trade_quality_score"] == 3  # BON = 3

    def test_trade_without_llm_falls_back(self):
        """Un trade sans analyse LLM utilise le fallback TECHNICAL."""
        trades = self.db.get_all_trades()
        result = self.engine.build_trade_features(trades[1])
        assert result["catalyst_type"] == "TECHNICAL"
        assert result["catalyst_confidence"] == 0.0
        assert result["has_clear_catalyst"] == 0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_feature_engine.py -v`
Expected: New/updated tests FAIL.

**Step 3: Update feature_engine.py**

Replace the `CATALYST_FEATURES` constant:

```python
CATALYST_FEATURES = [
    "catalyst_type", "catalyst_confidence", "news_sentiment",
    "trade_quality_score", "has_clear_catalyst", "buy_reason_length",
]
```

Replace the `_build_catalyst_features` method entirely:

```python
    def _build_catalyst_features(self, trade: dict) -> dict:
        """Construit les features catalyseur pour un trade.

        Utilise l'analyse LLM si disponible, sinon fallback TECHNICAL.
        """
        trade_id = trade["id"]
        analysis = self.db.get_trade_analysis(trade_id)

        if analysis:
            quality_map = {"EXCELLENT": 4, "BON": 3, "MOYEN": 2, "MAUVAIS": 1}
            return {
                "catalyst_type": analysis["catalyst_type"],
                "catalyst_confidence": analysis["catalyst_confidence"],
                "news_sentiment": analysis["news_sentiment"] or 0.0,
                "trade_quality_score": quality_map.get(
                    analysis["trade_quality"], 2),
                "has_clear_catalyst": 1 if analysis["primary_news_id"] else 0,
                "buy_reason_length": len(analysis["buy_reason"] or ""),
            }

        # Fallback: pas d'analyse LLM
        return {
            "catalyst_type": "TECHNICAL",
            "catalyst_confidence": 0.0,
            "news_sentiment": 0.0,
            "trade_quality_score": 2,
            "has_clear_catalyst": 0,
            "buy_reason_length": 0,
        }
```

Remove the imports `NewsClassifier` from feature_engine.py (no longer needed):
- Remove: `from src.analysis.news_classifier import NewsClassifier`
- Remove: `self.classifier = NewsClassifier()` from `__init__`

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_feature_engine.py -v`
Expected: All tests PASS (some old tests may need minor adjustments).

Run: `uv run pytest tests/ -v`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add src/analysis/feature_engine.py tests/test_feature_engine.py
git commit -m "feat(features): use LLM analyses instead of regex classification"
```

---

### Task 6: Run LLM analysis on all 141 trades

**Step 1: Run the analysis**

Run: `PYTHONPATH=. uv run python scripts/analyze_trades_llm.py`
Expected: ~141 trades analyzed over 3-5 minutes. Output shows progress.

**Step 2: Check stats**

Run: `PYTHONPATH=. uv run python scripts/analyze_trades_llm.py --stats`
Expected: Stats showing catalyst type distribution, quality distribution, avg confidence.

**Step 3: Spot check a few trades**

Run: `PYTHONPATH=. uv run python scripts/analyze_trades_llm.py --trade 3`
Run: `PYTHONPATH=. uv run python scripts/analyze_trades_llm.py --trade 16`
Expected: Meaningful analysis for each trade.

---

### Task 7: Retrain model with LLM features

**Step 1: Analyze features**

Run: `PYTHONPATH=. uv run python scripts/analyze_features.py`
Expected: Stats now show LLM-based catalyst types (fewer UNKNOWN).

**Step 2: Retrain the model**

Run: `PYTHONPATH=. uv run python scripts/train_model.py`
Expected: Model retrained. Compare metrics with previous run:
- Previous: accuracy 75.8%, precision 100%, baseline 90.9%
- New: should show improvement or different feature importance.

**Step 3: Commit the updated model**

```bash
git add -A
git commit -m "feat: retrain model with LLM-enriched features"
```

---

### Task 8: Update CLAUDE.md

**Step 1: Update commands and status**

Add to the commands section:
```
uv run python scripts/analyze_trades_llm.py           # Analyser trades via GPT-4o-mini
uv run python scripts/analyze_trades_llm.py --trade 42  # Analyser un trade specifique
uv run python scripts/analyze_trades_llm.py --stats     # Stats des analyses
```

Update the etape 4 status to mention LLM improvement.
Add `trade_analyses_llm` to the database tables section.
Add OpenAI to the stack technique table.

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for step 4bis LLM improvement"
```
