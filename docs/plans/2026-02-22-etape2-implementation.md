# Etape 2 — Collecte de donnees historiques — Plan d'implementation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Collecter les prix OHLCV et les news historiques autour des dates de chaque trade pour alimenter les etapes 3 (correlation) et 4 (ML).

**Architecture:** Batch one-shot: lire les trades existants, telecharger les prix via yfinance et les news via GNews, stocker en SQLite. 3 nouveaux fichiers dans src/data_collection/, 2 nouvelles tables (prices, news), 1 script CLI.

**Tech Stack:** yfinance (prix), gnews (news Google), sqlite3 (stockage), loguru (logs)

---

## Task 1: Ajouter les dependances

**Files:**
- Modify: `pyproject.toml`

**Step 1: Ajouter yfinance et gnews aux dependances**

Dans `pyproject.toml`, ajouter dans `dependencies`:
```toml
dependencies = [
    "loguru>=0.7.3",
    "pandas>=3.0.1",
    "pdfplumber>=0.11.9",
    "python-dotenv>=1.2.1",
    "pyyaml>=6.0.3",
    "yfinance>=0.2.50",
    "gnews>=0.4.3",
]
```

**Step 2: Installer les dependances**

Run: `uv sync`
Expected: Installation reussie, pas d'erreur

**Step 3: Verifier l'installation**

Run: `uv run python -c "import yfinance; import gnews; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add yfinance and gnews for historical data collection"
```

---

## Task 2: Ajouter les tables prices et news dans database.py

**Files:**
- Modify: `src/core/database.py`
- Test: `tests/test_database.py`

**Step 1: Ecrire les tests pour les nouvelles tables**

Ajouter dans `tests/test_database.py`:

```python
class TestPricesTable:
    """Tests pour la table prices."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_insert_price(self):
        """Insere un prix et le recupere."""
        price = {
            "ticker": "SAN.PA",
            "date": "2025-06-15",
            "open": 95.50,
            "high": 96.80,
            "low": 95.10,
            "close": 96.20,
            "volume": 1234567,
        }
        self.db.insert_price(price)
        prices = self.db.get_prices("SAN.PA")
        assert len(prices) == 1
        assert prices[0]["close"] == 96.20

    def test_insert_prices_batch(self):
        """Insere plusieurs prix en batch."""
        prices = [
            {"ticker": "SAN.PA", "date": "2025-06-15", "open": 95.5,
             "high": 96.8, "low": 95.1, "close": 96.2, "volume": 100},
            {"ticker": "SAN.PA", "date": "2025-06-16", "open": 96.2,
             "high": 97.0, "low": 95.8, "close": 96.5, "volume": 200},
        ]
        self.db.insert_prices_batch(prices)
        result = self.db.get_prices("SAN.PA")
        assert len(result) == 2

    def test_insert_price_doublon_ignore(self):
        """Un doublon (meme ticker+date) est ignore silencieusement."""
        price = {"ticker": "SAN.PA", "date": "2025-06-15", "open": 95.5,
                 "high": 96.8, "low": 95.1, "close": 96.2, "volume": 100}
        self.db.insert_price(price)
        self.db.insert_price(price)  # doublon
        prices = self.db.get_prices("SAN.PA")
        assert len(prices) == 1


class TestNewsTable:
    """Tests pour la table news."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_insert_news(self):
        """Insere une news et la recupere."""
        news = {
            "ticker": "SAN.PA",
            "title": "Sanofi: resultats T2 au-dessus des attentes",
            "source": "BFM Bourse",
            "url": "https://example.com/sanofi-t2",
            "published_at": "2025-06-15 08:30:00",
            "description": "Le laboratoire publie des resultats...",
        }
        self.db.insert_news(news)
        result = self.db.get_news("SAN.PA")
        assert len(result) == 1
        assert result[0]["title"] == "Sanofi: resultats T2 au-dessus des attentes"

    def test_insert_news_batch(self):
        """Insere plusieurs news en batch."""
        news_list = [
            {"ticker": "SAN.PA", "title": "News 1", "source": "BFM",
             "url": "https://example.com/1", "published_at": "2025-06-15 08:30:00",
             "description": "Desc 1"},
            {"ticker": "SAN.PA", "title": "News 2", "source": "Les Echos",
             "url": "https://example.com/2", "published_at": "2025-06-16 09:00:00",
             "description": "Desc 2"},
        ]
        self.db.insert_news_batch(news_list)
        result = self.db.get_news("SAN.PA")
        assert len(result) == 2

    def test_insert_news_doublon_url_ignore(self):
        """Un doublon (meme URL) est ignore silencieusement."""
        news = {"ticker": "SAN.PA", "title": "News 1", "source": "BFM",
                "url": "https://example.com/1", "published_at": "2025-06-15 08:30:00",
                "description": "Desc 1"}
        self.db.insert_news(news)
        self.db.insert_news(news)  # doublon
        result = self.db.get_news("SAN.PA")
        assert len(result) == 1
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_database.py -v -k "Prices or News"`
Expected: FAIL (methodes insert_price, get_prices, etc. n'existent pas encore)

**Step 3: Ajouter les tables et methodes dans database.py**

Dans `src/core/database.py`, ajouter dans `init_db()` (dans le executescript):

```sql
CREATE TABLE IF NOT EXISTS prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS news (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    title TEXT NOT NULL,
    source TEXT,
    url TEXT UNIQUE,
    published_at TEXT,
    description TEXT
);

CREATE INDEX IF NOT EXISTS idx_prices_ticker_date
    ON prices(ticker, date);
CREATE INDEX IF NOT EXISTS idx_news_ticker
    ON news(ticker);
```

Ajouter les methodes:

```python
def insert_price(self, price: dict):
    """Insere un prix OHLCV. Ignore les doublons (ticker+date)."""
    conn = self._connect()
    conn.execute("""
        INSERT OR IGNORE INTO prices
            (ticker, date, open, high, low, close, volume)
        VALUES
            (:ticker, :date, :open, :high, :low, :close, :volume)
    """, price)
    conn.commit()
    conn.close()

def insert_prices_batch(self, prices: list[dict]):
    """Insere plusieurs prix en batch. Ignore les doublons."""
    conn = self._connect()
    conn.executemany("""
        INSERT OR IGNORE INTO prices
            (ticker, date, open, high, low, close, volume)
        VALUES
            (:ticker, :date, :open, :high, :low, :close, :volume)
    """, prices)
    conn.commit()
    conn.close()
    logger.info(f"{len(prices)} prix inseres en batch")

def get_prices(self, ticker: str) -> list[dict]:
    """Recupere les prix pour un ticker, tries par date."""
    conn = self._connect()
    rows = conn.execute(
        "SELECT * FROM prices WHERE ticker = ? ORDER BY date",
        (ticker,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]

def insert_news(self, news: dict):
    """Insere une news. Ignore les doublons (meme URL)."""
    conn = self._connect()
    conn.execute("""
        INSERT OR IGNORE INTO news
            (ticker, title, source, url, published_at, description)
        VALUES
            (:ticker, :title, :source, :url, :published_at, :description)
    """, news)
    conn.commit()
    conn.close()

def insert_news_batch(self, news_list: list[dict]):
    """Insere plusieurs news en batch. Ignore les doublons."""
    conn = self._connect()
    conn.executemany("""
        INSERT OR IGNORE INTO news
            (ticker, title, source, url, published_at, description)
        VALUES
            (:ticker, :title, :source, :url, :published_at, :description)
    """, news_list)
    conn.commit()
    conn.close()
    logger.info(f"{len(news_list)} news inserees en batch")

def get_news(self, ticker: str) -> list[dict]:
    """Recupere les news pour un ticker, triees par date."""
    conn = self._connect()
    rows = conn.execute(
        "SELECT * FROM news WHERE ticker = ? ORDER BY published_at",
        (ticker,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]

def count_prices(self) -> int:
    """Compte le nombre total de prix."""
    conn = self._connect()
    count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    conn.close()
    return count

def count_news(self) -> int:
    """Compte le nombre total de news."""
    conn = self._connect()
    count = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    conn.close()
    return count
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_database.py -v`
Expected: ALL PASS (anciens + nouveaux)

**Step 5: Commit**

```bash
git add src/core/database.py tests/test_database.py
git commit -m "feat: add prices and news tables to database"
```

---

## Task 3: Creer ticker_mapper.py

**Files:**
- Create: `src/data_collection/__init__.py`
- Create: `src/data_collection/ticker_mapper.py`
- Test: `tests/test_ticker_mapper.py`

**Step 1: Creer __init__.py**

Fichier vide: `src/data_collection/__init__.py`

**Step 2: Ecrire les tests**

Creer `tests/test_ticker_mapper.py`:

```python
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
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_ticker_mapper.py -v`
Expected: FAIL (module n'existe pas)

**Step 4: Implementer ticker_mapper.py**

Creer `src/data_collection/ticker_mapper.py`:

```python
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
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_ticker_mapper.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/data_collection/__init__.py src/data_collection/ticker_mapper.py tests/test_ticker_mapper.py
git commit -m "feat: add ticker mapper for nom_action to Yahoo Finance symbols"
```

---

## Task 4: Creer price_collector.py

**Files:**
- Create: `src/data_collection/price_collector.py`
- Test: `tests/test_price_collector.py`

**Step 1: Ecrire les tests**

Creer `tests/test_price_collector.py`:

```python
"""Tests pour la collecte de prix historiques."""

import os
import tempfile
from datetime import datetime
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.core.database import Database
from src.data_collection.price_collector import PriceCollector


class TestPriceCollector:
    """Tests de collecte de prix (yfinance mocke)."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.collector = PriceCollector(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def _make_mock_df(self):
        """Cree un DataFrame similaire a yfinance.download()."""
        dates = pd.date_range("2025-06-01", periods=3, freq="B")
        return pd.DataFrame({
            "Open": [95.0, 95.5, 96.0],
            "High": [96.0, 96.5, 97.0],
            "Low": [94.5, 95.0, 95.5],
            "Close": [95.5, 96.0, 96.5],
            "Volume": [100000, 120000, 110000],
        }, index=dates)

    @patch("src.data_collection.price_collector.yf.download")
    def test_collect_prices_for_ticker(self, mock_download):
        """Collecte les prix pour un ticker et les insere en base."""
        mock_download.return_value = self._make_mock_df()

        count = self.collector.collect_for_ticker(
            "SAN.PA", "2025-06-01", "2025-06-05"
        )

        assert count == 3
        prices = self.db.get_prices("SAN.PA")
        assert len(prices) == 3
        assert prices[0]["close"] == 95.5

    @patch("src.data_collection.price_collector.yf.download")
    def test_collect_prices_empty_result(self, mock_download):
        """Retourne 0 si yfinance ne retourne rien."""
        mock_download.return_value = pd.DataFrame()

        count = self.collector.collect_for_ticker(
            "INCONNU.PA", "2025-06-01", "2025-06-05"
        )

        assert count == 0

    @patch("src.data_collection.price_collector.yf.download")
    def test_collect_deduplication(self, mock_download):
        """Les doublons sont ignores lors d'une 2e collecte."""
        mock_download.return_value = self._make_mock_df()

        self.collector.collect_for_ticker("SAN.PA", "2025-06-01", "2025-06-05")
        self.collector.collect_for_ticker("SAN.PA", "2025-06-01", "2025-06-05")

        prices = self.db.get_prices("SAN.PA")
        assert len(prices) == 3  # pas de doublons

    def test_compute_date_ranges_from_trades(self):
        """Calcule les plages de dates a partir des trades."""
        trades = [
            {"nom_action": "SANOFI", "isin": "FR123",
             "date_achat": "2025-06-15 10:00:00", "date_vente": "2025-06-25 14:00:00",
             "statut": "CLOTURE"},
            {"nom_action": "SANOFI", "isin": "FR123",
             "date_achat": "2025-07-01 09:00:00", "date_vente": "2025-07-10 15:00:00",
             "statut": "CLOTURE"},
        ]
        ranges = self.collector.compute_date_ranges(trades)
        # SANOFI: min_achat=2025-06-15, max_vente=2025-07-10
        # Avec 30j avant: 2025-05-16, fin: 2025-07-10
        assert "SANOFI" in ranges
        assert ranges["SANOFI"]["start"] == "2025-05-16"
        assert ranges["SANOFI"]["end"] == "2025-07-10"

    def test_compute_date_ranges_trade_ouvert(self):
        """Un trade ouvert (sans date_vente) utilise la date du jour."""
        trades = [
            {"nom_action": "ADOCIA", "isin": "FR456",
             "date_achat": "2025-11-04 09:00:00", "date_vente": None,
             "statut": "OUVERT"},
        ]
        ranges = self.collector.compute_date_ranges(trades)
        assert "ADOCIA" in ranges
        # La date de fin doit etre aujourd'hui ou apres
        assert ranges["ADOCIA"]["end"] >= "2026-02-22"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_price_collector.py -v`
Expected: FAIL (module n'existe pas)

**Step 3: Implementer price_collector.py**

Creer `src/data_collection/price_collector.py`:

```python
"""Collecte des prix OHLCV historiques via yfinance."""

from datetime import datetime, timedelta

import yfinance as yf
from loguru import logger

from src.core.database import Database
from src.data_collection.ticker_mapper import TickerMapper, TickerNotFoundError

DAYS_BEFORE_TRADE = 30


class PriceCollector:
    """Collecte les prix historiques pour les actions tradees."""

    def __init__(self, db: Database):
        self.db = db
        self.mapper = TickerMapper()

    def collect_for_ticker(self, ticker: str, start: str, end: str) -> int:
        """Telecharge les prix OHLCV pour un ticker et une periode.

        Returns:
            Nombre de jours de prix inseres.
        """
        logger.info(f"Collecte prix {ticker} du {start} au {end}")
        df = yf.download(ticker, start=start, end=end, progress=False)

        if df.empty:
            logger.warning(f"Aucun prix retourne pour {ticker}")
            return 0

        prices = []
        for date, row in df.iterrows():
            prices.append({
                "ticker": ticker,
                "date": date.strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "volume": int(row["Volume"]),
            })

        self.db.insert_prices_batch(prices)
        logger.info(f"{len(prices)} prix inseres pour {ticker}")
        return len(prices)

    def compute_date_ranges(self, trades: list[dict]) -> dict:
        """Calcule les plages de dates par action a partir des trades.

        Pour chaque action: start = min(date_achat) - 30 jours, end = max(date_vente).
        Si trade ouvert (date_vente=None), end = aujourd'hui.

        Returns:
            Dict {nom_action: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}}
        """
        ranges = {}
        today = datetime.now().strftime("%Y-%m-%d")

        for trade in trades:
            name = trade["nom_action"].lstrip("* ").strip()
            date_achat = trade["date_achat"][:10]  # "YYYY-MM-DD"
            date_vente = trade["date_vente"][:10] if trade["date_vente"] else today

            if name not in ranges:
                ranges[name] = {"start": date_achat, "end": date_vente}
            else:
                if date_achat < ranges[name]["start"]:
                    ranges[name]["start"] = date_achat
                if date_vente > ranges[name]["end"]:
                    ranges[name]["end"] = date_vente

        # Reculer le start de DAYS_BEFORE_TRADE jours
        for name in ranges:
            start_dt = datetime.strptime(ranges[name]["start"], "%Y-%m-%d")
            ranges[name]["start"] = (
                start_dt - timedelta(days=DAYS_BEFORE_TRADE)
            ).strftime("%Y-%m-%d")

        return ranges

    def collect_all(self) -> dict:
        """Collecte les prix pour toutes les actions tradees.

        Returns:
            Dict {"total_prices": int, "errors": list}
        """
        trades = self.db.get_all_trades()
        ranges = self.compute_date_ranges(trades)

        total = 0
        errors = []

        for nom_action, period in ranges.items():
            try:
                ticker = self.mapper.get_ticker(nom_action)
                count = self.collect_for_ticker(
                    ticker, period["start"], period["end"]
                )
                total += count
            except TickerNotFoundError as e:
                errors.append({"action": nom_action, "error": str(e)})
                logger.warning(f"Ticker inconnu: {nom_action}")
            except Exception as e:
                errors.append({"action": nom_action, "error": str(e)})
                logger.error(f"Erreur collecte prix {nom_action}: {e}")

        logger.info(f"Collecte prix terminee: {total} prix, {len(errors)} erreurs")
        return {"total_prices": total, "errors": errors}
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_price_collector.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/data_collection/price_collector.py tests/test_price_collector.py
git commit -m "feat: add price collector with yfinance for historical OHLCV"
```

---

## Task 5: Creer news_collector.py

**Files:**
- Create: `src/data_collection/news_collector.py`
- Test: `tests/test_news_collector.py`

**Step 1: Ecrire les tests**

Creer `tests/test_news_collector.py`:

```python
"""Tests pour la collecte de news historiques."""

import os
import tempfile
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from src.core.database import Database
from src.data_collection.news_collector import NewsCollector


class TestNewsCollector:
    """Tests de collecte de news (GNews mocke)."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.collector = NewsCollector(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def _make_mock_articles(self):
        """Cree des articles similaires a GNews.get_news()."""
        return [
            {
                "title": "Sanofi: resultats T2 solides",
                "description": "Le labo publie des resultats...",
                "url": "https://example.com/sanofi-1",
                "published date": "Mon, 15 Jun 2025 08:30:00 GMT",
                "publisher": {"title": "BFM Bourse"},
            },
            {
                "title": "Sanofi lance un nouveau medicament",
                "description": "Approbation FDA pour...",
                "url": "https://example.com/sanofi-2",
                "published date": "Tue, 16 Jun 2025 10:00:00 GMT",
                "publisher": {"title": "Les Echos"},
            },
        ]

    @patch("src.data_collection.news_collector.GNews")
    def test_collect_news_for_action(self, MockGNews):
        """Collecte les news pour une action et les insere en base."""
        mock_gn = MagicMock()
        mock_gn.get_news.return_value = self._make_mock_articles()
        MockGNews.return_value = mock_gn

        count = self.collector.collect_for_action(
            "SANOFI", "SAN.PA", "2025-06-08", "2025-06-18"
        )

        assert count == 2
        news = self.db.get_news("SAN.PA")
        assert len(news) == 2

    @patch("src.data_collection.news_collector.GNews")
    def test_collect_news_empty(self, MockGNews):
        """Retourne 0 si GNews ne retourne rien."""
        mock_gn = MagicMock()
        mock_gn.get_news.return_value = []
        MockGNews.return_value = mock_gn

        count = self.collector.collect_for_action(
            "INCONNU", "INC.PA", "2025-06-08", "2025-06-18"
        )

        assert count == 0

    @patch("src.data_collection.news_collector.GNews")
    def test_collect_news_deduplication(self, MockGNews):
        """Les doublons (meme URL) sont ignores."""
        mock_gn = MagicMock()
        mock_gn.get_news.return_value = self._make_mock_articles()
        MockGNews.return_value = mock_gn

        self.collector.collect_for_action("SANOFI", "SAN.PA", "2025-06-08", "2025-06-18")
        self.collector.collect_for_action("SANOFI", "SAN.PA", "2025-06-08", "2025-06-18")

        news = self.db.get_news("SAN.PA")
        assert len(news) == 2  # pas de doublons

    def test_compute_news_windows(self):
        """Calcule les fenetres de recherche de news autour des trades."""
        trades = [
            {"nom_action": "SANOFI", "date_achat": "2025-06-15 10:00:00",
             "date_vente": "2025-06-25 14:00:00", "statut": "CLOTURE"},
            {"nom_action": "SANOFI", "date_achat": "2025-07-01 09:00:00",
             "date_vente": "2025-07-10 15:00:00", "statut": "CLOTURE"},
        ]
        windows = self.collector.compute_news_windows(trades)
        # 2 trades SANOFI -> 2 fenetres distinctes
        sanofi_windows = [w for w in windows if w["nom_action"] == "SANOFI"]
        assert len(sanofi_windows) == 2
        # Premiere fenetre: 2025-06-08 a 2025-06-18
        assert sanofi_windows[0]["start"] == "2025-06-08"
        assert sanofi_windows[0]["end"] == "2025-06-18"

    @patch("src.data_collection.news_collector.GNews")
    def test_parse_article_handles_missing_fields(self, MockGNews):
        """Les articles avec des champs manquants sont geres proprement."""
        mock_gn = MagicMock()
        mock_gn.get_news.return_value = [
            {
                "title": "Article sans description",
                "url": "https://example.com/no-desc",
                "published date": "Mon, 15 Jun 2025 08:30:00 GMT",
                "publisher": {"title": "Source"},
            },
        ]
        MockGNews.return_value = mock_gn

        count = self.collector.collect_for_action(
            "SANOFI", "SAN.PA", "2025-06-08", "2025-06-18"
        )

        assert count == 1
        news = self.db.get_news("SAN.PA")
        assert news[0]["description"] is None or news[0]["description"] == ""
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_news_collector.py -v`
Expected: FAIL (module n'existe pas)

**Step 3: Implementer news_collector.py**

Creer `src/data_collection/news_collector.py`:

```python
"""Collecte des news historiques via GNews (Google News RSS)."""

import time
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime

from gnews import GNews
from loguru import logger

from src.core.database import Database
from src.data_collection.ticker_mapper import TickerMapper, TickerNotFoundError

DAYS_BEFORE = 7
DAYS_AFTER = 3
DELAY_BETWEEN_REQUESTS = 3  # secondes


class NewsCollector:
    """Collecte les news historiques pour les actions tradees."""

    def __init__(self, db: Database):
        self.db = db
        self.mapper = TickerMapper()

    def _parse_article(self, article: dict, ticker: str) -> dict:
        """Transforme un article GNews en dict pour la base."""
        published_at = ""
        raw_date = article.get("published date", "")
        if raw_date:
            try:
                dt = parsedate_to_datetime(raw_date)
                published_at = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                published_at = raw_date

        publisher = article.get("publisher", {})
        source = publisher.get("title", "") if isinstance(publisher, dict) else ""

        return {
            "ticker": ticker,
            "title": article.get("title", ""),
            "source": source,
            "url": article.get("url", ""),
            "published_at": published_at,
            "description": article.get("description", ""),
        }

    def collect_for_action(
        self, nom_action: str, ticker: str, start: str, end: str
    ) -> int:
        """Collecte les news pour une action sur une periode.

        Args:
            nom_action: Nom de l'action pour la recherche (ex: "SANOFI")
            ticker: Ticker Yahoo pour le stockage (ex: "SAN.PA")
            start: Date debut YYYY-MM-DD
            end: Date fin YYYY-MM-DD

        Returns:
            Nombre de news inserees.
        """
        logger.info(f"Collecte news {nom_action} du {start} au {end}")

        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")

        gn = GNews(
            language="fr",
            country="FR",
            start_date=(start_dt.year, start_dt.month, start_dt.day),
            end_date=(end_dt.year, end_dt.month, end_dt.day),
            max_results=100,
        )

        query = f"{nom_action} bourse action"
        articles = gn.get_news(query)

        if not articles:
            logger.warning(f"Aucune news pour {nom_action} ({start} -> {end})")
            return 0

        news_list = [self._parse_article(a, ticker) for a in articles]
        # Filtrer les articles sans URL (pas de deduplication possible)
        news_list = [n for n in news_list if n["url"]]

        self.db.insert_news_batch(news_list)
        logger.info(f"{len(news_list)} news inserees pour {nom_action}")
        return len(news_list)

    def compute_news_windows(self, trades: list[dict]) -> list[dict]:
        """Calcule les fenetres de recherche de news pour chaque trade.

        Pour chaque trade: start = date_achat - 7j, end = date_achat + 3j.
        Retourne une liste de fenetres (une par trade, pas par action).

        Returns:
            List[{"nom_action": str, "start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}]
        """
        windows = []
        for trade in trades:
            date_achat = datetime.strptime(
                trade["date_achat"][:10], "%Y-%m-%d"
            )
            start = (date_achat - timedelta(days=DAYS_BEFORE)).strftime("%Y-%m-%d")
            end = (date_achat + timedelta(days=DAYS_AFTER)).strftime("%Y-%m-%d")
            name = trade["nom_action"].lstrip("* ").strip()

            windows.append({
                "nom_action": name,
                "start": start,
                "end": end,
            })

        return windows

    def _deduplicate_windows(self, windows: list[dict]) -> list[dict]:
        """Fusionne les fenetres qui se chevauchent pour la meme action."""
        by_action = {}
        for w in windows:
            name = w["nom_action"]
            if name not in by_action:
                by_action[name] = []
            by_action[name].append((w["start"], w["end"]))

        result = []
        for name, periods in by_action.items():
            periods.sort()
            merged = [periods[0]]
            for start, end in periods[1:]:
                prev_start, prev_end = merged[-1]
                if start <= prev_end:
                    merged[-1] = (prev_start, max(prev_end, end))
                else:
                    merged.append((start, end))
            for start, end in merged:
                result.append({"nom_action": name, "start": start, "end": end})

        return result

    def collect_all(self) -> dict:
        """Collecte les news pour toutes les actions tradees.

        Returns:
            Dict {"total_news": int, "errors": list}
        """
        trades = self.db.get_all_trades()
        windows = self.compute_news_windows(trades)
        windows = self._deduplicate_windows(windows)

        total = 0
        errors = []

        for window in windows:
            try:
                ticker = self.mapper.get_ticker(window["nom_action"])
                count = self.collect_for_action(
                    window["nom_action"], ticker,
                    window["start"], window["end"],
                )
                total += count
                time.sleep(DELAY_BETWEEN_REQUESTS)
            except TickerNotFoundError as e:
                errors.append({"action": window["nom_action"], "error": str(e)})
                logger.warning(f"Ticker inconnu: {window['nom_action']}")
            except Exception as e:
                errors.append({"action": window["nom_action"], "error": str(e)})
                logger.error(f"Erreur collecte news {window['nom_action']}: {e}")

        logger.info(f"Collecte news terminee: {total} news, {len(errors)} erreurs")
        return {"total_news": total, "errors": errors}
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_news_collector.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/data_collection/news_collector.py tests/test_news_collector.py
git commit -m "feat: add news collector with GNews for historical articles"
```

---

## Task 6: Creer le script collect_historical.py

**Files:**
- Create: `scripts/collect_historical.py`

**Step 1: Creer le script**

```python
"""Script de collecte de donnees historiques.

Usage:
    uv run python scripts/collect_historical.py          # Tout collecter
    uv run python scripts/collect_historical.py --prices  # Seulement les prix
    uv run python scripts/collect_historical.py --news    # Seulement les news
"""

import argparse
import sys

from loguru import logger

from src.core.database import Database
from src.data_collection.price_collector import PriceCollector
from src.data_collection.news_collector import NewsCollector

DB_PATH = "data/trades.db"


def main():
    parser = argparse.ArgumentParser(description="Collecte donnees historiques")
    parser.add_argument("--prices", action="store_true", help="Collecter les prix")
    parser.add_argument("--news", action="store_true", help="Collecter les news")
    args = parser.parse_args()

    # Si aucun flag, tout collecter
    collect_prices = args.prices or (not args.prices and not args.news)
    collect_news = args.news or (not args.prices and not args.news)

    db = Database(DB_PATH)
    db.init_db()

    print(f"Base: {DB_PATH}")
    print(f"Trades en base: {db.count_trades()}")
    print()

    if collect_prices:
        print("=== Collecte des prix ===")
        collector = PriceCollector(db)
        result = collector.collect_all()
        print(f"Prix collectes: {result['total_prices']}")
        if result["errors"]:
            print(f"Erreurs: {len(result['errors'])}")
            for e in result["errors"]:
                print(f"  - {e['action']}: {e['error']}")
        print()

    if collect_news:
        print("=== Collecte des news ===")
        collector = NewsCollector(db)
        result = collector.collect_all()
        print(f"News collectees: {result['total_news']}")
        if result["errors"]:
            print(f"Erreurs: {len(result['errors'])}")
            for e in result["errors"]:
                print(f"  - {e['action']}: {e['error']}")
        print()

    print("=== Bilan ===")
    print(f"Total prix en base: {db.count_prices()}")
    print(f"Total news en base: {db.count_news()}")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/collect_historical.py
git commit -m "feat: add collect_historical.py CLI script"
```

---

## Task 7: Test d'integration — run complet

**Step 1: Lancer tous les tests unitaires**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS (anciens 30 + nouveaux ~15)

**Step 2: Verifier les tickers avec un appel reel (optionnel)**

Run: `uv run python -c "import yfinance as yf; print(yf.download('SAN.PA', period='5d'))"`
Expected: DataFrame avec 5 jours de prix SANOFI

**Step 3: Lancer la collecte des prix**

Run: `uv run python scripts/collect_historical.py --prices`
Expected: ~X prix collectes pour chaque action, quelques erreurs possibles sur les tickers

**Step 4: Lancer la collecte des news**

Run: `uv run python scripts/collect_historical.py --news`
Expected: News collectees pour chaque action, rapport final

**Step 5: Commit final**

```bash
git add -A
git commit -m "feat: etape 2 complete - collecte historique prix + news"
```

---

## Verification finale

Apres toutes les taches:

```
uv run pytest tests/ -v                    # Tous tests passent
uv run python scripts/collect_historical.py # Collecte complete
```

Tables SQLite attendues:
- `prices`: ~200-400 lignes (19 actions x ~10-20 jours chacune)
- `news`: ~50-200 articles (variable selon la couverture GNews)
