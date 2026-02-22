# Etape 3 — Catalyst Matcher Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Associer chaque trade a ses catalyseurs (news autour de la date d'achat) et peupler la table trade_catalyseurs.

**Architecture:** SQL jointure filtree par ticker + fenetre temporelle J-3/J+1, scoring Python (proximite temporelle + matching texte). Le CatalystMatcher utilise Database et TickerMapper existants.

**Tech Stack:** sqlite3 (stdlib), loguru, TickerMapper existant. Aucune nouvelle dependance.

**Design doc:** `docs/plans/2026-02-22-etape3-catalyst-matcher-design.md`

---

### Task 1: Table trade_catalyseurs dans Database

**Files:**
- Modify: `src/core/database.py` (ajouter table + 5 methodes)
- Modify: `scripts/init_db.py` (ajouter creation table)
- Test: `tests/test_database.py` (ajouter classe TestCatalyseursTable)

**Step 1: Write the failing tests**

Ajouter a la fin de `tests/test_database.py`:

```python
class TestCatalyseursTable:
    """Tests pour la table trade_catalyseurs."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        # Inserer un trade et une news pour les FK
        self.db.insert_trade_complet({
            "isin": "FR0000120578", "nom_action": "SANOFI",
            "date_achat": "2025-07-10", "date_vente": "2025-07-15",
            "prix_achat": 95.0, "prix_vente": 100.0, "quantite": 10,
            "rendement_brut_pct": 5.26, "rendement_net_pct": 5.0,
            "duree_jours": 5, "frais_totaux": 2.5, "statut": "CLOTURE",
        })
        self.db.insert_news({
            "ticker": "SAN.PA", "title": "Sanofi annonce resultats",
            "source": "Reuters", "url": "https://example.com/sanofi-1",
            "published_at": "2025-07-09", "description": "Bons resultats",
            "sentiment": 0.5, "source_api": "gnews",
        })

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_insert_catalyseur(self):
        """Insere un catalyseur et le recupere."""
        catalyseur = {
            "trade_id": 1, "news_id": 1,
            "score_pertinence": 0.8, "distance_jours": -1,
            "match_texte": 1,
        }
        self.db.insert_catalyseur(catalyseur)
        result = self.db.get_catalyseurs(trade_id=1)
        assert len(result) == 1
        assert result[0]["score_pertinence"] == 0.8
        assert result[0]["distance_jours"] == -1
        assert result[0]["match_texte"] == 1

    def test_insert_catalyseurs_batch(self):
        """Insere plusieurs catalyseurs en batch."""
        # Ajouter une 2e news
        self.db.insert_news({
            "ticker": "SAN.PA", "title": "Sanofi FDA approval",
            "source": "Bloomberg", "url": "https://example.com/sanofi-2",
            "published_at": "2025-07-10", "description": "FDA approves drug",
            "sentiment": 0.8, "source_api": "alpha_vantage",
        })
        catalyseurs = [
            {"trade_id": 1, "news_id": 1, "score_pertinence": 0.8,
             "distance_jours": -1, "match_texte": 1},
            {"trade_id": 1, "news_id": 2, "score_pertinence": 1.0,
             "distance_jours": 0, "match_texte": 1},
        ]
        self.db.insert_catalyseurs_batch(catalyseurs)
        result = self.db.get_catalyseurs(trade_id=1)
        assert len(result) == 2

    def test_insert_catalyseur_doublon_ignore(self):
        """Les doublons (meme trade_id+news_id) sont ignores."""
        cat = {"trade_id": 1, "news_id": 1, "score_pertinence": 0.8,
               "distance_jours": -1, "match_texte": 1}
        self.db.insert_catalyseur(cat)
        self.db.insert_catalyseur(cat)  # doublon
        assert self.db.count_catalyseurs() == 1

    def test_count_catalyseurs(self):
        """Compte le nombre total de catalyseurs."""
        assert self.db.count_catalyseurs() == 0
        self.db.insert_catalyseur({
            "trade_id": 1, "news_id": 1, "score_pertinence": 0.8,
            "distance_jours": -1, "match_texte": 1,
        })
        assert self.db.count_catalyseurs() == 1

    def test_clear_catalyseurs(self):
        """Vide la table trade_catalyseurs."""
        self.db.insert_catalyseur({
            "trade_id": 1, "news_id": 1, "score_pertinence": 0.8,
            "distance_jours": -1, "match_texte": 1,
        })
        self.db.clear_catalyseurs()
        assert self.db.count_catalyseurs() == 0

    def test_get_news_for_trade_window(self):
        """Recupere les news dans une fenetre temporelle pour un ticker."""
        result = self.db.get_news_in_window(
            ticker="SAN.PA",
            date_start="2025-07-07",
            date_end="2025-07-11",
        )
        assert len(result) == 1
        assert result[0]["title"] == "Sanofi annonce resultats"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_database.py::TestCatalyseursTable -v`
Expected: FAIL (methods not defined)

**Step 3: Implement database methods**

Dans `src/core/database.py`, ajouter dans `init_db()` apres le bloc `executescript`:

```python
        # Table trade_catalyseurs (etape 3)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trade_catalyseurs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER NOT NULL,
                news_id INTEGER NOT NULL,
                score_pertinence REAL NOT NULL,
                distance_jours INTEGER NOT NULL,
                match_texte INTEGER DEFAULT 0,
                UNIQUE(trade_id, news_id),
                FOREIGN KEY (trade_id) REFERENCES trades_complets(id),
                FOREIGN KEY (news_id) REFERENCES news(id)
            );
            CREATE INDEX IF NOT EXISTS idx_catalyseurs_trade
                ON trade_catalyseurs(trade_id);
            CREATE INDEX IF NOT EXISTS idx_catalyseurs_news
                ON trade_catalyseurs(news_id);
        """)
```

Ajouter ces methodes dans la classe Database (section `# --- Catalyseurs ---`):

```python
    # --- Catalyseurs ---

    def insert_catalyseur(self, catalyseur: dict):
        """Insere un catalyseur. Ignore les doublons (trade_id+news_id)."""
        conn = self._connect()
        conn.execute("""
            INSERT OR IGNORE INTO trade_catalyseurs
                (trade_id, news_id, score_pertinence, distance_jours, match_texte)
            VALUES
                (:trade_id, :news_id, :score_pertinence, :distance_jours, :match_texte)
        """, catalyseur)
        conn.commit()
        conn.close()

    def insert_catalyseurs_batch(self, catalyseurs: list[dict]):
        """Insere plusieurs catalyseurs en batch. Ignore les doublons."""
        conn = self._connect()
        conn.executemany("""
            INSERT OR IGNORE INTO trade_catalyseurs
                (trade_id, news_id, score_pertinence, distance_jours, match_texte)
            VALUES
                (:trade_id, :news_id, :score_pertinence, :distance_jours, :match_texte)
        """, catalyseurs)
        conn.commit()
        conn.close()
        logger.info(f"{len(catalyseurs)} catalyseurs inseres en batch")

    def get_catalyseurs(self, trade_id: int) -> list[dict]:
        """Recupere les catalyseurs pour un trade donne."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM trade_catalyseurs WHERE trade_id = ? ORDER BY distance_jours",
            (trade_id,),
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def count_catalyseurs(self) -> int:
        """Compte le nombre total de catalyseurs."""
        conn = self._connect()
        count = conn.execute(
            "SELECT COUNT(*) FROM trade_catalyseurs"
        ).fetchone()[0]
        conn.close()
        return count

    def clear_catalyseurs(self):
        """Vide la table trade_catalyseurs."""
        conn = self._connect()
        conn.execute("DELETE FROM trade_catalyseurs")
        conn.commit()
        conn.close()
        logger.info("Table trade_catalyseurs videe")

    def get_news_in_window(self, ticker: str, date_start: str, date_end: str) -> list[dict]:
        """Recupere les news pour un ticker dans une fenetre de dates."""
        conn = self._connect()
        rows = conn.execute("""
            SELECT * FROM news
            WHERE ticker = ?
              AND date(published_at) BETWEEN date(?) AND date(?)
            ORDER BY published_at
        """, (ticker, date_start, date_end)).fetchall()
        conn.close()
        return [dict(row) for row in rows]
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_database.py::TestCatalyseursTable -v`
Expected: 6 PASS

**Step 5: Run all database tests (regression)**

Run: `uv run pytest tests/test_database.py -v`
Expected: 18 PASS (12 existants + 6 nouveaux)

**Step 6: Update init_db.py**

Dans `scripts/init_db.py`, verifier que `db.init_db()` est appele (la creation de table est deja dans init_db). Pas de modification necessaire si init_db.py appelle simplement `db.init_db()`.

**Step 7: Commit**

```bash
git add src/core/database.py tests/test_database.py
git commit -m "feat(db): add trade_catalyseurs table and CRUD methods"
```

---

### Task 2: Package analysis + CatalystMatcher core

**Files:**
- Create: `src/analysis/__init__.py`
- Create: `src/analysis/catalyst_matcher.py`
- Create: `tests/test_catalyst_matcher.py`

**Step 1: Create the analysis package**

Creer `src/analysis/__init__.py` (fichier vide, comme les autres packages).

**Step 2: Write the failing tests**

Creer `tests/test_catalyst_matcher.py`:

```python
"""Tests pour le catalyst_matcher."""

import os
import tempfile

import pytest

from src.core.database import Database
from src.analysis.catalyst_matcher import CatalystMatcher


class TestComputeScore:
    """Tests du calcul de score de pertinence."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.matcher = CatalystMatcher(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_score_jour_achat(self):
        """J0 (jour d'achat) = score max 1.0."""
        assert self.matcher._compute_score(0, False) == 1.0

    def test_score_veille(self):
        """J-1 = 0.8."""
        assert self.matcher._compute_score(-1, False) == 0.8

    def test_score_j_moins_2(self):
        """J-2 = 0.6."""
        assert self.matcher._compute_score(-2, False) == 0.6

    def test_score_j_moins_3(self):
        """J-3 = 0.4."""
        assert self.matcher._compute_score(-3, False) == 0.4

    def test_score_lendemain(self):
        """J+1 = 0.7."""
        assert self.matcher._compute_score(1, False) == 0.7

    def test_score_avec_match_texte(self):
        """Match texte ajoute +0.2, cap a 1.0."""
        assert self.matcher._compute_score(-1, True) == 1.0  # 0.8 + 0.2
        assert self.matcher._compute_score(0, True) == 1.0   # 1.0 + 0.2 cappe a 1.0

    def test_score_distance_hors_fenetre(self):
        """Distance hors fenetre retourne 0.0."""
        assert self.matcher._compute_score(-5, False) == 0.0
        assert self.matcher._compute_score(3, False) == 0.0


class TestCheckTextMatch:
    """Tests du matching texte nom_action dans titre/description."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.matcher = CatalystMatcher(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_match_exact_titre(self):
        """Nom exact dans le titre."""
        assert self.matcher._check_text_match(
            "SANOFI", "Sanofi annonce ses resultats Q3", ""
        ) is True

    def test_match_exact_description(self):
        """Nom exact dans la description."""
        assert self.matcher._check_text_match(
            "SANOFI", "Actu pharma", "Sanofi a publie ses resultats"
        ) is True

    def test_no_match(self):
        """Aucun match."""
        assert self.matcher._check_text_match(
            "SANOFI", "L'Oreal publie ses resultats", "Bons chiffres"
        ) is False

    def test_match_case_insensitive(self):
        """Matching insensible a la casse."""
        assert self.matcher._check_text_match(
            "AIR LIQUIDE", "air liquide en hausse", ""
        ) is True

    def test_match_nom_avec_etoile(self):
        """Noms avec prefixe * (achats anterieurs aux PDF)."""
        assert self.matcher._check_text_match(
            "* GENFIT", "Genfit obtient un brevet", ""
        ) is True

    def test_match_none_description(self):
        """Description None ne crashe pas."""
        assert self.matcher._check_text_match(
            "SANOFI", "Sanofi news", None
        ) is True

    def test_no_match_none_description(self):
        """Description None, pas de match dans titre."""
        assert self.matcher._check_text_match(
            "SANOFI", "L'Oreal news", None
        ) is False


class TestMatchTrade:
    """Tests du matching d'un trade avec les news."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.matcher = CatalystMatcher(self.db)
        # Inserer des news de test
        self.db.insert_news_batch([
            {"ticker": "SAN.PA", "title": "Sanofi resultats Q3",
             "source": "Reuters", "url": "https://ex.com/1",
             "published_at": "2025-07-09 10:00:00",
             "description": "Bons resultats Sanofi",
             "sentiment": 0.5, "source_api": "gnews"},
            {"ticker": "SAN.PA", "title": "FDA approves Sanofi drug",
             "source": "Bloomberg", "url": "https://ex.com/2",
             "published_at": "2025-07-10 08:00:00",
             "description": "New Sanofi treatment approved",
             "sentiment": 0.8, "source_api": "alpha_vantage"},
            {"ticker": "SAN.PA", "title": "Marche en hausse",
             "source": "BFM", "url": "https://ex.com/3",
             "published_at": "2025-07-10 14:00:00",
             "description": "Le CAC40 gagne 1%",
             "sentiment": None, "source_api": "rss_google_news_bourse"},
            {"ticker": "SAN.PA", "title": "Sanofi hors fenetre",
             "source": "Reuters", "url": "https://ex.com/4",
             "published_at": "2025-07-01 10:00:00",
             "description": "Trop ancien",
             "sentiment": None, "source_api": "gnews"},
        ])

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_match_trade_trouve_catalyseurs(self):
        """Match un trade et trouve les news dans la fenetre."""
        trade = {
            "id": 1, "nom_action": "SANOFI", "date_achat": "2025-07-10",
            "isin": "FR0000120578", "statut": "CLOTURE",
        }
        catalyseurs = self.matcher.match_trade(trade)
        # 3 news dans fenetre [07-07, 07-11], pas la news du 07-01
        assert len(catalyseurs) == 3

    def test_match_trade_scores_corrects(self):
        """Verifie les scores: J-1 avec match texte, J0 avec match, J0 sans match."""
        trade = {
            "id": 1, "nom_action": "SANOFI", "date_achat": "2025-07-10",
            "isin": "FR0000120578", "statut": "CLOTURE",
        }
        catalyseurs = self.matcher.match_trade(trade)
        # Trier par news_id pour un ordre previsible
        catalyseurs.sort(key=lambda c: c["news_id"])
        # News 1: 07-09, J-1, match texte "Sanofi" -> 0.8 + 0.2 = 1.0
        assert catalyseurs[0]["score_pertinence"] == 1.0
        assert catalyseurs[0]["distance_jours"] == -1
        assert catalyseurs[0]["match_texte"] == 1
        # News 2: 07-10, J0, match texte "Sanofi" -> 1.0 + 0.2 = 1.0 (cap)
        assert catalyseurs[1]["score_pertinence"] == 1.0
        assert catalyseurs[1]["distance_jours"] == 0
        # News 3: 07-10, J0, pas de match texte "Sanofi" -> 1.0
        assert catalyseurs[2]["score_pertinence"] == 1.0
        assert catalyseurs[2]["distance_jours"] == 0
        assert catalyseurs[2]["match_texte"] == 0

    def test_match_trade_ticker_inconnu(self):
        """Trade avec nom_action sans ticker connu retourne liste vide."""
        trade = {
            "id": 99, "nom_action": "INCONNU SA", "date_achat": "2025-07-10",
            "isin": "FRINCONNU", "statut": "CLOTURE",
        }
        catalyseurs = self.matcher.match_trade(trade)
        assert catalyseurs == []

    def test_match_trade_aucune_news(self):
        """Trade sans news dans la fenetre retourne liste vide."""
        trade = {
            "id": 1, "nom_action": "SANOFI", "date_achat": "2025-01-01",
            "isin": "FR0000120578", "statut": "CLOTURE",
        }
        catalyseurs = self.matcher.match_trade(trade)
        assert catalyseurs == []


class TestMatchAllTrades:
    """Tests du matching batch sur tous les trades."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.matcher = CatalystMatcher(self.db)
        # 2 trades
        self.db.insert_trades_batch([
            {"isin": "FR0000120578", "nom_action": "SANOFI",
             "date_achat": "2025-07-10", "date_vente": "2025-07-15",
             "prix_achat": 95.0, "prix_vente": 100.0, "quantite": 10,
             "rendement_brut_pct": 5.26, "rendement_net_pct": 5.0,
             "duree_jours": 5, "frais_totaux": 2.5, "statut": "CLOTURE"},
            {"isin": "FR0000120271", "nom_action": "AIR LIQUIDE",
             "date_achat": "2025-07-10", "date_vente": "2025-07-12",
             "prix_achat": 170.0, "prix_vente": 175.0, "quantite": 5,
             "rendement_brut_pct": 2.94, "rendement_net_pct": 2.5,
             "duree_jours": 2, "frais_totaux": 1.5, "statut": "CLOTURE"},
        ])
        # News pour SANOFI
        self.db.insert_news({
            "ticker": "SAN.PA", "title": "Sanofi news",
            "source": "Reuters", "url": "https://ex.com/san1",
            "published_at": "2025-07-10", "description": "Sanofi actu",
            "sentiment": 0.5, "source_api": "gnews",
        })
        # News pour AIR LIQUIDE
        self.db.insert_news({
            "ticker": "AI.PA", "title": "Air Liquide en hausse",
            "source": "BFM", "url": "https://ex.com/ai1",
            "published_at": "2025-07-09", "description": "Air Liquide annonce",
            "sentiment": None, "source_api": "rss_google_news_air_liquide",
        })

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_match_all_trades_resume(self):
        """Matche tous les trades et retourne un resume."""
        result = self.matcher.match_all_trades()
        assert result["total_trades"] == 2
        assert result["trades_avec_catalyseurs"] == 2
        assert result["total_associations"] == 2
        assert result["erreurs"] == 0

    def test_match_all_trades_peuple_table(self):
        """Apres match_all, la table trade_catalyseurs est peuplee."""
        self.matcher.match_all_trades()
        assert self.db.count_catalyseurs() == 2

    def test_match_all_trades_clear_avant(self):
        """match_all_trades vide la table avant de re-matcher."""
        self.matcher.match_all_trades()
        assert self.db.count_catalyseurs() == 2
        # Re-matcher ne duplique pas
        self.matcher.match_all_trades()
        assert self.db.count_catalyseurs() == 2


class TestGetStats:
    """Tests des statistiques."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.matcher = CatalystMatcher(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_stats_base_vide(self):
        """Stats sur une base sans catalyseurs."""
        stats = self.matcher.get_stats()
        assert stats["total_catalyseurs"] == 0
        assert stats["total_trades"] == 0
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_catalyst_matcher.py -v`
Expected: FAIL (catalyst_matcher module not found)

**Step 4: Implement CatalystMatcher**

Creer `src/analysis/catalyst_matcher.py`:

```python
"""Correlation entre trades et catalyseurs (news autour de la date d'achat)."""

from datetime import datetime, timedelta

from loguru import logger

from src.core.database import Database
from src.data_collection.ticker_mapper import TickerMapper, TickerNotFoundError


# Ponderations temporelles: distance_jours -> score de base
TEMPORAL_WEIGHTS = {
    0: 1.0,   # Jour d'achat
    -1: 0.8,  # Veille
    -2: 0.6,
    -3: 0.4,
    1: 0.7,   # Lendemain
}

TEXT_MATCH_BONUS = 0.2


class CatalystMatcher:
    """Associe les trades a leurs catalyseurs (news dans la fenetre temporelle)."""

    def __init__(self, db: Database, days_before: int = 3, days_after: int = 1):
        self.db = db
        self.days_before = days_before
        self.days_after = days_after
        self.ticker_mapper = TickerMapper()

    def match_trade(self, trade: dict) -> list[dict]:
        """Trouve les news catalyseurs pour un trade donne.

        Retourne une liste de dicts prets pour insert dans trade_catalyseurs.
        Retourne [] si ticker inconnu ou aucune news trouvee.
        """
        nom_action = trade["nom_action"]
        try:
            ticker = self.ticker_mapper.get_ticker(nom_action)
        except TickerNotFoundError:
            logger.warning(f"Ticker inconnu pour '{nom_action}', skip")
            return []

        date_achat = datetime.strptime(trade["date_achat"][:10], "%Y-%m-%d")
        date_start = (date_achat - timedelta(days=self.days_before)).strftime("%Y-%m-%d")
        date_end = (date_achat + timedelta(days=self.days_after)).strftime("%Y-%m-%d")

        news_list = self.db.get_news_in_window(ticker, date_start, date_end)

        catalyseurs = []
        for news in news_list:
            news_date = datetime.strptime(news["published_at"][:10], "%Y-%m-%d")
            distance = (news_date - date_achat).days
            match_texte = self._check_text_match(
                nom_action, news["title"], news.get("description")
            )
            score = self._compute_score(distance, match_texte)
            if score > 0:
                catalyseurs.append({
                    "trade_id": trade["id"],
                    "news_id": news["id"],
                    "score_pertinence": score,
                    "distance_jours": distance,
                    "match_texte": 1 if match_texte else 0,
                })

        return catalyseurs

    def match_all_trades(self) -> dict:
        """Matche tous les trades. Vide la table avant, puis re-peuple.

        Retourne un resume: {total_trades, trades_avec_catalyseurs,
        total_associations, erreurs}.
        """
        self.db.clear_catalyseurs()
        trades = self.db.get_all_trades()
        total_associations = 0
        trades_avec = 0
        erreurs = 0

        for trade in trades:
            try:
                catalyseurs = self.match_trade(trade)
                if catalyseurs:
                    self.db.insert_catalyseurs_batch(catalyseurs)
                    trades_avec += 1
                    total_associations += len(catalyseurs)
            except Exception as e:
                logger.error(f"Erreur matching trade {trade['id']} "
                             f"({trade['nom_action']}): {e}")
                erreurs += 1

        result = {
            "total_trades": len(trades),
            "trades_avec_catalyseurs": trades_avec,
            "total_associations": total_associations,
            "erreurs": erreurs,
        }
        logger.info(f"Matching termine: {result}")
        return result

    def get_stats(self) -> dict:
        """Statistiques globales sur les catalyseurs."""
        total_catalyseurs = self.db.count_catalyseurs()
        total_trades = self.db.count_trades()
        return {
            "total_catalyseurs": total_catalyseurs,
            "total_trades": total_trades,
        }

    def _compute_score(self, distance_jours: int, match_texte: bool) -> float:
        """Calcule le score de pertinence.

        Score = ponderation temporelle + bonus match texte (cap a 1.0).
        Retourne 0.0 si la distance est hors fenetre.
        """
        base = TEMPORAL_WEIGHTS.get(distance_jours, 0.0)
        if base == 0.0:
            return 0.0
        bonus = TEXT_MATCH_BONUS if match_texte else 0.0
        return min(base + bonus, 1.0)

    def _check_text_match(self, nom_action: str, title: str, description: str | None) -> bool:
        """Verifie si le nom de l'action apparait dans le titre ou la description."""
        clean_name = nom_action.lstrip("* ").strip().lower()
        title_lower = title.lower() if title else ""
        desc_lower = description.lower() if description else ""
        return clean_name in title_lower or clean_name in desc_lower
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_catalyst_matcher.py -v`
Expected: 18 PASS

**Step 6: Run all tests (regression)**

Run: `uv run pytest tests/ -v`
Expected: 87 PASS (69 existants + 18 nouveaux)

**Step 7: Commit**

```bash
git add src/analysis/__init__.py src/analysis/catalyst_matcher.py tests/test_catalyst_matcher.py
git commit -m "feat(analysis): add CatalystMatcher with scoring and text matching"
```

---

### Task 3: Script CLI match_catalysts.py

**Files:**
- Create: `scripts/match_catalysts.py`

**Step 1: Implement the script**

```python
"""Script CLI pour matcher les trades avec leurs catalyseurs.

Usage:
    uv run python scripts/match_catalysts.py           # Matcher tous les trades
    uv run python scripts/match_catalysts.py --stats    # Stats seulement
"""

import argparse
import sqlite3

from src.core.database import Database
from src.analysis.catalyst_matcher import CatalystMatcher


DB_PATH = "data/trades.db"


def print_stats(db: Database, matcher: CatalystMatcher):
    """Affiche les statistiques detaillees des catalyseurs."""
    stats = matcher.get_stats()

    if stats["total_catalyseurs"] == 0:
        print("Aucun catalyseur en base. Lancez d'abord le matching sans --stats.")
        return

    # Stats globales
    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row

    total_trades = stats["total_trades"]
    trades_avec = conn.execute(
        "SELECT COUNT(DISTINCT trade_id) FROM trade_catalyseurs"
    ).fetchone()[0]
    trades_sans = total_trades - trades_avec
    total_assoc = stats["total_catalyseurs"]
    score_moy = conn.execute(
        "SELECT AVG(score_pertinence) FROM trade_catalyseurs"
    ).fetchone()[0]

    print("=== CATALYST MATCHING ===")
    print(f"Trades analyses: {total_trades}")
    print(f"Trades avec catalyseurs: {trades_avec} ({100*trades_avec/total_trades:.1f}%)")
    print(f"Trades sans catalyseurs: {trades_sans} ({100*trades_sans/total_trades:.1f}%)")
    print(f"Total associations: {total_assoc}")
    print(f"Score moyen: {score_moy:.2f}")

    # Par source
    rows = conn.execute("""
        SELECT n.source_api, COUNT(*) as cnt
        FROM trade_catalyseurs tc
        JOIN news n ON tc.news_id = n.id
        GROUP BY n.source_api
        ORDER BY cnt DESC
    """).fetchall()
    print("\n=== PAR SOURCE ===")
    for r in rows:
        print(f"  {r['source_api']:30s}: {r['cnt']} associations")

    # Top 5 trades
    rows = conn.execute("""
        SELECT t.nom_action, t.date_achat, COUNT(*) as cnt,
               AVG(tc.score_pertinence) as score_moy
        FROM trade_catalyseurs tc
        JOIN trades_complets t ON tc.trade_id = t.id
        GROUP BY tc.trade_id
        ORDER BY cnt DESC
        LIMIT 5
    """).fetchall()
    print("\n=== TOP 5 TRADES (plus de catalyseurs) ===")
    for i, r in enumerate(rows, 1):
        print(f"  {i}. {r['nom_action']} (achat {r['date_achat'][:10]}): "
              f"{r['cnt']} catalyseurs, score moy {r['score_moy']:.2f}")

    # Gagnants vs perdants
    rows = conn.execute("""
        SELECT
            CASE WHEN t.rendement_brut_pct > 0 THEN 'gagnant' ELSE 'perdant' END as type,
            AVG(sub.cnt) as moy_catalyseurs,
            AVG(sub.score_moy) as moy_score
        FROM (
            SELECT tc.trade_id, COUNT(*) as cnt,
                   AVG(tc.score_pertinence) as score_moy
            FROM trade_catalyseurs tc
            GROUP BY tc.trade_id
        ) sub
        JOIN trades_complets t ON sub.trade_id = t.id
        WHERE t.statut = 'CLOTURE'
        GROUP BY type
    """).fetchall()
    print("\n=== GAGNANTS vs PERDANTS ===")
    for r in rows:
        print(f"  {r['type'].capitalize()}s: {r['moy_catalyseurs']:.1f} catalyseurs "
              f"en moyenne, score moy {r['moy_score']:.2f}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Matcher trades <-> catalyseurs")
    parser.add_argument("--stats", action="store_true", help="Afficher stats seulement")
    args = parser.parse_args()

    db = Database(DB_PATH)
    db.init_db()
    matcher = CatalystMatcher(db)

    if args.stats:
        print_stats(db, matcher)
        return

    print("Matching trades <-> catalyseurs...")
    result = matcher.match_all_trades()

    print(f"\n=== RESULTAT ===")
    print(f"Trades analyses: {result['total_trades']}")
    print(f"Trades avec catalyseurs: {result['trades_avec_catalyseurs']}")
    print(f"Total associations: {result['total_associations']}")
    print(f"Erreurs: {result['erreurs']}")

    print("\n--- Stats detaillees ---")
    print_stats(db, matcher)


if __name__ == "__main__":
    main()
```

**Step 2: Test the script manuellement**

Run: `uv run python scripts/match_catalysts.py`
Expected: Affiche le matching sur les 166 trades reels

Run: `uv run python scripts/match_catalysts.py --stats`
Expected: Affiche les stats sans re-matcher

**Step 3: Commit**

```bash
git add scripts/match_catalysts.py
git commit -m "feat(scripts): add match_catalysts.py CLI"
```

---

### Task 4: Mise a jour CLAUDE.md + commit final

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Ajouter la commande dans CLAUDE.md**

Dans la section `## Commandes`, ajouter:

```bash
uv run python scripts/match_catalysts.py           # Matcher trades <-> catalyseurs
uv run python scripts/match_catalysts.py --stats    # Stats catalyseurs seulement
```

**Step 2: Mettre a jour le statut de l'etape 3**

Dans la table `## Etapes d'implementation`, changer:

```
| 3 | TODO | Correlation historique (trades <-> catalyseurs) |
```

en:

```
| 3 | DONE | Correlation historique — catalyst_matcher, trade_catalyseurs |
```

**Step 3: Mettre a jour Current Project State**

Mettre a jour la ligne Tests avec le nouveau total (87 tests).

**Step 4: Run all tests one final time**

Run: `uv run pytest tests/ -v`
Expected: 87 PASS

**Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for etape 3 completion"
```
