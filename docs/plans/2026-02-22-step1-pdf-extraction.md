# Étape 1 — Extraction et Structuration des PDF : Plan d'Implémentation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transformer les ~200 avis d'exécution PDF Société Générale en une base SQLite structurée avec tables `executions` et `trades_complets`.

**Architecture:** Script one-shot (pas de temps réel). Pipeline : PDF brut -> extraction texte (pdfplumber) -> parsing regex -> validation -> SQLite. Un module de matching reconstruit les trades complets (achat -> vente) à partir des exécutions individuelles.

**Tech Stack:** Python 3.13, uv, pdfplumber, sqlite3 (stdlib), pandas, pytest

---

## Task 1: Installer uv et initialiser le projet

**Files:**
- Create: `pyproject.toml` (via uv init)
- Create: `.gitignore`
- Create: `.env.example`

**Step 1: Installer uv**

```bash
pip install uv
```

Vérifier: `uv --version` doit afficher une version.

**Step 2: Initialiser le projet avec uv**

```bash
cd C:/Users/skwar/Desktop/BOT/nicolas
uv init --name pea-scanner
```

Cela crée `pyproject.toml` et un virtualenv.

**Step 3: Ajouter les dépendances de l'étape 1**

```bash
uv add pdfplumber pandas loguru pyyaml python-dotenv
uv add --dev pytest
```

**Step 4: Créer le .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/

# Environment
.env

# Data
data/pdfs/
data/*.db

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Logs
logs/
```

**Step 5: Créer la structure de dossiers**

```bash
mkdir -p src/extraction src/core data/pdfs data/models config scripts tests logs
```

Créer les fichiers `__init__.py` dans chaque package :
- `src/__init__.py`
- `src/extraction/__init__.py`
- `src/core/__init__.py`

**Step 6: Créer .env.example**

```env
# PEA Scanner - Variables d'environnement
# Copier ce fichier en .env et remplir les valeurs

# Pas de clés API nécessaires pour l'étape 1
```

**Step 7: Initialiser git et commit**

```bash
git init
git add .
git commit -m "chore: init project with uv, dependencies step 1"
```

---

## Task 2: Créer la couche base de données (database.py)

**Files:**
- Create: `src/core/database.py`
- Test: `tests/test_database.py`

**Step 1: Écrire le test**

```python
# tests/test_database.py
import sqlite3
import os
import tempfile
from src.core.database import Database


def test_database_creates_tables():
    """Vérifie que init_db crée toutes les tables nécessaires."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = Database(db_path)
        db.init_db()

        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert "executions" in tables
        assert "trades_complets" in tables


def test_database_insert_execution():
    """Vérifie qu'on peut insérer une exécution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = Database(db_path)
        db.init_db()

        execution = {
            "date_execution": "2024-03-15",
            "heure_execution": "09:32:15",
            "sens": "ACHAT",
            "nom_action": "LVMH",
            "isin": "FR0000121014",
            "quantite": 50,
            "prix_unitaire": 842.30,
            "montant_brut": 42115.00,
            "commission": 4.95,
            "frais": 0.00,
            "montant_net": 42119.95,
            "fichier_source": "avis_001.pdf",
        }
        db.insert_execution(execution)

        results = db.get_all_executions()
        assert len(results) == 1
        assert results[0]["nom_action"] == "LVMH"
        assert results[0]["sens"] == "ACHAT"


def test_database_insert_trade_complet():
    """Vérifie qu'on peut insérer un trade complet."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = Database(db_path)
        db.init_db()

        trade = {
            "isin": "FR0000121014",
            "nom_action": "LVMH",
            "date_achat": "2024-03-15 09:32:15",
            "date_vente": "2024-03-20 14:15:30",
            "prix_achat": 842.30,
            "prix_vente": 878.10,
            "quantite": 50,
            "rendement_brut_pct": 4.25,
            "rendement_net_pct": 4.13,
            "duree_jours": 5.2,
            "frais_totaux": 9.90,
            "statut": "CLOTURE",
        }
        db.insert_trade_complet(trade)

        results = db.get_all_trades()
        assert len(results) == 1
        assert results[0]["nom_action"] == "LVMH"
        assert results[0]["statut"] == "CLOTURE"
```

**Step 2: Lancer le test pour vérifier qu'il échoue**

```bash
uv run pytest tests/test_database.py -v
```

Attendu: FAIL (ModuleNotFoundError: No module named 'src.core.database')

**Step 3: Implémenter database.py**

```python
# src/core/database.py
"""Couche d'accès à la base de données SQLite."""

import sqlite3
import os
from loguru import logger


class Database:
    """Gère la connexion et les opérations sur la base SQLite."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        # Créer le dossier parent si nécessaire
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        """Crée une connexion avec row_factory pour accès par nom de colonne."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def init_db(self):
        """Crée les tables si elles n'existent pas."""
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date_execution DATE NOT NULL,
                heure_execution TIME NOT NULL,
                sens TEXT NOT NULL CHECK (sens IN ('ACHAT', 'VENTE')),
                nom_action TEXT NOT NULL,
                isin TEXT,
                quantite INTEGER NOT NULL,
                prix_unitaire REAL NOT NULL,
                montant_brut REAL NOT NULL,
                commission REAL DEFAULT 0,
                frais REAL DEFAULT 0,
                montant_net REAL,
                fichier_source TEXT
            );

            CREATE TABLE IF NOT EXISTS trades_complets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                isin TEXT,
                nom_action TEXT NOT NULL,
                date_achat DATETIME NOT NULL,
                date_vente DATETIME,
                prix_achat REAL NOT NULL,
                prix_vente REAL,
                quantite INTEGER NOT NULL,
                rendement_brut_pct REAL,
                rendement_net_pct REAL,
                duree_jours REAL,
                frais_totaux REAL,
                statut TEXT DEFAULT 'OUVERT' CHECK (statut IN ('OUVERT', 'CLOTURE'))
            );

            CREATE INDEX IF NOT EXISTS idx_executions_date
                ON executions(date_execution);
            CREATE INDEX IF NOT EXISTS idx_executions_isin
                ON executions(isin);
            CREATE INDEX IF NOT EXISTS idx_trades_isin
                ON trades_complets(isin);
        """)
        conn.commit()
        conn.close()
        logger.info(f"Base de données initialisée: {self.db_path}")

    def insert_execution(self, execution: dict):
        """Insère une exécution (un PDF parsé) dans la base."""
        conn = self._connect()
        conn.execute("""
            INSERT INTO executions
                (date_execution, heure_execution, sens, nom_action, isin,
                 quantite, prix_unitaire, montant_brut, commission, frais,
                 montant_net, fichier_source)
            VALUES
                (:date_execution, :heure_execution, :sens, :nom_action, :isin,
                 :quantite, :prix_unitaire, :montant_brut, :commission, :frais,
                 :montant_net, :fichier_source)
        """, execution)
        conn.commit()
        conn.close()

    def insert_executions_batch(self, executions: list[dict]):
        """Insère plusieurs exécutions en une transaction."""
        conn = self._connect()
        conn.executemany("""
            INSERT INTO executions
                (date_execution, heure_execution, sens, nom_action, isin,
                 quantite, prix_unitaire, montant_brut, commission, frais,
                 montant_net, fichier_source)
            VALUES
                (:date_execution, :heure_execution, :sens, :nom_action, :isin,
                 :quantite, :prix_unitaire, :montant_brut, :commission, :frais,
                 :montant_net, :fichier_source)
        """, executions)
        conn.commit()
        conn.close()
        logger.info(f"{len(executions)} exécutions insérées en batch")

    def insert_trade_complet(self, trade: dict):
        """Insère un trade complet (achat->vente reconstitué)."""
        conn = self._connect()
        conn.execute("""
            INSERT INTO trades_complets
                (isin, nom_action, date_achat, date_vente, prix_achat,
                 prix_vente, quantite, rendement_brut_pct, rendement_net_pct,
                 duree_jours, frais_totaux, statut)
            VALUES
                (:isin, :nom_action, :date_achat, :date_vente, :prix_achat,
                 :prix_vente, :quantite, :rendement_brut_pct, :rendement_net_pct,
                 :duree_jours, :frais_totaux, :statut)
        """, trade)
        conn.commit()
        conn.close()

    def insert_trades_batch(self, trades: list[dict]):
        """Insère plusieurs trades complets en une transaction."""
        conn = self._connect()
        conn.executemany("""
            INSERT INTO trades_complets
                (isin, nom_action, date_achat, date_vente, prix_achat,
                 prix_vente, quantite, rendement_brut_pct, rendement_net_pct,
                 duree_jours, frais_totaux, statut)
            VALUES
                (:isin, :nom_action, :date_achat, :date_vente, :prix_achat,
                 :prix_vente, :quantite, :rendement_brut_pct, :rendement_net_pct,
                 :duree_jours, :frais_totaux, :statut)
        """, trades)
        conn.commit()
        conn.close()
        logger.info(f"{len(trades)} trades complets insérés en batch")

    def get_all_executions(self) -> list[dict]:
        """Récupère toutes les exécutions."""
        conn = self._connect()
        rows = conn.execute("SELECT * FROM executions ORDER BY date_execution, heure_execution").fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_executions_by_isin(self, isin: str) -> list[dict]:
        """Récupère les exécutions pour un ISIN donné, triées chronologiquement."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM executions WHERE isin = ? ORDER BY date_execution, heure_execution",
            (isin,)
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_all_trades(self) -> list[dict]:
        """Récupère tous les trades complets."""
        conn = self._connect()
        rows = conn.execute("SELECT * FROM trades_complets ORDER BY date_achat").fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_distinct_isins(self) -> list[str]:
        """Récupère la liste des ISIN distincts dans les exécutions."""
        conn = self._connect()
        rows = conn.execute("SELECT DISTINCT isin FROM executions WHERE isin IS NOT NULL").fetchall()
        conn.close()
        return [row["isin"] for row in rows]

    def count_executions(self) -> int:
        """Compte le nombre total d'exécutions."""
        conn = self._connect()
        count = conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0]
        conn.close()
        return count

    def count_trades(self) -> int:
        """Compte le nombre total de trades complets."""
        conn = self._connect()
        count = conn.execute("SELECT COUNT(*) FROM trades_complets").fetchone()[0]
        conn.close()
        return count
```

**Step 4: Lancer les tests**

```bash
uv run pytest tests/test_database.py -v
```

Attendu: 3 tests PASS

**Step 5: Commit**

```bash
git add src/core/database.py tests/test_database.py
git commit -m "feat: add database layer with executions and trades_complets tables"
```

---

## Task 3: Créer le parser PDF (pdf_parser.py)

**Files:**
- Create: `src/extraction/pdf_parser.py`
- Create: `tests/test_pdf_parser.py`
- Create: `tests/fixtures/` (PDF de test)

**Prérequis:** L'utilisateur doit fournir 2-3 PDF d'exemple dans `data/pdfs/`. On analysera leur structure exacte avant de coder les regex.

**Step 1: Analyser la structure d'un PDF SG**

Script exploratoire pour voir le texte brut extrait :

```python
# scripts/explore_pdf.py
"""Script exploratoire : affiche le texte brut extrait d'un PDF SG."""

import sys
import pdfplumber

def explore_pdf(pdf_path: str):
    """Extrait et affiche le texte de chaque page d'un PDF."""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            print(f"=== PAGE {i+1} ===")
            text = page.extract_text()
            print(text)
            print()

            # Afficher aussi les tableaux détectés
            tables = page.extract_tables()
            if tables:
                print(f"--- TABLEAUX PAGE {i+1} ---")
                for j, table in enumerate(tables):
                    print(f"Table {j+1}:")
                    for row in table:
                        print(row)
                    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/explore_pdf.py <chemin_pdf>")
        sys.exit(1)
    explore_pdf(sys.argv[1])
```

Lancer sur un PDF d'exemple :

```bash
uv run python scripts/explore_pdf.py data/pdfs/exemple.pdf
```

**Step 2: Coder les regex basées sur la structure observée**

> NOTE: Les regex exactes seront adaptées après l'analyse du Step 1.
> Le code ci-dessous est un template qui sera ajusté.

```python
# src/extraction/pdf_parser.py
"""Parser pour les avis d'exécution PDF Société Générale."""

import re
import os
import pdfplumber
from loguru import logger


class PDFParseError(Exception):
    """Erreur lors du parsing d'un PDF."""
    pass


class SGPDFParser:
    """Parse les avis d'exécution Société Générale au format PDF."""

    def parse_pdf(self, pdf_path: str) -> dict:
        """
        Extrait les données d'un avis d'exécution PDF SG.

        Args:
            pdf_path: Chemin vers le fichier PDF

        Returns:
            Dict avec les champs: date_execution, heure_execution, sens,
            nom_action, isin, quantite, prix_unitaire, montant_brut,
            commission, frais, montant_net, fichier_source

        Raises:
            PDFParseError: Si le PDF ne peut pas être parsé correctement
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
        """
        Parse le texte extrait pour en extraire les champs.

        NOTE: Les regex ci-dessous sont un TEMPLATE.
        Elles seront ajustées après analyse des vrais PDF SG.
        """
        data = {}

        # --- Les patterns ci-dessous seront adaptés au format réel ---

        # Date d'exécution (format attendu: JJ/MM/AAAA ou AAAA-MM-JJ)
        date_match = re.search(
            r"(?:Date\s+(?:d')?ex[ée]cution|Date\s*:)\s*(\d{2}/\d{2}/\d{4})",
            text, re.IGNORECASE
        )
        if date_match:
            # Convertir JJ/MM/AAAA en AAAA-MM-JJ
            d, m, y = date_match.group(1).split("/")
            data["date_execution"] = f"{y}-{m}-{d}"

        # Heure d'exécution
        heure_match = re.search(
            r"(?:Heure|Horodatage)\s*:?\s*(\d{2}:\d{2}(?::\d{2})?)",
            text, re.IGNORECASE
        )
        if heure_match:
            data["heure_execution"] = heure_match.group(1)
        else:
            data["heure_execution"] = "00:00:00"

        # Sens (ACHAT/VENTE)
        sens_match = re.search(r"\b(ACHAT|VENTE)\b", text, re.IGNORECASE)
        if sens_match:
            data["sens"] = sens_match.group(1).upper()

        # ISIN
        isin_match = re.search(r"([A-Z]{2}\d{10})", text)
        if isin_match:
            data["isin"] = isin_match.group(1)

        # Nom de l'action (sera affiné selon le format SG)
        nom_match = re.search(
            r"(?:Libell[ée]|D[ée]signation|Valeur)\s*:?\s*(.+?)(?:\n|$)",
            text, re.IGNORECASE
        )
        if nom_match:
            data["nom_action"] = nom_match.group(1).strip()

        # Quantité
        qte_match = re.search(
            r"(?:Quantit[ée]|Nombre|Qté)\s*:?\s*(\d+)",
            text, re.IGNORECASE
        )
        if qte_match:
            data["quantite"] = int(qte_match.group(1))

        # Prix unitaire (gère les formats 842,30 et 842.30)
        prix_match = re.search(
            r"(?:Prix\s+(?:unitaire|d'ex[ée]cution)|Cours)\s*:?\s*([\d\s]+[.,]\d{2,})",
            text, re.IGNORECASE
        )
        if prix_match:
            prix_str = prix_match.group(1).replace(" ", "").replace(",", ".")
            data["prix_unitaire"] = float(prix_str)

        # Montant brut
        brut_match = re.search(
            r"(?:Montant\s+brut|Montant\s+(?:de\s+)?(?:l')?op[ée]ration)\s*:?\s*([\d\s]+[.,]\d{2})",
            text, re.IGNORECASE
        )
        if brut_match:
            brut_str = brut_match.group(1).replace(" ", "").replace(",", ".")
            data["montant_brut"] = float(brut_str)

        # Commission / Courtage
        comm_match = re.search(
            r"(?:Commission|Courtage|Frais\s+de\s+courtage)\s*:?\s*([\d\s]+[.,]\d{2})",
            text, re.IGNORECASE
        )
        if comm_match:
            comm_str = comm_match.group(1).replace(" ", "").replace(",", ".")
            data["commission"] = float(comm_str)
        else:
            data["commission"] = 0.0

        # Frais divers
        frais_match = re.search(
            r"(?:Frais\s+divers|Autres\s+frais)\s*:?\s*([\d\s]+[.,]\d{2})",
            text, re.IGNORECASE
        )
        if frais_match:
            frais_str = frais_match.group(1).replace(" ", "").replace(",", ".")
            data["frais"] = float(frais_str)
        else:
            data["frais"] = 0.0

        # Montant net
        net_match = re.search(
            r"(?:Montant\s+net|Net\s*:?\s*)\s*:?\s*([\d\s]+[.,]\d{2})",
            text, re.IGNORECASE
        )
        if net_match:
            net_str = net_match.group(1).replace(" ", "").replace(",", ".")
            data["montant_net"] = float(net_str)

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
        if "montant_brut" in data and "quantite" in data and "prix_unitaire" in data:
            expected = data["quantite"] * data["prix_unitaire"]
            actual = data["montant_brut"]
            if abs(actual - expected) / expected > 0.02:  # tolérance 2%
                logger.warning(
                    f"Incohérence montant brut dans {data.get('fichier_source')}: "
                    f"attendu {expected:.2f}, trouvé {actual:.2f}"
                )

        # Calculer montant_net si absent
        if "montant_net" not in data and "montant_brut" in data:
            commission = data.get("commission", 0)
            frais = data.get("frais", 0)
            if data.get("sens") == "ACHAT":
                data["montant_net"] = data["montant_brut"] + commission + frais
            else:
                data["montant_net"] = data["montant_brut"] - commission - frais

    def parse_directory(self, directory: str) -> list[dict]:
        """
        Parse tous les PDF d'un dossier.

        Returns:
            Liste de dicts (un par PDF parsé avec succès)
            Les erreurs sont loguées mais n'arrêtent pas le batch.
        """
        results = []
        errors = []
        pdf_files = sorted([f for f in os.listdir(directory) if f.lower().endswith(".pdf")])

        if not pdf_files:
            logger.warning(f"Aucun PDF trouvé dans {directory}")
            return results

        logger.info(f"Parsing de {len(pdf_files)} PDF dans {directory}")

        for filename in pdf_files:
            filepath = os.path.join(directory, filename)
            try:
                data = self.parse_pdf(filepath)
                results.append(data)
                logger.debug(f"OK: {filename} -> {data['sens']} {data['nom_action']}")
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
```

**Step 2b: Écrire les tests unitaires**

```python
# tests/test_pdf_parser.py
"""Tests pour le parser PDF SG."""

import os
import tempfile
import pytest
from src.extraction.pdf_parser import SGPDFParser, PDFParseError


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
            "date_execution": "2024-03-15",
            "heure_execution": "09:32:15",
            "sens": "ACHAT",
            "nom_action": "LVMH",
            "isin": "FR0000121014",
            "quantite": 50,
            "prix_unitaire": 842.30,
            "montant_brut": 42115.00,
            "commission": 4.95,
            "frais": 0.00,
            "fichier_source": "test.pdf",
        }
        # Ne doit pas lever d'exception
        self.parser._validate(data)
        # montant_net doit être calculé automatiquement
        assert "montant_net" in data

    def test_validate_calculates_net_for_achat(self):
        """Pour un ACHAT, montant_net = brut + commission + frais."""
        data = {
            "date_execution": "2024-03-15",
            "sens": "ACHAT",
            "nom_action": "LVMH",
            "quantite": 50,
            "prix_unitaire": 842.30,
            "montant_brut": 42115.00,
            "commission": 4.95,
            "frais": 1.00,
            "fichier_source": "test.pdf",
        }
        self.parser._validate(data)
        assert data["montant_net"] == 42115.00 + 4.95 + 1.00

    def test_validate_calculates_net_for_vente(self):
        """Pour une VENTE, montant_net = brut - commission - frais."""
        data = {
            "date_execution": "2024-03-20",
            "sens": "VENTE",
            "nom_action": "LVMH",
            "quantite": 50,
            "prix_unitaire": 878.10,
            "montant_brut": 43905.00,
            "commission": 4.95,
            "frais": 0.00,
            "fichier_source": "test.pdf",
        }
        self.parser._validate(data)
        assert data["montant_net"] == 43905.00 - 4.95

    def test_parse_nonexistent_file(self):
        """Un fichier inexistant doit lever FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            self.parser.parse_pdf("/chemin/inexistant.pdf")

    def test_parse_directory_empty(self):
        """Un dossier vide retourne une liste vide."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = self.parser.parse_directory(tmpdir)
            assert results == []
```

**Step 3: Lancer les tests**

```bash
uv run pytest tests/test_pdf_parser.py -v
```

Attendu: Tous les tests passent.

**Step 4: Commit**

```bash
git add src/extraction/pdf_parser.py tests/test_pdf_parser.py scripts/explore_pdf.py
git commit -m "feat: add PDF parser for SG execution notices"
```

---

## Task 4: Créer le trade matcher (trade_matcher.py)

**Files:**
- Create: `src/extraction/trade_matcher.py`
- Create: `tests/test_trade_matcher.py`

**Step 1: Écrire les tests**

```python
# tests/test_trade_matcher.py
"""Tests pour le matching achat/vente et reconstruction des trades."""

from src.extraction.trade_matcher import TradeMatcher


def test_simple_buy_sell():
    """Un achat suivi d'une vente = 1 trade complet."""
    executions = [
        {
            "date_execution": "2024-03-15",
            "heure_execution": "09:32:15",
            "sens": "ACHAT",
            "nom_action": "LVMH",
            "isin": "FR0000121014",
            "quantite": 50,
            "prix_unitaire": 842.30,
            "montant_brut": 42115.00,
            "commission": 4.95,
            "frais": 0.00,
            "montant_net": 42119.95,
        },
        {
            "date_execution": "2024-03-20",
            "heure_execution": "14:15:30",
            "sens": "VENTE",
            "nom_action": "LVMH",
            "isin": "FR0000121014",
            "quantite": 50,
            "prix_unitaire": 878.10,
            "montant_brut": 43905.00,
            "commission": 4.95,
            "frais": 0.00,
            "montant_net": 43900.05,
        },
    ]

    matcher = TradeMatcher()
    trades = matcher.match_trades(executions)

    assert len(trades) == 1
    trade = trades[0]
    assert trade["nom_action"] == "LVMH"
    assert trade["statut"] == "CLOTURE"
    assert trade["quantite"] == 50
    assert trade["prix_achat"] == 842.30
    assert trade["prix_vente"] == 878.10
    assert trade["duree_jours"] == 5.0  # 15 mars -> 20 mars


def test_buy_without_sell():
    """Un achat sans vente = trade ouvert."""
    executions = [
        {
            "date_execution": "2024-03-15",
            "heure_execution": "09:32:15",
            "sens": "ACHAT",
            "nom_action": "TOTALENERGIES",
            "isin": "FR0000120271",
            "quantite": 100,
            "prix_unitaire": 62.50,
            "montant_brut": 6250.00,
            "commission": 4.95,
            "frais": 0.00,
            "montant_net": 6254.95,
        },
    ]

    matcher = TradeMatcher()
    trades = matcher.match_trades(executions)

    assert len(trades) == 1
    assert trades[0]["statut"] == "OUVERT"
    assert trades[0]["prix_vente"] is None


def test_partial_sell():
    """Achat 100 puis vente 50 = 1 trade clôturé (50) + 1 trade ouvert (50)."""
    executions = [
        {
            "date_execution": "2024-03-10",
            "heure_execution": "09:00:00",
            "sens": "ACHAT",
            "nom_action": "BNP",
            "isin": "FR0000131104",
            "quantite": 100,
            "prix_unitaire": 60.00,
            "montant_brut": 6000.00,
            "commission": 4.95,
            "frais": 0.00,
            "montant_net": 6004.95,
        },
        {
            "date_execution": "2024-03-15",
            "heure_execution": "10:00:00",
            "sens": "VENTE",
            "nom_action": "BNP",
            "isin": "FR0000131104",
            "quantite": 50,
            "prix_unitaire": 63.00,
            "montant_brut": 3150.00,
            "commission": 4.95,
            "frais": 0.00,
            "montant_net": 3145.05,
        },
    ]

    matcher = TradeMatcher()
    trades = matcher.match_trades(executions)

    assert len(trades) == 2
    closed = [t for t in trades if t["statut"] == "CLOTURE"]
    opened = [t for t in trades if t["statut"] == "OUVERT"]
    assert len(closed) == 1
    assert len(opened) == 1
    assert closed[0]["quantite"] == 50
    assert opened[0]["quantite"] == 50


def test_rendement_calculation():
    """Le rendement brut % doit être correctement calculé."""
    executions = [
        {
            "date_execution": "2024-03-15",
            "heure_execution": "09:00:00",
            "sens": "ACHAT",
            "nom_action": "TEST",
            "isin": "FR0000000001",
            "quantite": 10,
            "prix_unitaire": 100.00,
            "montant_brut": 1000.00,
            "commission": 5.00,
            "frais": 0.00,
            "montant_net": 1005.00,
        },
        {
            "date_execution": "2024-03-20",
            "heure_execution": "09:00:00",
            "sens": "VENTE",
            "nom_action": "TEST",
            "isin": "FR0000000001",
            "quantite": 10,
            "prix_unitaire": 105.00,
            "montant_brut": 1050.00,
            "commission": 5.00,
            "frais": 0.00,
            "montant_net": 1045.00,
        },
    ]

    matcher = TradeMatcher()
    trades = matcher.match_trades(executions)

    trade = trades[0]
    assert trade["rendement_brut_pct"] == 5.0  # (105-100)/100 * 100
    assert trade["frais_totaux"] == 10.0  # 5 achat + 5 vente


def test_multiple_actions():
    """Des exécutions mélangées sur plusieurs actions sont bien séparées."""
    executions = [
        {
            "date_execution": "2024-03-10", "heure_execution": "09:00:00",
            "sens": "ACHAT", "nom_action": "LVMH", "isin": "FR0000121014",
            "quantite": 10, "prix_unitaire": 800.0, "montant_brut": 8000.0,
            "commission": 5.0, "frais": 0.0, "montant_net": 8005.0,
        },
        {
            "date_execution": "2024-03-11", "heure_execution": "09:00:00",
            "sens": "ACHAT", "nom_action": "BNP", "isin": "FR0000131104",
            "quantite": 50, "prix_unitaire": 60.0, "montant_brut": 3000.0,
            "commission": 5.0, "frais": 0.0, "montant_net": 3005.0,
        },
        {
            "date_execution": "2024-03-15", "heure_execution": "09:00:00",
            "sens": "VENTE", "nom_action": "LVMH", "isin": "FR0000121014",
            "quantite": 10, "prix_unitaire": 840.0, "montant_brut": 8400.0,
            "commission": 5.0, "frais": 0.0, "montant_net": 8395.0,
        },
    ]

    matcher = TradeMatcher()
    trades = matcher.match_trades(executions)

    assert len(trades) == 2
    lvmh = [t for t in trades if t["nom_action"] == "LVMH"][0]
    bnp = [t for t in trades if t["nom_action"] == "BNP"][0]
    assert lvmh["statut"] == "CLOTURE"
    assert bnp["statut"] == "OUVERT"
```

**Step 2: Lancer les tests pour vérifier qu'ils échouent**

```bash
uv run pytest tests/test_trade_matcher.py -v
```

Attendu: FAIL (ModuleNotFoundError)

**Step 3: Implémenter trade_matcher.py**

```python
# src/extraction/trade_matcher.py
"""Reconstruction des trades complets à partir des exécutions individuelles.

Logique FIFO : les premières actions achetées sont les premières vendues.
"""

from datetime import datetime
from loguru import logger


class TradeMatcher:
    """Reconstruit les trades complets (achat->vente) à partir des exécutions."""

    def match_trades(self, executions: list[dict]) -> list[dict]:
        """
        À partir d'une liste d'exécutions (potentiellement multi-actions),
        reconstruit les trades complets.

        Approche FIFO par ISIN : les achats sont consommés dans l'ordre chronologique.

        Args:
            executions: Liste de dicts avec les champs d'une exécution

        Returns:
            Liste de trades complets (dicts compatibles avec table trades_complets)
        """
        # Regrouper par ISIN
        by_isin = {}
        for ex in executions:
            isin = ex.get("isin", ex.get("nom_action", "UNKNOWN"))
            if isin not in by_isin:
                by_isin[isin] = []
            by_isin[isin].append(ex)

        all_trades = []
        for isin, execs in by_isin.items():
            trades = self._match_single_isin(execs)
            all_trades.extend(trades)

        logger.info(
            f"Matching terminé: {len(all_trades)} trades reconstitués "
            f"({sum(1 for t in all_trades if t['statut'] == 'CLOTURE')} clôturés, "
            f"{sum(1 for t in all_trades if t['statut'] == 'OUVERT')} ouverts)"
        )

        return all_trades

    def _match_single_isin(self, executions: list[dict]) -> list[dict]:
        """Reconstruit les trades pour un seul ISIN (méthode FIFO)."""
        # Trier par date+heure
        sorted_execs = sorted(
            executions,
            key=lambda e: f"{e['date_execution']} {e['heure_execution']}"
        )

        # File FIFO des achats non encore vendus
        # Chaque élément = (exec_dict, quantite_restante)
        buy_queue: list[tuple[dict, int]] = []
        trades = []

        for ex in sorted_execs:
            if ex["sens"] == "ACHAT":
                buy_queue.append((ex, ex["quantite"]))
            elif ex["sens"] == "VENTE":
                remaining_to_sell = ex["quantite"]

                while remaining_to_sell > 0 and buy_queue:
                    buy_exec, buy_remaining = buy_queue[0]
                    matched_qty = min(remaining_to_sell, buy_remaining)

                    # Créer le trade pour cette portion
                    trade = self._create_trade(
                        buy_exec=buy_exec,
                        sell_exec=ex,
                        quantite=matched_qty,
                    )
                    trades.append(trade)

                    remaining_to_sell -= matched_qty
                    buy_remaining -= matched_qty

                    if buy_remaining == 0:
                        buy_queue.pop(0)
                    else:
                        buy_queue[0] = (buy_exec, buy_remaining)

                if remaining_to_sell > 0:
                    logger.warning(
                        f"Vente de {remaining_to_sell} {ex['nom_action']} "
                        f"sans achat correspondant le {ex['date_execution']}"
                    )

        # Les achats restants dans la queue = trades ouverts
        for buy_exec, remaining_qty in buy_queue:
            trade = self._create_trade(
                buy_exec=buy_exec,
                sell_exec=None,
                quantite=remaining_qty,
            )
            trades.append(trade)

        return trades

    def _create_trade(
        self,
        buy_exec: dict,
        sell_exec: dict | None,
        quantite: int,
    ) -> dict:
        """Crée un dict de trade complet."""
        trade = {
            "isin": buy_exec.get("isin"),
            "nom_action": buy_exec["nom_action"],
            "date_achat": f"{buy_exec['date_execution']} {buy_exec['heure_execution']}",
            "prix_achat": buy_exec["prix_unitaire"],
            "quantite": quantite,
        }

        if sell_exec:
            trade["date_vente"] = f"{sell_exec['date_execution']} {sell_exec['heure_execution']}"
            trade["prix_vente"] = sell_exec["prix_unitaire"]
            trade["statut"] = "CLOTURE"

            # Rendement brut %
            trade["rendement_brut_pct"] = round(
                (sell_exec["prix_unitaire"] - buy_exec["prix_unitaire"])
                / buy_exec["prix_unitaire"] * 100,
                2
            )

            # Frais totaux (proportionnels à la quantité matchée)
            buy_commission = buy_exec.get("commission", 0) * (quantite / buy_exec["quantite"])
            buy_frais = buy_exec.get("frais", 0) * (quantite / buy_exec["quantite"])
            sell_commission = sell_exec.get("commission", 0) * (quantite / sell_exec["quantite"])
            sell_frais = sell_exec.get("frais", 0) * (quantite / sell_exec["quantite"])
            trade["frais_totaux"] = round(
                buy_commission + buy_frais + sell_commission + sell_frais, 2
            )

            # Rendement net %
            montant_achat = buy_exec["prix_unitaire"] * quantite
            montant_vente = sell_exec["prix_unitaire"] * quantite
            profit_net = montant_vente - montant_achat - trade["frais_totaux"]
            trade["rendement_net_pct"] = round(profit_net / montant_achat * 100, 2)

            # Durée en jours
            dt_achat = datetime.strptime(trade["date_achat"], "%Y-%m-%d %H:%M:%S")
            dt_vente = datetime.strptime(trade["date_vente"], "%Y-%m-%d %H:%M:%S")
            trade["duree_jours"] = round((dt_vente - dt_achat).total_seconds() / 86400, 1)
        else:
            trade["date_vente"] = None
            trade["prix_vente"] = None
            trade["statut"] = "OUVERT"
            trade["rendement_brut_pct"] = None
            trade["rendement_net_pct"] = None
            trade["duree_jours"] = None
            trade["frais_totaux"] = round(
                buy_exec.get("commission", 0) * (quantite / buy_exec["quantite"])
                + buy_exec.get("frais", 0) * (quantite / buy_exec["quantite"]),
                2
            )

        return trade
```

**Step 4: Lancer les tests**

```bash
uv run pytest tests/test_trade_matcher.py -v
```

Attendu: Tous les tests PASS

**Step 5: Commit**

```bash
git add src/extraction/trade_matcher.py tests/test_trade_matcher.py
git commit -m "feat: add trade matcher to reconstruct buy/sell trades (FIFO)"
```

---

## Task 5: Script d'import batch + script principal

**Files:**
- Create: `scripts/import_pdfs.py`
- Create: `scripts/init_db.py`

**Step 1: Créer init_db.py**

```python
# scripts/init_db.py
"""Initialise la base de données SQLite avec les tables nécessaires."""

import os
import sys

# Ajouter le répertoire racine au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.database import Database
from loguru import logger


def main():
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trades.db")
    db = Database(db_path)
    db.init_db()
    logger.info(f"Base de données créée/vérifiée: {db_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Créer import_pdfs.py**

```python
# scripts/import_pdfs.py
"""Import batch des PDF d'avis d'exécution SG dans la base SQLite."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.database import Database
from src.extraction.pdf_parser import SGPDFParser
from src.extraction.trade_matcher import TradeMatcher
from loguru import logger


def main():
    # Chemins
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_dir = os.path.join(base_dir, "data", "pdfs")
    db_path = os.path.join(base_dir, "data", "trades.db")

    if not os.path.exists(pdf_dir):
        logger.error(f"Dossier PDF introuvable: {pdf_dir}")
        logger.info("Placez vos PDF dans data/pdfs/ et relancez.")
        sys.exit(1)

    # 1. Init DB
    db = Database(db_path)
    db.init_db()

    # 2. Parser tous les PDF
    parser = SGPDFParser()
    executions = parser.parse_directory(pdf_dir)

    if not executions:
        logger.warning("Aucune exécution extraite. Vérifiez vos PDF.")
        sys.exit(1)

    # 3. Insérer les exécutions en base
    db.insert_executions_batch(executions)
    logger.info(f"{len(executions)} exécutions insérées en base")

    # 4. Reconstruire les trades complets
    matcher = TradeMatcher()
    trades = matcher.match_trades(executions)

    # 5. Insérer les trades
    db.insert_trades_batch(trades)
    logger.info(f"{len(trades)} trades reconstitués et insérés en base")

    # 6. Résumé
    logger.info("=== RÉSUMÉ ===")
    logger.info(f"PDF parsés: {len(executions)}")
    logger.info(f"Trades reconstitués: {len(trades)}")
    logger.info(f"  - Clôturés: {sum(1 for t in trades if t['statut'] == 'CLOTURE')}")
    logger.info(f"  - Ouverts: {sum(1 for t in trades if t['statut'] == 'OUVERT')}")

    closed_trades = [t for t in trades if t["statut"] == "CLOTURE"]
    if closed_trades:
        avg_return = sum(t["rendement_brut_pct"] for t in closed_trades) / len(closed_trades)
        avg_duration = sum(t["duree_jours"] for t in closed_trades) / len(closed_trades)
        winners = sum(1 for t in closed_trades if t["rendement_brut_pct"] > 0)
        logger.info(f"  - Rendement moyen: {avg_return:.2f}%")
        logger.info(f"  - Durée moyenne: {avg_duration:.1f} jours")
        logger.info(f"  - Taux de réussite: {winners}/{len(closed_trades)} ({winners/len(closed_trades)*100:.0f}%)")


if __name__ == "__main__":
    main()
```

**Step 3: Tester init_db**

```bash
uv run python scripts/init_db.py
```

Attendu: "Base de données créée/vérifiée: .../data/trades.db"

**Step 4: Commit**

```bash
git add scripts/init_db.py scripts/import_pdfs.py
git commit -m "feat: add init_db and import_pdfs batch scripts"
```

---

## Task 6: Tester sur les vrais PDF et ajuster

**Step 1: L'utilisateur place 2-3 PDF dans data/pdfs/**

**Step 2: Explorer un PDF**

```bash
uv run python scripts/explore_pdf.py data/pdfs/<premier_pdf>.pdf
```

**Step 3: Analyser la sortie et ajuster les regex dans pdf_parser.py**

Comparer le texte brut extrait avec les regex du parser. Ajuster les patterns pour matcher exactement le format SG réel.

**Step 4: Tester le parsing sur les exemples**

```bash
uv run python -c "
from src.extraction.pdf_parser import SGPDFParser
parser = SGPDFParser()
result = parser.parse_pdf('data/pdfs/<premier_pdf>.pdf')
for k, v in result.items():
    print(f'{k}: {v}')
"
```

**Step 5: Ajuster les regex jusqu'à obtenir un parsing correct**

**Step 6: Lancer l'import complet**

```bash
uv run python scripts/import_pdfs.py
```

**Step 7: Vérifier manuellement 10-15 trades**

```bash
uv run python -c "
from src.core.database import Database
db = Database('data/trades.db')
trades = db.get_all_trades()
for t in trades[:15]:
    print(f\"{t['nom_action']:15s} {t['date_achat'][:10]} -> {str(t.get('date_vente', 'OUVERT'))[:10]}  {t.get('rendement_brut_pct', '?'):>6}%  {t['statut']}\")
"
```

**Step 8: Lancer tous les tests**

```bash
uv run pytest tests/ -v
```

Attendu: Tous les tests PASS

**Step 9: Commit final**

```bash
git add -A
git commit -m "feat: complete step 1 - PDF extraction and trade reconstruction"
```

---

## Résumé de l'Étape 1

| Task | Description | Fichiers créés |
|------|-------------|----------------|
| 1 | Init projet (uv, git, structure) | pyproject.toml, .gitignore, structure dossiers |
| 2 | Couche BDD | src/core/database.py, tests/test_database.py |
| 3 | Parser PDF | src/extraction/pdf_parser.py, tests/test_pdf_parser.py, scripts/explore_pdf.py |
| 4 | Trade matcher | src/extraction/trade_matcher.py, tests/test_trade_matcher.py |
| 5 | Scripts batch | scripts/init_db.py, scripts/import_pdfs.py |
| 6 | Test sur vrais PDF | Ajustements regex, validation manuelle |

**Critères de succès de l'étape 1:**
- Les ~200 PDF sont parsés sans erreur (ou avec un taux d'erreur < 5%)
- Les trades reconstitués sont vérifiés manuellement (10-15 trades)
- Tous les tests passent
- La base SQLite contient les tables `executions` et `trades_complets` correctement remplies
