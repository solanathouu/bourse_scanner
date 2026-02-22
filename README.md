# PEA Scanner Bot

Bot de veille boursiere pour le PEA. Analyse les donnees de marche, correle avec les actualites, applique un modele predictif ML, et envoie des alertes Telegram.

## Installation

```bash
# Prerequis: Python 3.13+, uv
pip install uv

# Cloner et installer
cd pea-scanner
uv sync
```

## Utilisation rapide

### 1. Importer les PDF d'avis d'execution

Placer les PDF Societe Generale dans `data/pdfs/`, puis :

```bash
uv run python scripts/import_pdfs.py
```

Resultat: base SQLite `data/trades.db` avec les tables `executions` et `trades_complets`.

### 2. Explorer un PDF (debug)

```bash
uv run python scripts/explore_pdf.py data/pdfs/mon_avis.pdf
```

### 3. Lancer les tests

```bash
uv run pytest tests/ -v
```

## Architecture

```
src/
├── core/               # Fondation: BDD, config, logging
├── extraction/         # Module 1: PDF -> SQLite
├── data_collection/    # Module 2: Prix, news, fondamentaux
├── analysis/           # Module 3: Features, sentiment, catalyseurs
├── model/              # Module 3: ML training + scoring
└── alerts/             # Module 4: Telegram
```

## Statut

- [x] Etape 1 — Extraction PDF (214 PDF, 166 trades, 89% win rate)
- [ ] Etape 2 — Collecte donnees marche + news
- [ ] Etape 3 — Correlation historique
- [ ] Etape 4 — Feature engineering + ML
- [ ] Etape 5 — Pipeline temps reel + Telegram
- [ ] Etape 6 — Tests, stabilisation, deploiement VPS
