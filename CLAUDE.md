# PEA Scanner Bot

Bot Python de veille boursiere PEA. Analyse des donnees marche en temps reel, correlation avec les actualites, modele predictif ML entraine sur l'historique de trading personnel, alertes Telegram avec score de confiance.

## Commandes

```bash
uv run pytest tests/ -v          # Lancer tous les tests
uv run pytest tests/ -v -x       # Stopper au premier echec
uv run python scripts/init_db.py       # Initialiser la base SQLite
uv run python scripts/import_pdfs.py   # Importer les PDF dans la base
uv run python scripts/explore_pdf.py <fichier.pdf>  # Debug: voir le texte brut d'un PDF
```

## Architecture

```
pea-scanner/
├── src/                        # Code source (packages Python)
│   ├── core/                   # Couche fondation (BDD, logging, config)
│   ├── extraction/             # Module 1: Extraction PDF -> SQLite
│   ├── data_collection/        # Module 2: Collecte prix/news (a venir)
│   ├── analysis/               # Module 3: Features + ML (a venir)
│   ├── model/                  # Module 3: Entrainement + scoring (a venir)
│   └── alerts/                 # Module 4: Telegram (a venir)
├── scripts/                    # Points d'entree CLI
├── tests/                      # Tests (miroir de src/)
├── config/                     # Fichiers de configuration YAML
├── data/                       # Donnees (non versionne: PDFs, .db, models)
├── docs/plans/                 # Documents de design et plans d'implementation
└── logs/                       # Logs applicatifs (non versionne)
```

**Flux de donnees:**
```
PDF (one-shot) -> SQLite -> [Scheduler] -> Collecte donnees -> Feature engine -> ML scoring -> Filtrage -> Telegram
```

**Dependances entre modules (sens unique, jamais circulaire):**
```
core (fondation, 0 deps internes)
  <- extraction (utilise core.database)
  <- data_collection (utilise core.database, core.config)
  <- analysis (utilise core.database, data_collection)
  <- model (utilise core.database, analysis)
  <- alerts (utilise core.database, model)
```

## Regles d'architecture

- Lire `.claude/rules/architecture.md` pour les regles completes
- Lire `.claude/rules/python-conventions.md` pour les conventions de code
- Lire `.claude/rules/testing.md` pour les regles de tests

## Stack technique

| Composant | Choix |
|-----------|-------|
| Python | 3.13 |
| Package manager | uv |
| BDD | SQLite (sqlite3 stdlib) |
| PDF | pdfplumber |
| Data | pandas, numpy |
| ML | xgboost, scikit-learn |
| Indicateurs techniques | ta |
| Scheduler | APScheduler |
| Telegram | python-telegram-bot |
| Logging | loguru |
| Config | pyyaml + python-dotenv |
| Tests | pytest |

## Base de donnees

SQLite dans `data/trades.db`. Tables principales:

- `executions` — Avis d'execution PDF parses (1 ligne = 1 PDF)
- `trades_complets` — Trades reconstitues achat->vente (FIFO)
- `trade_catalyseurs` — News/events associes a chaque trade
- `prices` — Prix OHLCV collectes
- `indicators` — Indicateurs techniques calcules
- `news` — Actualites collectees
- `signals` — Signaux emis par le modele

## Etapes d'implementation

| Etape | Statut | Description |
|-------|--------|-------------|
| 1 | DONE | Extraction PDF SG -> SQLite (214 PDF, 166 trades) |
| 2 | EN COURS | Collecte donnees historiques (prix + news) — design fait, implementation a lancer |
| 3 | TODO | Correlation historique (trades <-> catalyseurs) |
| 4 | TODO | Feature engineering + entrainement ML |
| 5 | TODO | Pipeline temps reel + alertes Telegram |
| 6 | TODO | Tests integration, stabilisation, deploiement VPS |

## Current Project State

| Aspect | Status | Details |
|--------|--------|---------|
| Code | Etape 1 DONE, Etape 2 design fait | pdf_parser, trade_matcher, database OK. data_collection/ pas encore code |
| Config | .env configure | .env avec ALPHA_VANTAGE_API_KEY + MARKETAUX_API_KEY |
| Tests | 30/30 PASS | test_database (6), test_pdf_parser (17), test_trade_matcher (7) |
| Data | Importee | 214 PDF -> 166 trades (141 clotures, 25 ouverts) en SQLite |
| Docs | Design + plan Etape 2 | docs/plans/2026-02-22-etape2-*.md (design + implementation) |
| Git | Remote configure | origin sur GitHub |

## Next Immediate Action

**Etape 2: Executer le plan d'implementation.** Le design et le plan sont faits.

Lire `docs/plans/2026-02-22-etape2-implementation.md` et executer les 7 taches dans l'ordre:

1. ~~Brainstorming + design~~ FAIT
2. Ajouter deps yfinance + gnews dans pyproject.toml (`uv sync`)
3. Ajouter tables prices + news dans database.py (TDD)
4. Creer ticker_mapper.py avec mapping des 19 actions (TDD)
5. Creer price_collector.py avec yfinance (TDD)
6. Creer news_collector.py avec GNews (TDD)
7. Creer scripts/collect_historical.py
8. Test d'integration + run complet

**Approche:** Subagent-driven development (un agent par tache, review entre chaque).

**API Keys deja configurees dans .env:**
- ALPHA_VANTAGE_API_KEY (gratuite)
- MARKETAUX_API_KEY (gratuite)

**Sources de donnees validees:**
- Prix: yfinance (gratuit, pas de cle)
- News: GNews Python lib (gratuit, pas de cle) + Alpha Vantage + Marketaux
- Watchlist 30 valeurs fournie (screenshots dans data/) — pour etape 5 seulement
- Focus etape 2: donnees HISTORIQUES des 19 actions deja tradees

## Donnees cles du trading

- **Style**: Swing court terme (quelques jours), objectif 4-5% par trade
- **Perimetre**: Actions eligibles PEA (marches europeens)
- **Watchlist**: 30 valeurs (liste fournie par l'utilisateur)
- **Performance historique**: 89% win rate, +9.21% rendement moyen, 20.7j duree moyenne
- **19 actions tradees** sur ~10 mois (mai 2025 - fev 2026)
