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
| 2 | TODO | Collecte donnees marche + news |
| 3 | TODO | Correlation historique (trades <-> catalyseurs) |
| 4 | TODO | Feature engineering + entrainement ML |
| 5 | TODO | Pipeline temps reel + alertes Telegram |
| 6 | TODO | Tests integration, stabilisation, deploiement VPS |

## Current Project State

| Aspect | Status | Details |
|--------|--------|---------|
| Code | Etape 1 DONE | pdf_parser, trade_matcher, database — tout fonctionne |
| Config | Minimal | pyproject.toml, .env.example, pas encore de config.yaml |
| Tests | 30/30 PASS | test_database (6), test_pdf_parser (17), test_trade_matcher (7) |
| Data | Importee | 214 PDF -> 166 trades (141 clotures, 25 ouverts) en SQLite |
| Docs | Complets | CLAUDE.md, .claude/rules/ (4 fichiers), README.md, docs/plans/ |
| Git | Clean | 6 commits, pas de remote configure |

## Next Immediate Action

**Etape 2: Collecte de donnees marche + news.** Pour demarrer:

1. Demander a l'utilisateur sa watchlist de 30 tickers (format Yahoo: MC.PA, BN.PA, etc.)
2. Creer `config/config.yaml` et `config/tickers_watchlist.yaml`
3. Implementer `src/data_collection/price_collector.py` avec yfinance
4. L'utilisateur doit creer des comptes API (Alpha Vantage, Finnhub, etc.) au besoin

Utiliser le skill `superpowers:brainstorming` avant de commencer l'etape 2 pour valider l'approche.

## Donnees cles du trading

- **Style**: Swing court terme (quelques jours), objectif 4-5% par trade
- **Perimetre**: Actions eligibles PEA (marches europeens)
- **Watchlist**: 30 valeurs (liste fournie par l'utilisateur)
- **Performance historique**: 89% win rate, +9.21% rendement moyen, 20.7j duree moyenne
- **19 actions tradees** sur ~10 mois (mai 2025 - fev 2026)
