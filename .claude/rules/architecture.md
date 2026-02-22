# Regles d'architecture — PEA Scanner

## Principe fondamental

```
Les dependances coulent vers le BAS, jamais vers le HAUT.
core -> extraction -> data_collection -> analysis -> model -> alerts
```

Un module ne peut importer QUE les modules situes en dessous de lui dans la hierarchie.

## Structure des modules

```
src/
├── core/           # Niveau 0 — Fondation (ZERO import interne)
│   ├── database.py     # Classe Database (connexion, CRUD)
│   ├── config.py       # Chargement config YAML + .env
│   └── logger.py       # Configuration loguru (si besoin de centraliser)
│
├── extraction/     # Niveau 1 — Import uniquement de core/
│   ├── pdf_parser.py       # Parse PDF SG -> dict
│   └── trade_matcher.py    # Reconstruit trades FIFO
│
├── data_collection/ # Niveau 1 — Import uniquement de core/
│   ├── price_collector.py      # yfinance + APIs prix
│   ├── news_collector.py       # APIs news + RSS
│   ├── fundamental_collector.py # Donnees fondamentales
│   ├── rss_feeds.py            # Parsing flux RSS
│   └── scrapers/               # Scrapers web (isoles)
│       ├── investing_scraper.py
│       ├── boursorama_scraper.py
│       └── zonebourse_scraper.py
│
├── analysis/       # Niveau 2 — Import de core/ et data_collection/
│   ├── feature_engine.py       # Calcul du vecteur de features
│   ├── technical_indicators.py # RSI, MACD, Bollinger...
│   ├── sentiment_analyzer.py   # Score sentiment des news
│   └── catalyst_matcher.py     # Correlation trades <-> events
│
├── model/          # Niveau 3 — Import de core/, analysis/
│   ├── trainer.py      # Entrainement XGBoost
│   ├── backtester.py   # Walk-forward validation
│   ├── predictor.py    # Scoring temps reel
│   └── evaluator.py    # Metriques de performance
│
└── alerts/         # Niveau 4 — Import de core/, model/
    ├── telegram_bot.py     # Envoi messages Telegram
    ├── signal_filter.py    # Logique de filtrage intelligent
    └── formatter.py        # Formatage des alertes
```

## Regles strictes

### Fichiers
- **Max 300 lignes par fichier**. Au-dela, decouper en sous-modules.
- **1 fichier = 1 responsabilite**. Un fichier ne fait qu'une seule chose.
- **1 classe principale par fichier**. Fonctions utilitaires ok en complement.

### Imports
- **Jamais d'import circulaire**. Si A importe B, B ne peut pas importer A.
- **Imports relatifs interdits** (`from ..core` interdit). Toujours `from src.core`.
- **Ordre des imports**: stdlib -> packages externes -> src/ internes.

### Separation des responsabilites
- `core/` = fondation pure (BDD, config). Aucune logique metier.
- `extraction/` = extraction de donnees brutes. Pas de calcul d'indicateurs.
- `data_collection/` = collecte externe. Pas d'analyse.
- `analysis/` = calcul de features. Pas d'entrainement de modele.
- `model/` = ML et prediction. Pas d'envoi d'alertes.
- `alerts/` = notification. Pas de calcul de score.

### Scripts
- Les fichiers dans `scripts/` sont des **points d'entree CLI uniquement**.
- Toute la logique est dans `src/`. Les scripts appellent `src/`.
- Un script fait: parse args -> appeler src/ -> afficher resultat. C'est tout.

### Configuration
- Secrets (API keys, tokens) dans `.env` (jamais versionne).
- Configuration (seuils, intervalles, watchlist) dans `config/*.yaml`.
- Pas de valeurs en dur dans le code. Tout passe par config ou constantes.

## Anti-patterns interdits

| Interdit | Faire plutot |
|----------|-------------|
| Logique metier dans un script | Mettre dans src/, appeler depuis le script |
| Import circulaire | Reorganiser les deps, extraire dans core/ |
| Fichier > 300 lignes | Decouper en sous-modules |
| Valeur en dur (API URL, seuil) | Mettre dans config.yaml ou .env |
| try/except vide (pass) | Logger l'erreur au minimum |
| print() pour debug | Utiliser loguru (logger.debug/info/warning/error) |
| Connexion BDD dans chaque fonction | Passer par Database class dans core/ |
