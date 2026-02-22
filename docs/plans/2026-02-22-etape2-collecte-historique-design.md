# Design Etape 2 — Collecte de donnees historiques

**Date**: 2026-02-22
**Statut**: Valide
**Focus**: Donnees historiques des 19 actions deja tradees

## Objectif

Collecter les prix OHLCV et les news autour des dates de chaque trade historique
pour alimenter l'etape 3 (correlation trades <-> catalyseurs) et l'etape 4 (ML).

## Decisions cles

| Aspect | Decision |
|--------|----------|
| Focus | Donnees historiques des 19 actions tradees (pas la watchlist) |
| Prix | yfinance, journalier EOD, 30j avant achat -> vente |
| News | 5 sources par vagues |
| Fenetre news | 7j avant + 3j apres chaque trade |
| Implementation | Par vagues successives |
| Cout | 0 EUR (toutes les cles API sont gratuites) |

## Architecture

### Nouveaux fichiers

```
src/data_collection/
├── __init__.py
├── price_collector.py    # Collecte prix OHLCV via yfinance
├── news_collector.py     # Collecte news via GNews + APIs
└── ticker_mapper.py      # Mapping nom_action -> ticker Yahoo
```

### Script d'entree

```
scripts/collect_historical.py
  --prices   Collecter seulement les prix
  --news     Collecter seulement les news
  (defaut)   Collecter tout
```

## Sources de donnees

### Prix
| Source | Type | Cle API | Usage |
|--------|------|---------|-------|
| yfinance | Python lib | Non | Prix OHLCV journalier |

### News (par vagues)
| Vague | Source | Type | Cle API | Ce qu'elle apporte |
|-------|--------|------|---------|-------------------|
| 1 | GNews | Google News RSS | Non | News FR avec dates, ~100 articles/requete |
| 2 | Boursorama scraper | Web scraping | Non | News FR specifiques par ticker |
| 2 | Zonebourse scraper | Web scraping | Non | Analyses, consensus |
| 3 | Alpha Vantage | API REST | Oui (gratuite) | News EN + sentiment |
| 3 | Marketaux | API REST | Oui (gratuite) | News multi-langue |

### Sources eliminees
| Source | Raison |
|--------|--------|
| Finnhub | Ne couvre PAS les news d'actions europeennes |
| NewsAPI.org | Seulement 1 mois d'historique gratuit |
| EODHD | News payant (19.99 EUR/mois) |
| Finlight | 1 mois d'historique gratuit seulement |

## Schema de donnees

### Table `prices`
| Colonne | Type | Description |
|---------|------|-------------|
| id | INTEGER PK | Auto-increment |
| ticker | TEXT | Ex: SAN.PA |
| date | TEXT | YYYY-MM-DD |
| open | REAL | Prix ouverture |
| high | REAL | Plus haut |
| low | REAL | Plus bas |
| close | REAL | Cloture |
| volume | INTEGER | Volume echange |
| UNIQUE(ticker, date) | | Pas de doublons |

### Table `news`
| Colonne | Type | Description |
|---------|------|-------------|
| id | INTEGER PK | Auto-increment |
| ticker | TEXT | Ex: SAN.PA |
| title | TEXT | Titre de l'article |
| source | TEXT | Ex: BFM Bourse, Les Echos |
| url | TEXT UNIQUE | Lien article (deduplication) |
| published_at | TEXT | YYYY-MM-DD HH:MM:SS |
| description | TEXT | Resume/extrait |

## Logique de collecte

### PriceCollector
1. Lire trades_complets -> extraire (ticker, date_achat, date_vente)
2. Pour chaque trade: yfinance.download(ticker, start=achat-30j, end=vente)
3. Deduplication: fusionner les periodes si plusieurs trades sur le meme ticker
4. Insert dans table prices

### NewsCollector
1. Lire trades_complets -> extraire (nom_action, date_achat)
2. Pour chaque trade: chercher les news (start=achat-7j, end=achat+3j)
3. Sources interrogees dans l'ordre: GNews, puis scrapers, puis APIs
4. Delai entre requetes (2-3 sec) pour eviter le rate limiting
5. Insert dans table news (dedoublonnage par URL)

### Gestion d'erreurs
- Ticker yfinance vide -> log warning, continuer batch
- GNews rate limit -> attendre 30 sec, retry 3x, puis skip
- API erreur -> log error, continuer avec les autres sources
- Rapport final: X prix, Y news, Z erreurs

## Tests

| Fichier | Ce qu'on teste |
|---------|---------------|
| test_ticker_mapper.py | Mapping nom -> ticker, noms inconnus |
| test_price_collector.py | Transformation yfinance, insertion BDD (mock) |
| test_news_collector.py | Transformation articles, deduplication (mock) |

Principe: mock tous les appels externes, tester la logique de transformation.

## Watchlist de 30 valeurs

Fournie par l'utilisateur (2 screenshots). Sera utilisee a l'etape 5 pour le scan
temps reel. Les 30 valeurs:

2CRSI, AB SCIENCE, ADOCIA, AFYREN, AM.MSCI WORLD SWAP, AVENIR TELECOM REG,
BIOSYNEX, BNPP S&P500EUR ETF, CAPITAL B., CROSSJECT, DBV TECHNOLOGIES,
EXAIL TECHNOLOGIES, GENFIT, INVENTIVA, KALRAY, MAUREL PROM, MEDIAN TECHNOLOG.,
MEMSCAP REGPT, NANOBIOTIX, POXEL, REXEL, SANOFI, SCHNEIDER ELECTRIC,
SENSORION, SOITEC REGROUPEM., UBISOFT ENTERTAIN, VALBIOTIS S.A., VALNEVA SE,
VINCI, WORLDLINE
