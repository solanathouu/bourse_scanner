# Etape 3 — Catalyst Matcher Design

**Date**: 2026-02-22
**Objectif**: Associer chaque trade a ses catalyseurs (news/events autour de la date d'achat)

## Contexte

- 166 trades (mai 2025 -> fev 2026), 141 clotures + 25 ouverts
- 1824 news de 4 sources (GNews, Alpha Vantage, Marketaux, RSS)
- Lien trade -> news via TickerMapper (nom_action -> ticker Yahoo)
- Table trade_catalyseurs a creer

## Decisions de design

| Decision | Choix | Justification |
|----------|-------|---------------|
| Fenetre temporelle | J-3 a J+1 | Style swing court terme, cible les catalyseurs immediats |
| Filtrage bruit AV | Garder tout, scorer | Croisement multi-sources pour fiabilite |
| Criteres de score | Proximite temporelle + matching texte | Simple, deterministe, pas de NLP lourd |
| Approche technique | SQL jointure + Python scoring | Performant, simple, pas de nouvelle dep |
| Sortie | Table + rapport console + stats | Associations brutes pour etape 4, stats pour analyse |

## Table trade_catalyseurs

```sql
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
CREATE INDEX IF NOT EXISTS idx_catalyseurs_trade ON trade_catalyseurs(trade_id);
CREATE INDEX IF NOT EXISTS idx_catalyseurs_news ON trade_catalyseurs(news_id);
```

**Colonnes**:
- `score_pertinence`: 0.0 a 1.0, combine proximite temporelle + bonus texte
- `distance_jours`: -3 a +1 (negatif = avant achat, 0 = jour d'achat)
- `match_texte`: 1 si nom_action trouve dans titre/description de la news

## Classe CatalystMatcher

Fichier: `src/analysis/catalyst_matcher.py`

```python
class CatalystMatcher:
    """Associe les trades a leurs catalyseurs (news dans la fenetre temporelle)."""

    def __init__(self, db: Database, days_before: int = 3, days_after: int = 1)
    def match_trade(self, trade: dict) -> list[dict]
    def match_all_trades(self) -> dict
    def get_stats(self) -> dict
    def _compute_score(self, distance_jours: int, match_texte: bool) -> float
    def _check_text_match(self, nom_action: str, title: str, description: str) -> bool
```

### Logique de scoring

```
Score = ponderation_temporelle + bonus_texte

Ponderation temporelle:
  J0  (jour achat): 1.0
  J-1 (veille):     0.8
  J-2:              0.6
  J-3:              0.4
  J+1 (lendemain):  0.7

Bonus match texte: +0.2 (cap a 1.0)
```

### Matching texte

- Case-insensitive: "SANOFI" matche "Sanofi annonce..."
- Cherche dans titre ET description
- Gere les noms avec prefixe `*` (via TickerMapper._clean_name)

### Flux de traitement

```
Pour chaque trade:
  1. nom_action -> ticker (via TickerMapper)
  2. SQL: SELECT news WHERE ticker = ? AND published_at BETWEEN date_achat-3 AND date_achat+1
  3. Pour chaque news trouvee:
     a. Calculer distance_jours
     b. Verifier match_texte (nom_action dans titre/description)
     c. Calculer score_pertinence
     d. INSERT INTO trade_catalyseurs
  4. Si erreur (ticker inconnu, etc.): logger, continuer le batch
```

## Script CLI

Fichier: `scripts/match_catalysts.py`

```
uv run python scripts/match_catalysts.py           # Matcher tous les trades
uv run python scripts/match_catalysts.py --stats    # Afficher stats seulement
```

### Rapport console

Affiche apres matching:
- Trades analyses / avec catalyseurs / sans catalyseurs
- Total associations, score moyen
- Repartition par source (gnews, alpha_vantage, etc.)
- Top 5 trades avec le plus de catalyseurs
- Comparaison gagnants vs perdants (nombre moyen de catalyseurs, score moyen)

## Modifications sur code existant

| Fichier | Modification |
|---------|-------------|
| `src/core/database.py` | Ajouter: create table, insert_catalyseur, insert_catalyseurs_batch, get_catalyseurs, count_catalyseurs, clear_catalyseurs |
| `scripts/init_db.py` | Ajouter creation table trade_catalyseurs |
| `CLAUDE.md` | Ajouter commande match_catalysts.py |

## Fichiers a creer

| Fichier | Description |
|---------|-------------|
| `src/analysis/__init__.py` | Package analysis |
| `src/analysis/catalyst_matcher.py` | Classe CatalystMatcher |
| `scripts/match_catalysts.py` | Script CLI |
| `tests/test_catalyst_matcher.py` | Tests unitaires |

## Dependances

```
catalyst_matcher
  <- src.core.database (Database)
  <- src.data_collection.ticker_mapper (TickerMapper)
```

Aucune nouvelle dependance Python. Tout avec sqlite3 + modules existants.
