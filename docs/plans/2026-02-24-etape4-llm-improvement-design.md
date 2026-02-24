# Etape 4 bis — Amelioration ML via analyse LLM des trades

**Date** : 2026-02-24
**Statut** : APPROUVE

## Probleme

Le pipeline ML actuel a des faiblesses majeures :
- 48% des trades gagnants classifies "UNKNOWN" (regex trop basiques)
- 32% des trades (45/141) sans aucun catalyseur associe
- 49% des catalyseurs associes n'ont pas de match texte (news au hasard)
- Exemple : trade EXAIL +26.9% matche avec une news sur les fromages Savencia
- Feature #1 du modele = "has_text_match" (43%) = le nom apparait dans la news
- Le modele ne bat pas le baseline (toujours predire gagnant)

## Solution

Utiliser GPT-4o-mini pour analyser chaque trade historique en profondeur :
- Lire et comprendre le CONTENU des news
- Identifier le vrai catalyseur declencheur
- Classifier avec precision (plus de 48% UNKNOWN)
- Expliquer pourquoi Nicolas a achete et vendu

## Architecture

### Nouvelle table `trade_analyses_llm`

```sql
CREATE TABLE trade_analyses_llm (
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
```

### Nouveau module `src/analysis/llm_analyzer.py`

Pour chaque trade, envoie a GPT-4o-mini :
- Infos du trade (action, dates, prix, rendement)
- News dans la fenetre J-5 a J+1 (elargie)
- Indicateurs techniques au moment de l'achat

Demande au LLM :
1. Quelle news a declenche l'achat ? (selection du vrai catalyseur)
2. Type de catalyseur (EARNINGS, UPGRADE, etc.)
3. Confiance 0.0-1.0
4. Resume : "Nicolas a achete parce que..."
5. Sentiment de la news (-1.0 a +1.0)
6. Pourquoi il a vendu
7. Qualite du trade (EXCELLENT/BON/MOYEN/MAUVAIS)

Reponse forcee en JSON structure (response_format).
Batch de 5 en parallele. Reprise incrementale (skip si deja analyse).

### Features enrichies pour XGBoost

Anciennes features catalyseur (remplacees) :
- catalyst_type (regex, 48% UNKNOWN)
- nb_catalysts, best_catalyst_score, has_text_match, sentiment_avg, nb_news_sources

Nouvelles features (via LLM) :
- catalyst_type (classifie par LLM)
- catalyst_confidence (0.0-1.0)
- news_sentiment (-1.0 a +1.0)
- trade_quality_score (EXCELLENT=4, BON=3, MOYEN=2, MAUVAIS=1)
- has_clear_catalyst (1/0)
- buy_reason_length (proxy complexite)

Total : ~23 features (13 techniques + 6 LLM + 4 contexte).

## Fichiers

| Fichier | Action |
|---------|--------|
| `src/core/database.py` | Ajouter table + CRUD trade_analyses_llm |
| `src/analysis/llm_analyzer.py` | NOUVEAU — analyse LLM des trades |
| `src/analysis/feature_engine.py` | Lire trade_analyses_llm au lieu du regex |
| `scripts/analyze_trades_llm.py` | NOUVEAU — CLI one-shot |
| `tests/test_llm_analyzer.py` | NOUVEAU — tests du module |
| `tests/test_feature_engine.py` | Adapter aux nouvelles features |
| `pyproject.toml` | Ajouter dependance openai |

## Cout et temps

- ~166 appels GPT-4o-mini
- ~500 tokens input + ~200 tokens output par trade
- ~3$ total
- ~3-5 minutes d'execution

## Dependances

- `openai>=1.0.0` (a ajouter)
- `OPENAI_API_KEY` dans `.env`
