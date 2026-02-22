# PEA Scanner Bot — Design Document

**Date** : 2026-02-22
**Statut** : Approuvé

## Contexte

Bot Python de veille boursière pour le PEA. Analyse des données marché en temps réel, corrélation avec les actualités, modèle prédictif ML entraîné sur l'historique de trading personnel (~200 avis d'exécution PDF Société Générale), alertes Telegram avec score de confiance.

**Style de trading** : Swing court terme (quelques jours), objectif 4-5% par trade.

## Décisions techniques

| Composant | Choix | Raison |
|-----------|-------|--------|
| Architecture | Monolithe simple, modules séparés | Suffisant pour 30 tickers, 1 utilisateur |
| Package manager | `uv` | Rapide, moderne, virtualenv intégré |
| BDD | SQLite (`sqlite3`) | Pas de serveur, simple |
| Scheduler | APScheduler | Léger, intégré au process |
| ML | XGBoost (classification binaire) | Performant sur petits datasets |
| PDF | pdfplumber | Meilleur pour tableaux/structures fixes |
| Indicateurs techniques | `ta` | RSI, MACD, Bollinger, etc. |
| Telegram | python-telegram-bot | Officielle, bien documentée |
| Logging | loguru | Simple, puissant |
| Config | pyyaml + python-dotenv | Sépare config de secrets |

## Architecture (4 modules)

```
PDF (one-shot) -> SQLite -> [Scheduler] -> Collecte données -> Feature engine -> ML scoring -> Filtrage -> Telegram
```

- **Module 1** : Extraction PDF -> tables `executions` + `trades_complets`
- **Module 2** : Collecte prix/volumes/news -> tables `prices`, `news`, `indicators`, `fundamentals`, `events`
- **Module 3** : Feature engineering + entraînement ML + scoring temps réel
- **Module 4** : Filtrage intelligent + alertes Telegram

## Plan d'implémentation (6 étapes séquentielles)

| Étape | Description | Dépendances |
|-------|-------------|-------------|
| 1 | Extraction PDF SG -> SQLite | Aucune (offline) |
| 2 | Collecte données marché + news | API keys |
| 3 | Corrélation historique (trades <-> catalyseurs) | Étapes 1+2 |
| 4 | Feature engineering + entraînement ML | Étapes 1+2+3 |
| 5 | Pipeline temps réel + Telegram | Étapes 2+4 |
| 6 | Tests, stabilisation, déploiement VPS | Tout |

## Contraintes

- Dev sur Windows, déploiement VPS Linux
- Niveau Python intermédiaire -> code clair, bien commenté
- Approche incrémentale : chaque étape validée avant la suivante
- SQLite suffisant, pas de migration prévue
- Aucune API key disponible pour l'instant (étape 1 = 100% offline)
