# Conventions Python — PEA Scanner

## Nommage

| Element | Convention | Exemple |
|---------|-----------|---------|
| Fichiers | snake_case | `pdf_parser.py` |
| Classes | PascalCase | `SGPDFParser`, `TradeMatcher` |
| Fonctions/methodes | snake_case | `parse_pdf()`, `match_trades()` |
| Variables | snake_case | `prix_unitaire`, `montant_brut` |
| Constantes | UPPER_SNAKE | `MAX_RETRIES`, `DEFAULT_THRESHOLD` |
| Methodes privees | _prefixe | `_parse_text()`, `_validate()` |
| Fichiers de test | test_ prefixe | `test_pdf_parser.py` |

## Style de code

### Fonctions
- Max 30 lignes par fonction. Au-dela, extraire des sous-fonctions.
- Une fonction fait UNE chose. Si le nom contient "et/and", decouper.
- Docstring obligatoire pour les fonctions publiques.
- Type hints sur les signatures publiques.

```python
# Bien
def parse_pdf(self, pdf_path: str) -> dict:
    """Extrait les donnees d'un avis d'execution PDF SG."""
    ...

# Mal
def process(data):  # pas de types, pas de docstring, nom vague
    ...
```

### Classes
- Docstring de classe obligatoire (1 ligne: ce que fait la classe).
- Methodes publiques en premier, privees ensuite.
- Pas de methode de plus de 30 lignes.

### Gestion d'erreurs
- Exceptions custom pour chaque module (`PDFParseError`, etc.).
- Jamais de `except Exception: pass`. Toujours logger.
- Les erreurs de parsing/collecte ne doivent PAS arreter le batch.
  Pattern: logger l'erreur, continuer le batch, rapporter les erreurs a la fin.

```python
# Bien: batch resilient
for pdf in pdf_files:
    try:
        result = parser.parse_pdf(pdf)
        results.append(result)
    except PDFParseError as e:
        errors.append({"file": pdf, "error": str(e)})
        logger.error(f"Echec parsing {pdf}: {e}")

# Mal: un PDF en erreur arrete tout
for pdf in pdf_files:
    result = parser.parse_pdf(pdf)  # crash si un PDF est malorme
```

### Logging
- Utiliser `loguru` partout (`from loguru import logger`).
- Niveaux: `debug` (details), `info` (etapes), `warning` (anomalies non bloquantes), `error` (echecs).
- Jamais de `print()` dans src/. Uniquement dans scripts/ pour l'affichage CLI.

### Nombres et monnaie
- Les montants sont en `float` (precision suffisante pour notre usage).
- Format stockage: point decimal (`842.30`), pas de separateur milliers.
- Format affichage: `f"{montant:.2f}"` pour 2 decimales.
- Les pourcentages sont stockes en % (`4.25` = 4.25%), pas en ratio.

## Structure d'un module type

```python
"""Description du module en une ligne.

Details supplementaires si necessaire.
"""

# Imports stdlib
import os
import re
from datetime import datetime

# Imports externes
import pandas as pd
from loguru import logger

# Imports internes
from src.core.database import Database


class MonModule:
    """Description de la classe en une ligne."""

    def methode_publique(self, arg: str) -> dict:
        """Ce que fait cette methode."""
        ...

    def _methode_privee(self, data: dict) -> None:
        """Detail interne."""
        ...
```

## Fichiers de configuration

### config.yaml
- Toute valeur parametrable (seuils, intervalles, URLs).
- Pas de secrets (API keys).
- Commente chaque section.

### .env
- Uniquement les secrets (API keys, tokens).
- Jamais versionne.
- Template dans `.env.example`.
