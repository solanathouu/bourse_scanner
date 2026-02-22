# Regles de tests — PEA Scanner

## Principes

- **TDD**: Ecrire les tests AVANT le code.
- **Chaque module a ses tests**. `src/extraction/pdf_parser.py` -> `tests/test_pdf_parser.py`.
- **Les tests sont independants**. Chaque test peut tourner seul, dans n'importe quel ordre.
- **Pas de donnees externes requises**. Utiliser `tempfile` et des mocks pour les tests unitaires.
  Exception: les tests sur vrais PDF utilisent `pytest.skip()` si les fichiers sont absents.

## Commandes

```bash
uv run pytest tests/ -v              # Tous les tests, verbose
uv run pytest tests/ -v -x           # Stopper au premier echec
uv run pytest tests/test_database.py -v  # Un fichier specifique
uv run pytest tests/ -k "pdf"        # Tests contenant "pdf" dans le nom
uv run pytest tests/ --tb=short      # Tracebacks courtes
```

## Structure d'un fichier de test

```python
"""Tests pour [module]."""

import os
import tempfile
import pytest

from src.module.fichier import MaClasse, MonErreur


class TestValidation:
    """Tests de validation/logique (pas de deps externes)."""

    def setup_method(self):
        """Setup commun a tous les tests de la classe."""
        self.instance = MaClasse()

    def test_cas_nominal(self):
        """Description claire de ce qui est teste."""
        result = self.instance.methode(input_valide)
        assert result == attendu

    def test_cas_erreur(self):
        """Verifie que l'erreur appropriee est levee."""
        with pytest.raises(MonErreur, match="message attendu"):
            self.instance.methode(input_invalide)


class TestAvecDonneesReelles:
    """Tests sur donnees reelles (skip si absentes)."""

    def _skip_if_no_data(self):
        if not os.path.exists("data/pdfs"):
            pytest.skip("Donnees de test absentes")

    def test_sur_vrai_fichier(self):
        self._skip_if_no_data()
        # ...
```

## Conventions de nommage des tests

```python
def test_<quoi>_<condition>():
    """Description en francais."""

# Exemples:
def test_parse_pdf_achat_sans_commission():
def test_validate_champs_manquants_leve_erreur():
def test_match_trades_vente_partielle():
def test_insert_execution_en_base():
```

## Ce qui doit etre teste

| Module | Tester | Ne PAS tester |
|--------|--------|---------------|
| database.py | CRUD, creation tables, requetes | Internals SQLite |
| pdf_parser.py | Parsing, validation, edge cases | pdfplumber lui-meme |
| trade_matcher.py | FIFO, ventes partielles, rendements | Tri stdlib |
| collectors | Transformation des donnees | Appels API externes (mocker) |
| model | Predictions, metriques | Internals XGBoost |

## Fixtures et donnees de test

- Utiliser `tempfile.TemporaryDirectory()` pour les bases SQLite temporaires.
- Utiliser `pytest.fixture` pour le setup reutilisable.
- Les vrais PDF sont dans `data/pdfs/` mais JAMAIS versionnes.
- Pour les tests qui dependent de donnees reelles, toujours `pytest.skip()` si absentes.

## Seuils de qualite

- **Tous les tests doivent passer** avant chaque commit.
- **Coverage minimum**: chaque fonction publique a au moins 1 test.
- **Pas de tests flaky**: un test passe toujours ou echoue toujours.
