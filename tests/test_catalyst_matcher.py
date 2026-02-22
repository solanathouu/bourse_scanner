"""Tests pour le catalyst_matcher."""

import os
import tempfile

import pytest

from src.core.database import Database
from src.analysis.catalyst_matcher import CatalystMatcher


class TestComputeScore:
    """Tests du calcul de score de pertinence."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.matcher = CatalystMatcher(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_score_jour_achat(self):
        """J0 (jour d'achat) = score max 1.0."""
        assert self.matcher._compute_score(0, False) == 1.0

    def test_score_veille(self):
        """J-1 = 0.8."""
        assert self.matcher._compute_score(-1, False) == 0.8

    def test_score_j_moins_2(self):
        """J-2 = 0.6."""
        assert self.matcher._compute_score(-2, False) == 0.6

    def test_score_j_moins_3(self):
        """J-3 = 0.4."""
        assert self.matcher._compute_score(-3, False) == 0.4

    def test_score_lendemain(self):
        """J+1 = 0.7."""
        assert self.matcher._compute_score(1, False) == 0.7

    def test_score_avec_match_texte(self):
        """Match texte ajoute +0.2, cap a 1.0."""
        assert self.matcher._compute_score(-1, True) == 1.0  # 0.8 + 0.2
        assert self.matcher._compute_score(0, True) == 1.0   # 1.0 + 0.2 cappe a 1.0

    def test_score_distance_hors_fenetre(self):
        """Distance hors fenetre retourne 0.0."""
        assert self.matcher._compute_score(-5, False) == 0.0
        assert self.matcher._compute_score(3, False) == 0.0


class TestCheckTextMatch:
    """Tests du matching texte nom_action dans titre/description."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.matcher = CatalystMatcher(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_match_exact_titre(self):
        """Nom exact dans le titre."""
        assert self.matcher._check_text_match(
            "SANOFI", "Sanofi annonce ses resultats Q3", ""
        ) is True

    def test_match_exact_description(self):
        """Nom exact dans la description."""
        assert self.matcher._check_text_match(
            "SANOFI", "Actu pharma", "Sanofi a publie ses resultats"
        ) is True

    def test_no_match(self):
        """Aucun match."""
        assert self.matcher._check_text_match(
            "SANOFI", "L'Oreal publie ses resultats", "Bons chiffres"
        ) is False

    def test_match_case_insensitive(self):
        """Matching insensible a la casse."""
        assert self.matcher._check_text_match(
            "AIR LIQUIDE", "air liquide en hausse", ""
        ) is True

    def test_match_nom_avec_etoile(self):
        """Noms avec prefixe * (achats anterieurs aux PDF)."""
        assert self.matcher._check_text_match(
            "* GENFIT", "Genfit obtient un brevet", ""
        ) is True

    def test_match_none_description(self):
        """Description None ne crashe pas."""
        assert self.matcher._check_text_match(
            "SANOFI", "Sanofi news", None
        ) is True

    def test_no_match_none_description(self):
        """Description None, pas de match dans titre."""
        assert self.matcher._check_text_match(
            "SANOFI", "L'Oreal news", None
        ) is False


class TestMatchTrade:
    """Tests du matching d'un trade avec les news."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.matcher = CatalystMatcher(self.db)
        # Inserer des news de test
        self.db.insert_news_batch([
            {"ticker": "SAN.PA", "title": "Sanofi resultats Q3",
             "source": "Reuters", "url": "https://ex.com/1",
             "published_at": "2025-07-09 10:00:00",
             "description": "Bons resultats Sanofi",
             "sentiment": 0.5, "source_api": "gnews"},
            {"ticker": "SAN.PA", "title": "FDA approves Sanofi drug",
             "source": "Bloomberg", "url": "https://ex.com/2",
             "published_at": "2025-07-10 08:00:00",
             "description": "New Sanofi treatment approved",
             "sentiment": 0.8, "source_api": "alpha_vantage"},
            {"ticker": "SAN.PA", "title": "Marche en hausse",
             "source": "BFM", "url": "https://ex.com/3",
             "published_at": "2025-07-10 14:00:00",
             "description": "Le CAC40 gagne 1%",
             "sentiment": None, "source_api": "rss_google_news_bourse"},
            {"ticker": "SAN.PA", "title": "Sanofi hors fenetre",
             "source": "Reuters", "url": "https://ex.com/4",
             "published_at": "2025-07-01 10:00:00",
             "description": "Trop ancien",
             "sentiment": None, "source_api": "gnews"},
        ])

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_match_trade_trouve_catalyseurs(self):
        """Match un trade et trouve les news dans la fenetre."""
        trade = {
            "id": 1, "nom_action": "SANOFI", "date_achat": "2025-07-10",
            "isin": "FR0000120578", "statut": "CLOTURE",
        }
        catalyseurs = self.matcher.match_trade(trade)
        # 3 news dans fenetre [07-07, 07-11], pas la news du 07-01
        assert len(catalyseurs) == 3

    def test_match_trade_scores_corrects(self):
        """Verifie les scores: J-1 avec match texte, J0 avec match, J0 sans match."""
        trade = {
            "id": 1, "nom_action": "SANOFI", "date_achat": "2025-07-10",
            "isin": "FR0000120578", "statut": "CLOTURE",
        }
        catalyseurs = self.matcher.match_trade(trade)
        # Trier par news_id pour un ordre previsible
        catalyseurs.sort(key=lambda c: c["news_id"])
        # News 1: 07-09, J-1, match texte "Sanofi" -> 0.8 + 0.2 = 1.0
        assert catalyseurs[0]["score_pertinence"] == 1.0
        assert catalyseurs[0]["distance_jours"] == -1
        assert catalyseurs[0]["match_texte"] == 1
        # News 2: 07-10, J0, match texte "Sanofi" -> 1.0 + 0.2 = 1.0 (cap)
        assert catalyseurs[1]["score_pertinence"] == 1.0
        assert catalyseurs[1]["distance_jours"] == 0
        # News 3: 07-10, J0, pas de match texte "Sanofi" -> 1.0
        assert catalyseurs[2]["score_pertinence"] == 1.0
        assert catalyseurs[2]["distance_jours"] == 0
        assert catalyseurs[2]["match_texte"] == 0

    def test_match_trade_ticker_inconnu(self):
        """Trade avec nom_action sans ticker connu retourne liste vide."""
        trade = {
            "id": 99, "nom_action": "INCONNU SA", "date_achat": "2025-07-10",
            "isin": "FRINCONNU", "statut": "CLOTURE",
        }
        catalyseurs = self.matcher.match_trade(trade)
        assert catalyseurs == []

    def test_match_trade_aucune_news(self):
        """Trade sans news dans la fenetre retourne liste vide."""
        trade = {
            "id": 1, "nom_action": "SANOFI", "date_achat": "2025-01-01",
            "isin": "FR0000120578", "statut": "CLOTURE",
        }
        catalyseurs = self.matcher.match_trade(trade)
        assert catalyseurs == []


class TestMatchAllTrades:
    """Tests du matching batch sur tous les trades."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.matcher = CatalystMatcher(self.db)
        # 2 trades
        self.db.insert_trades_batch([
            {"isin": "FR0000120578", "nom_action": "SANOFI",
             "date_achat": "2025-07-10", "date_vente": "2025-07-15",
             "prix_achat": 95.0, "prix_vente": 100.0, "quantite": 10,
             "rendement_brut_pct": 5.26, "rendement_net_pct": 5.0,
             "duree_jours": 5, "frais_totaux": 2.5, "statut": "CLOTURE"},
            {"isin": "FR0000120271", "nom_action": "AIR LIQUIDE",
             "date_achat": "2025-07-10", "date_vente": "2025-07-12",
             "prix_achat": 170.0, "prix_vente": 175.0, "quantite": 5,
             "rendement_brut_pct": 2.94, "rendement_net_pct": 2.5,
             "duree_jours": 2, "frais_totaux": 1.5, "statut": "CLOTURE"},
        ])
        # News pour SANOFI
        self.db.insert_news({
            "ticker": "SAN.PA", "title": "Sanofi news",
            "source": "Reuters", "url": "https://ex.com/san1",
            "published_at": "2025-07-10", "description": "Sanofi actu",
            "sentiment": 0.5, "source_api": "gnews",
        })
        # News pour AIR LIQUIDE
        self.db.insert_news({
            "ticker": "AI.PA", "title": "Air Liquide en hausse",
            "source": "BFM", "url": "https://ex.com/ai1",
            "published_at": "2025-07-09", "description": "Air Liquide annonce",
            "sentiment": None, "source_api": "rss_google_news_air_liquide",
        })

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_match_all_trades_resume(self):
        """Matche tous les trades et retourne un resume."""
        result = self.matcher.match_all_trades()
        assert result["total_trades"] == 2
        assert result["trades_avec_catalyseurs"] == 2
        assert result["total_associations"] == 2
        assert result["erreurs"] == 0

    def test_match_all_trades_peuple_table(self):
        """Apres match_all, la table trade_catalyseurs est peuplee."""
        self.matcher.match_all_trades()
        assert self.db.count_catalyseurs() == 2

    def test_match_all_trades_clear_avant(self):
        """match_all_trades vide la table avant de re-matcher."""
        self.matcher.match_all_trades()
        assert self.db.count_catalyseurs() == 2
        # Re-matcher ne duplique pas
        self.matcher.match_all_trades()
        assert self.db.count_catalyseurs() == 2


class TestGetStats:
    """Tests des statistiques."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.matcher = CatalystMatcher(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_stats_base_vide(self):
        """Stats sur une base sans catalyseurs."""
        stats = self.matcher.get_stats()
        assert stats["total_catalyseurs"] == 0
        assert stats["total_trades"] == 0
