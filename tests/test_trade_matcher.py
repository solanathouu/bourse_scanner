"""Tests pour le matching achat/vente et reconstruction des trades."""

from src.extraction.trade_matcher import TradeMatcher


def test_simple_buy_sell():
    """Un achat suivi d'une vente = 1 trade complet clôturé."""
    executions = [
        {
            "date_execution": "2025-05-09",
            "heure_execution": "14:22:06",
            "sens": "ACHAT",
            "nom_action": "EXAIL TECHNOLOGIES",
            "isin": "FR0000062671",
            "quantite": 10,
            "prix_unitaire": 54.0,
            "montant_brut": 540.0,
            "commission": 0.0,
            "frais": 0.0,
            "montant_net": 540.0,
        },
        {
            "date_execution": "2025-05-23",
            "heure_execution": "10:30:00",
            "sens": "VENTE",
            "nom_action": "EXAIL TECHNOLOGIES",
            "isin": "FR0000062671",
            "quantite": 10,
            "prix_unitaire": 62.0,
            "montant_brut": 620.0,
            "commission": 3.10,
            "frais": 0.0,
            "montant_net": 616.90,
        },
    ]

    matcher = TradeMatcher()
    trades = matcher.match_trades(executions)

    assert len(trades) == 1
    trade = trades[0]
    assert trade["nom_action"] == "EXAIL TECHNOLOGIES"
    assert trade["isin"] == "FR0000062671"
    assert trade["statut"] == "CLOTURE"
    assert trade["quantite"] == 10
    assert trade["prix_achat"] == 54.0
    assert trade["prix_vente"] == 62.0
    assert trade["duree_jours"] == 13.8  # 9 mai 14:22 -> 23 mai 10:30 (heures prises en compte)


def test_buy_without_sell():
    """Un achat sans vente = trade ouvert."""
    executions = [
        {
            "date_execution": "2025-08-20",
            "heure_execution": "09:04:07",
            "sens": "ACHAT",
            "nom_action": "ADOCIA",
            "isin": "FR0011184241",
            "quantite": 75,
            "prix_unitaire": 8.35,
            "montant_brut": 626.25,
            "commission": 3.13,
            "frais": 0.0,
            "montant_net": 629.38,
        },
    ]

    matcher = TradeMatcher()
    trades = matcher.match_trades(executions)

    assert len(trades) == 1
    assert trades[0]["statut"] == "OUVERT"
    assert trades[0]["prix_vente"] is None
    assert trades[0]["rendement_brut_pct"] is None


def test_partial_sell():
    """Achat 100 puis vente 50 = 1 trade clôturé (50) + 1 trade ouvert (50)."""
    executions = [
        {
            "date_execution": "2025-03-10",
            "heure_execution": "09:00:00",
            "sens": "ACHAT",
            "nom_action": "BNP",
            "isin": "FR0000131104",
            "quantite": 100,
            "prix_unitaire": 60.00,
            "montant_brut": 6000.00,
            "commission": 4.95,
            "frais": 0.0,
            "montant_net": 6004.95,
        },
        {
            "date_execution": "2025-03-15",
            "heure_execution": "10:00:00",
            "sens": "VENTE",
            "nom_action": "BNP",
            "isin": "FR0000131104",
            "quantite": 50,
            "prix_unitaire": 63.00,
            "montant_brut": 3150.00,
            "commission": 4.95,
            "frais": 0.0,
            "montant_net": 3145.05,
        },
    ]

    matcher = TradeMatcher()
    trades = matcher.match_trades(executions)

    assert len(trades) == 2
    closed = [t for t in trades if t["statut"] == "CLOTURE"]
    opened = [t for t in trades if t["statut"] == "OUVERT"]
    assert len(closed) == 1
    assert len(opened) == 1
    assert closed[0]["quantite"] == 50
    assert opened[0]["quantite"] == 50


def test_rendement_calculation():
    """Le rendement brut % doit être correctement calculé."""
    executions = [
        {
            "date_execution": "2025-03-15",
            "heure_execution": "09:00:00",
            "sens": "ACHAT",
            "nom_action": "TEST",
            "isin": "FR0000000001",
            "quantite": 10,
            "prix_unitaire": 100.00,
            "montant_brut": 1000.00,
            "commission": 5.00,
            "frais": 0.0,
            "montant_net": 1005.00,
        },
        {
            "date_execution": "2025-03-20",
            "heure_execution": "09:00:00",
            "sens": "VENTE",
            "nom_action": "TEST",
            "isin": "FR0000000001",
            "quantite": 10,
            "prix_unitaire": 105.00,
            "montant_brut": 1050.00,
            "commission": 5.00,
            "frais": 0.0,
            "montant_net": 1045.00,
        },
    ]

    matcher = TradeMatcher()
    trades = matcher.match_trades(executions)

    trade = trades[0]
    assert trade["rendement_brut_pct"] == 5.0  # (105-100)/100 * 100
    assert trade["frais_totaux"] == 10.0  # 5 achat + 5 vente


def test_multiple_actions_separated():
    """Des exécutions sur plusieurs actions sont bien séparées par ISIN."""
    executions = [
        {
            "date_execution": "2025-03-10", "heure_execution": "09:00:00",
            "sens": "ACHAT", "nom_action": "EXAIL", "isin": "FR0000062671",
            "quantite": 10, "prix_unitaire": 50.0, "montant_brut": 500.0,
            "commission": 0.0, "frais": 0.0, "montant_net": 500.0,
        },
        {
            "date_execution": "2025-03-11", "heure_execution": "09:00:00",
            "sens": "ACHAT", "nom_action": "ADOCIA", "isin": "FR0011184241",
            "quantite": 50, "prix_unitaire": 8.0, "montant_brut": 400.0,
            "commission": 2.0, "frais": 0.0, "montant_net": 402.0,
        },
        {
            "date_execution": "2025-03-15", "heure_execution": "09:00:00",
            "sens": "VENTE", "nom_action": "EXAIL", "isin": "FR0000062671",
            "quantite": 10, "prix_unitaire": 55.0, "montant_brut": 550.0,
            "commission": 2.75, "frais": 0.0, "montant_net": 547.25,
        },
    ]

    matcher = TradeMatcher()
    trades = matcher.match_trades(executions)

    assert len(trades) == 2
    exail = [t for t in trades if t["isin"] == "FR0000062671"][0]
    adocia = [t for t in trades if t["isin"] == "FR0011184241"][0]
    assert exail["statut"] == "CLOTURE"
    assert adocia["statut"] == "OUVERT"


def test_multiple_buys_then_full_sell():
    """Plusieurs achats puis une vente totale = FIFO."""
    executions = [
        {
            "date_execution": "2025-05-09", "heure_execution": "14:22:06",
            "sens": "ACHAT", "nom_action": "EXAIL", "isin": "FR0000062671",
            "quantite": 10, "prix_unitaire": 54.0, "montant_brut": 540.0,
            "commission": 0.0, "frais": 0.0, "montant_net": 540.0,
        },
        {
            "date_execution": "2025-05-12", "heure_execution": "09:30:00",
            "sens": "ACHAT", "nom_action": "EXAIL", "isin": "FR0000062671",
            "quantite": 5, "prix_unitaire": 48.85, "montant_brut": 244.25,
            "commission": 0.0, "frais": 0.0, "montant_net": 244.25,
        },
        {
            "date_execution": "2025-05-23", "heure_execution": "10:00:00",
            "sens": "VENTE", "nom_action": "EXAIL", "isin": "FR0000062671",
            "quantite": 15, "prix_unitaire": 62.0, "montant_brut": 930.0,
            "commission": 4.65, "frais": 0.0, "montant_net": 925.35,
        },
    ]

    matcher = TradeMatcher()
    trades = matcher.match_trades(executions)

    # FIFO: premier achat (10 @ 54) vendu en premier, puis second (5 @ 48.85)
    assert len(trades) == 2
    assert all(t["statut"] == "CLOTURE" for t in trades)

    # Premier trade FIFO : 10 @ 54 -> 62
    t1 = [t for t in trades if t["quantite"] == 10][0]
    assert t1["prix_achat"] == 54.0
    assert t1["prix_vente"] == 62.0

    # Second trade FIFO : 5 @ 48.85 -> 62
    t2 = [t for t in trades if t["quantite"] == 5][0]
    assert t2["prix_achat"] == 48.85
    assert t2["prix_vente"] == 62.0


def test_rendement_net_includes_frais():
    """Le rendement net % doit prendre en compte les frais totaux."""
    executions = [
        {
            "date_execution": "2025-03-15", "heure_execution": "09:00:00",
            "sens": "ACHAT", "nom_action": "TEST", "isin": "FR0000000001",
            "quantite": 100, "prix_unitaire": 10.0, "montant_brut": 1000.0,
            "commission": 5.0, "frais": 0.0, "montant_net": 1005.0,
        },
        {
            "date_execution": "2025-03-20", "heure_execution": "09:00:00",
            "sens": "VENTE", "nom_action": "TEST", "isin": "FR0000000001",
            "quantite": 100, "prix_unitaire": 10.50, "montant_brut": 1050.0,
            "commission": 5.0, "frais": 0.0, "montant_net": 1045.0,
        },
    ]

    matcher = TradeMatcher()
    trades = matcher.match_trades(executions)
    trade = trades[0]

    # Rendement brut = (10.50 - 10.0) / 10.0 * 100 = 5%
    assert trade["rendement_brut_pct"] == 5.0

    # Frais totaux = 5 + 5 = 10
    assert trade["frais_totaux"] == 10.0

    # Rendement net = (1050 - 1000 - 10) / 1000 * 100 = 4%
    assert trade["rendement_net_pct"] == 4.0
