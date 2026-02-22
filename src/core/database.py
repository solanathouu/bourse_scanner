"""Couche d'accès à la base de données SQLite."""

import sqlite3
import os
from loguru import logger


class Database:
    """Gère la connexion et les opérations sur la base SQLite."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        # Créer le dossier parent si nécessaire
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        """Crée une connexion avec row_factory pour accès par nom de colonne."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def init_db(self):
        """Crée les tables si elles n'existent pas."""
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date_execution DATE NOT NULL,
                heure_execution TIME NOT NULL,
                sens TEXT NOT NULL CHECK (sens IN ('ACHAT', 'VENTE')),
                nom_action TEXT NOT NULL,
                isin TEXT,
                quantite INTEGER NOT NULL,
                prix_unitaire REAL NOT NULL,
                montant_brut REAL NOT NULL,
                commission REAL DEFAULT 0,
                frais REAL DEFAULT 0,
                montant_net REAL,
                fichier_source TEXT
            );

            CREATE TABLE IF NOT EXISTS trades_complets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                isin TEXT,
                nom_action TEXT NOT NULL,
                date_achat DATETIME NOT NULL,
                date_vente DATETIME,
                prix_achat REAL NOT NULL,
                prix_vente REAL,
                quantite INTEGER NOT NULL,
                rendement_brut_pct REAL,
                rendement_net_pct REAL,
                duree_jours REAL,
                frais_totaux REAL,
                statut TEXT DEFAULT 'OUVERT' CHECK (statut IN ('OUVERT', 'CLOTURE'))
            );

            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                UNIQUE(ticker, date)
            );

            CREATE TABLE IF NOT EXISTS news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                title TEXT NOT NULL,
                source TEXT,
                url TEXT UNIQUE,
                published_at TEXT,
                description TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_executions_date
                ON executions(date_execution);
            CREATE INDEX IF NOT EXISTS idx_executions_isin
                ON executions(isin);
            CREATE INDEX IF NOT EXISTS idx_trades_isin
                ON trades_complets(isin);
            CREATE INDEX IF NOT EXISTS idx_prices_ticker_date
                ON prices(ticker, date);
            CREATE INDEX IF NOT EXISTS idx_news_ticker
                ON news(ticker);
        """)
        conn.commit()
        # Migration: ajouter les colonnes sentiment et source_api si absentes
        self._migrate_news_columns(conn)
        conn.close()
        logger.info(f"Base de données initialisée: {self.db_path}")

    def _migrate_news_columns(self, conn: sqlite3.Connection):
        """Ajoute les colonnes sentiment et source_api a la table news si absentes."""
        existing = [row[1] for row in conn.execute("PRAGMA table_info(news)").fetchall()]
        if "sentiment" not in existing:
            conn.execute("ALTER TABLE news ADD COLUMN sentiment REAL")
            logger.info("Migration: colonne 'sentiment' ajoutee a la table news")
        if "source_api" not in existing:
            conn.execute("ALTER TABLE news ADD COLUMN source_api TEXT DEFAULT 'gnews'")
            logger.info("Migration: colonne 'source_api' ajoutee a la table news")
        conn.commit()

    def insert_execution(self, execution: dict):
        """Insère une exécution (un PDF parsé) dans la base."""
        conn = self._connect()
        conn.execute("""
            INSERT INTO executions
                (date_execution, heure_execution, sens, nom_action, isin,
                 quantite, prix_unitaire, montant_brut, commission, frais,
                 montant_net, fichier_source)
            VALUES
                (:date_execution, :heure_execution, :sens, :nom_action, :isin,
                 :quantite, :prix_unitaire, :montant_brut, :commission, :frais,
                 :montant_net, :fichier_source)
        """, execution)
        conn.commit()
        conn.close()

    def insert_executions_batch(self, executions: list[dict]):
        """Insère plusieurs exécutions en une transaction."""
        conn = self._connect()
        conn.executemany("""
            INSERT INTO executions
                (date_execution, heure_execution, sens, nom_action, isin,
                 quantite, prix_unitaire, montant_brut, commission, frais,
                 montant_net, fichier_source)
            VALUES
                (:date_execution, :heure_execution, :sens, :nom_action, :isin,
                 :quantite, :prix_unitaire, :montant_brut, :commission, :frais,
                 :montant_net, :fichier_source)
        """, executions)
        conn.commit()
        conn.close()
        logger.info(f"{len(executions)} exécutions insérées en batch")

    def insert_trade_complet(self, trade: dict):
        """Insère un trade complet (achat->vente reconstitué)."""
        conn = self._connect()
        conn.execute("""
            INSERT INTO trades_complets
                (isin, nom_action, date_achat, date_vente, prix_achat,
                 prix_vente, quantite, rendement_brut_pct, rendement_net_pct,
                 duree_jours, frais_totaux, statut)
            VALUES
                (:isin, :nom_action, :date_achat, :date_vente, :prix_achat,
                 :prix_vente, :quantite, :rendement_brut_pct, :rendement_net_pct,
                 :duree_jours, :frais_totaux, :statut)
        """, trade)
        conn.commit()
        conn.close()

    def insert_trades_batch(self, trades: list[dict]):
        """Insère plusieurs trades complets en une transaction."""
        conn = self._connect()
        conn.executemany("""
            INSERT INTO trades_complets
                (isin, nom_action, date_achat, date_vente, prix_achat,
                 prix_vente, quantite, rendement_brut_pct, rendement_net_pct,
                 duree_jours, frais_totaux, statut)
            VALUES
                (:isin, :nom_action, :date_achat, :date_vente, :prix_achat,
                 :prix_vente, :quantite, :rendement_brut_pct, :rendement_net_pct,
                 :duree_jours, :frais_totaux, :statut)
        """, trades)
        conn.commit()
        conn.close()
        logger.info(f"{len(trades)} trades complets insérés en batch")

    def get_all_executions(self) -> list[dict]:
        """Récupère toutes les exécutions."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM executions ORDER BY date_execution, heure_execution"
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_executions_by_isin(self, isin: str) -> list[dict]:
        """Récupère les exécutions pour un ISIN donné, triées chronologiquement."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM executions WHERE isin = ? ORDER BY date_execution, heure_execution",
            (isin,),
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_all_trades(self) -> list[dict]:
        """Récupère tous les trades complets."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM trades_complets ORDER BY date_achat"
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_distinct_isins(self) -> list[str]:
        """Récupère la liste des ISIN distincts dans les exécutions."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT DISTINCT isin FROM executions WHERE isin IS NOT NULL"
        ).fetchall()
        conn.close()
        return [row["isin"] for row in rows]

    def count_executions(self) -> int:
        """Compte le nombre total d'exécutions."""
        conn = self._connect()
        count = conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0]
        conn.close()
        return count

    def count_trades(self) -> int:
        """Compte le nombre total de trades complets."""
        conn = self._connect()
        count = conn.execute("SELECT COUNT(*) FROM trades_complets").fetchone()[0]
        conn.close()
        return count

    # --- Prices ---

    def insert_price(self, price: dict):
        """Insere un prix OHLCV. Ignore les doublons (ticker+date)."""
        conn = self._connect()
        conn.execute("""
            INSERT OR IGNORE INTO prices
                (ticker, date, open, high, low, close, volume)
            VALUES
                (:ticker, :date, :open, :high, :low, :close, :volume)
        """, price)
        conn.commit()
        conn.close()

    def insert_prices_batch(self, prices: list[dict]):
        """Insere plusieurs prix en batch. Ignore les doublons."""
        conn = self._connect()
        conn.executemany("""
            INSERT OR IGNORE INTO prices
                (ticker, date, open, high, low, close, volume)
            VALUES
                (:ticker, :date, :open, :high, :low, :close, :volume)
        """, prices)
        conn.commit()
        conn.close()
        logger.info(f"{len(prices)} prix inseres en batch")

    def get_prices(self, ticker: str) -> list[dict]:
        """Recupere les prix pour un ticker, tries par date."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM prices WHERE ticker = ? ORDER BY date",
            (ticker,),
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def count_prices(self) -> int:
        """Compte le nombre total de prix."""
        conn = self._connect()
        count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        conn.close()
        return count

    # --- News ---

    def insert_news(self, news: dict):
        """Insere une news. Ignore les doublons (meme URL)."""
        conn = self._connect()
        conn.execute("""
            INSERT OR IGNORE INTO news
                (ticker, title, source, url, published_at, description,
                 sentiment, source_api)
            VALUES
                (:ticker, :title, :source, :url, :published_at, :description,
                 :sentiment, :source_api)
        """, {
            "sentiment": None, "source_api": "gnews",
            **news,
        })
        conn.commit()
        conn.close()

    def insert_news_batch(self, news_list: list[dict]):
        """Insere plusieurs news en batch. Ignore les doublons."""
        enriched = [
            {"sentiment": None, "source_api": "gnews", **n}
            for n in news_list
        ]
        conn = self._connect()
        conn.executemany("""
            INSERT OR IGNORE INTO news
                (ticker, title, source, url, published_at, description,
                 sentiment, source_api)
            VALUES
                (:ticker, :title, :source, :url, :published_at, :description,
                 :sentiment, :source_api)
        """, enriched)
        conn.commit()
        conn.close()
        logger.info(f"{len(news_list)} news inserees en batch")

    def get_news(self, ticker: str) -> list[dict]:
        """Recupere les news pour un ticker, triees par date."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM news WHERE ticker = ? ORDER BY published_at",
            (ticker,),
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def count_news(self) -> int:
        """Compte le nombre total de news."""
        conn = self._connect()
        count = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
        conn.close()
        return count
