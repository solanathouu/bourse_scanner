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

        # Table trade_catalyseurs (etape 3)
        conn.executescript("""
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
            CREATE INDEX IF NOT EXISTS idx_catalyseurs_trade
                ON trade_catalyseurs(trade_id);
            CREATE INDEX IF NOT EXISTS idx_catalyseurs_news
                ON trade_catalyseurs(news_id);
        """)

        # Table trade_analyses_llm (etape 4 bis — analyse LLM)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trade_analyses_llm (
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
            CREATE INDEX IF NOT EXISTS idx_analyses_trade
                ON trade_analyses_llm(trade_id);
        """)

        # Table fundamentals (etape 4ter — donnees fondamentales yfinance)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS fundamentals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                pe_ratio REAL,
                pb_ratio REAL,
                market_cap INTEGER,
                dividend_yield REAL,
                target_price REAL,
                analyst_count INTEGER,
                recommendation TEXT,
                earnings_date TEXT,
                UNIQUE(ticker, date)
            );
            CREATE INDEX IF NOT EXISTS idx_fundamentals_ticker_date
                ON fundamentals(ticker, date);
        """)

        # Table signals (etape 5 — signaux temps reel)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                score REAL NOT NULL,
                catalyst_type TEXT,
                catalyst_news_title TEXT,
                features_json TEXT,
                sent_at TEXT,
                UNIQUE(ticker, date)
            );
            CREATE INDEX IF NOT EXISTS idx_signals_ticker_date
                ON signals(ticker, date);
        """)

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

    # --- Catalyseurs ---

    def insert_catalyseur(self, catalyseur: dict):
        """Insere un catalyseur. Ignore les doublons (trade_id+news_id)."""
        conn = self._connect()
        conn.execute("""
            INSERT OR IGNORE INTO trade_catalyseurs
                (trade_id, news_id, score_pertinence, distance_jours, match_texte)
            VALUES
                (:trade_id, :news_id, :score_pertinence, :distance_jours, :match_texte)
        """, catalyseur)
        conn.commit()
        conn.close()

    def insert_catalyseurs_batch(self, catalyseurs: list[dict]):
        """Insere plusieurs catalyseurs en batch. Ignore les doublons."""
        conn = self._connect()
        conn.executemany("""
            INSERT OR IGNORE INTO trade_catalyseurs
                (trade_id, news_id, score_pertinence, distance_jours, match_texte)
            VALUES
                (:trade_id, :news_id, :score_pertinence, :distance_jours, :match_texte)
        """, catalyseurs)
        conn.commit()
        conn.close()
        logger.info(f"{len(catalyseurs)} catalyseurs inseres en batch")

    def get_catalyseurs(self, trade_id: int) -> list[dict]:
        """Recupere les catalyseurs pour un trade donne."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM trade_catalyseurs WHERE trade_id = ? ORDER BY distance_jours",
            (trade_id,),
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def count_catalyseurs(self) -> int:
        """Compte le nombre total de catalyseurs."""
        conn = self._connect()
        count = conn.execute(
            "SELECT COUNT(*) FROM trade_catalyseurs"
        ).fetchone()[0]
        conn.close()
        return count

    def clear_catalyseurs(self):
        """Vide la table trade_catalyseurs."""
        conn = self._connect()
        conn.execute("DELETE FROM trade_catalyseurs")
        conn.commit()
        conn.close()
        logger.info("Table trade_catalyseurs videe")

    # --- Trade Analyses LLM ---

    def insert_trade_analysis(self, analysis: dict):
        """Insere ou remplace une analyse LLM pour un trade."""
        conn = self._connect()
        conn.execute("""
            INSERT OR REPLACE INTO trade_analyses_llm
                (trade_id, primary_news_id, catalyst_type, catalyst_summary,
                 catalyst_confidence, news_sentiment, buy_reason, sell_reason,
                 trade_quality, model_used, analyzed_at)
            VALUES
                (:trade_id, :primary_news_id, :catalyst_type, :catalyst_summary,
                 :catalyst_confidence, :news_sentiment, :buy_reason, :sell_reason,
                 :trade_quality, :model_used, :analyzed_at)
        """, analysis)
        conn.commit()
        conn.close()

    def get_trade_analysis(self, trade_id: int) -> dict | None:
        """Recupere l'analyse LLM pour un trade. None si absente."""
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM trade_analyses_llm WHERE trade_id = ?",
            (trade_id,),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_all_trade_analyses(self) -> list[dict]:
        """Recupere toutes les analyses LLM."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM trade_analyses_llm ORDER BY trade_id"
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def count_trade_analyses(self) -> int:
        """Compte le nombre d'analyses LLM."""
        conn = self._connect()
        count = conn.execute(
            "SELECT COUNT(*) FROM trade_analyses_llm"
        ).fetchone()[0]
        conn.close()
        return count

    def get_news_without_sentiment(self) -> list[dict]:
        """Recupere les news sans sentiment (NULL)."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM news WHERE sentiment IS NULL ORDER BY id"
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def update_news_sentiment(self, news_id: int, sentiment: float):
        """Met a jour le sentiment d'une news."""
        conn = self._connect()
        conn.execute(
            "UPDATE news SET sentiment = ? WHERE id = ?",
            (sentiment, news_id),
        )
        conn.commit()
        conn.close()

    def get_news_in_window(self, ticker: str, date_start: str, date_end: str) -> list[dict]:
        """Recupere les news pour un ticker dans une fenetre de dates."""
        conn = self._connect()
        rows = conn.execute("""
            SELECT * FROM news
            WHERE ticker = ?
              AND date(published_at) BETWEEN date(?) AND date(?)
            ORDER BY published_at
        """, (ticker, date_start, date_end)).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    # --- Fundamentals ---

    def insert_fundamental(self, fundamental: dict):
        """Insere des donnees fondamentales. Ignore les doublons (ticker+date)."""
        conn = self._connect()
        conn.execute("""
            INSERT OR IGNORE INTO fundamentals
                (ticker, date, pe_ratio, pb_ratio, market_cap,
                 dividend_yield, target_price, analyst_count,
                 recommendation, earnings_date)
            VALUES
                (:ticker, :date, :pe_ratio, :pb_ratio, :market_cap,
                 :dividend_yield, :target_price, :analyst_count,
                 :recommendation, :earnings_date)
        """, fundamental)
        conn.commit()
        conn.close()

    def get_fundamentals(self, ticker: str) -> list[dict]:
        """Recupere les fondamentaux pour un ticker, tries par date."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM fundamentals WHERE ticker = ? ORDER BY date",
            (ticker,),
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_fundamental_at_date(self, ticker: str, date: str) -> dict | None:
        """Recupere les fondamentaux les plus recents pour un ticker a une date."""
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM fundamentals WHERE ticker = ? AND date <= ? ORDER BY date DESC LIMIT 1",
            (ticker, date),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def count_fundamentals(self) -> int:
        """Compte le nombre total de fondamentaux."""
        conn = self._connect()
        count = conn.execute("SELECT COUNT(*) FROM fundamentals").fetchone()[0]
        conn.close()
        return count

    # --- Signals ---

    def insert_signal(self, signal: dict):
        """Insere un signal. Ignore les doublons (ticker+date)."""
        conn = self._connect()
        conn.execute("""
            INSERT OR IGNORE INTO signals
                (ticker, date, score, catalyst_type,
                 catalyst_news_title, features_json, sent_at)
            VALUES
                (:ticker, :date, :score, :catalyst_type,
                 :catalyst_news_title, :features_json, :sent_at)
        """, {
            "catalyst_type": None, "catalyst_news_title": None,
            "features_json": None, "sent_at": None,
            **signal,
        })
        conn.commit()
        conn.close()

    def get_signals(self, ticker: str | None = None) -> list[dict]:
        """Recupere les signaux, optionnellement filtres par ticker."""
        conn = self._connect()
        if ticker:
            rows = conn.execute(
                "SELECT * FROM signals WHERE ticker = ? ORDER BY date DESC",
                (ticker,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM signals ORDER BY date DESC"
            ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_latest_signal(self, ticker: str) -> dict | None:
        """Recupere le signal le plus recent pour un ticker."""
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM signals WHERE ticker = ? ORDER BY date DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def count_signals(self) -> int:
        """Compte le nombre total de signaux."""
        conn = self._connect()
        count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        conn.close()
        return count
