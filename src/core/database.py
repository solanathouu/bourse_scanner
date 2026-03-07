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
        # Migration: ajouter signal_price si absente
        self._migrate_signals_columns(conn)

        # Table signal_reviews (etape 6 — feedback loop)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS signal_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER NOT NULL UNIQUE,
                ticker TEXT NOT NULL,
                signal_date TEXT NOT NULL,
                signal_price REAL NOT NULL,
                review_date TEXT NOT NULL,
                review_price REAL NOT NULL,
                performance_pct REAL NOT NULL,
                outcome TEXT NOT NULL CHECK (outcome IN ('WIN', 'LOSS', 'NEUTRAL')),
                failure_reason TEXT,
                catalyst_type TEXT,
                features_json TEXT,
                reviewed_at TEXT NOT NULL,
                FOREIGN KEY (signal_id) REFERENCES signals(id)
            );
            CREATE INDEX IF NOT EXISTS idx_signal_reviews_ticker
                ON signal_reviews(ticker);
            CREATE INDEX IF NOT EXISTS idx_signal_reviews_date
                ON signal_reviews(review_date);
        """)

        # Table filter_rules (etape 6 — regles de filtrage adaptatives)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS filter_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_type TEXT NOT NULL,
                rule_json TEXT NOT NULL,
                win_rate REAL,
                sample_size INTEGER,
                created_at TEXT NOT NULL,
                active INTEGER DEFAULT 1
            );
        """)

        # Table model_versions (etape 6 — versioning modeles)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                file_path TEXT NOT NULL,
                trained_at TEXT NOT NULL,
                training_signals INTEGER DEFAULT 0,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                f1 REAL,
                is_active INTEGER DEFAULT 0,
                notes TEXT
            );
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

    def _migrate_signals_columns(self, conn: sqlite3.Connection):
        """Ajoute la colonne signal_price a la table signals si absente."""
        existing = [row[1] for row in conn.execute("PRAGMA table_info(signals)").fetchall()]
        if "signal_price" not in existing:
            conn.execute("ALTER TABLE signals ADD COLUMN signal_price REAL")
            logger.info("Migration: colonne 'signal_price' ajoutee a la table signals")
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
                 catalyst_news_title, features_json, sent_at, signal_price)
            VALUES
                (:ticker, :date, :score, :catalyst_type,
                 :catalyst_news_title, :features_json, :sent_at, :signal_price)
        """, {
            "catalyst_type": None, "catalyst_news_title": None,
            "features_json": None, "sent_at": None, "signal_price": None,
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

    # --- Signal Reviews ---

    def insert_signal_review(self, review: dict):
        """Insere une review de signal. Ignore si signal_id deja reviewe."""
        conn = self._connect()
        conn.execute("""
            INSERT OR IGNORE INTO signal_reviews
                (signal_id, ticker, signal_date, signal_price, review_date,
                 review_price, performance_pct, outcome, failure_reason,
                 catalyst_type, features_json, reviewed_at)
            VALUES
                (:signal_id, :ticker, :signal_date, :signal_price, :review_date,
                 :review_price, :performance_pct, :outcome, :failure_reason,
                 :catalyst_type, :features_json, :reviewed_at)
        """, review)
        conn.commit()
        conn.close()

    def get_signal_reviews(self, ticker: str | None = None) -> list[dict]:
        """Recupere les reviews, optionnellement filtrees par ticker."""
        conn = self._connect()
        if ticker:
            rows = conn.execute(
                "SELECT * FROM signal_reviews WHERE ticker = ? ORDER BY signal_date DESC",
                (ticker,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM signal_reviews ORDER BY signal_date DESC"
            ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_pending_signal_reviews(self, current_date: str) -> list[dict]:
        """Recupere les signaux envoyes il y a 3+ jours sans review."""
        conn = self._connect()
        rows = conn.execute("""
            SELECT s.* FROM signals s
            LEFT JOIN signal_reviews sr ON sr.signal_id = s.id
            WHERE sr.id IS NULL
              AND s.sent_at IS NOT NULL
              AND julianday(?) - julianday(s.date) >= 3
            ORDER BY s.date
        """, (current_date,)).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_reviews_in_period(self, date_start: str, date_end: str) -> list[dict]:
        """Recupere les reviews dont le signal_date est dans la periode."""
        conn = self._connect()
        rows = conn.execute("""
            SELECT * FROM signal_reviews
            WHERE signal_date BETWEEN ? AND ?
            ORDER BY signal_date
        """, (date_start, date_end)).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_review_stats(self) -> dict:
        """Retourne les stats globales des reviews."""
        conn = self._connect()
        rows = conn.execute("""
            SELECT outcome, COUNT(*) as cnt
            FROM signal_reviews
            GROUP BY outcome
        """).fetchall()
        conn.close()
        stats = {"total": 0, "wins": 0, "losses": 0, "neutrals": 0}
        for row in rows:
            outcome = row["outcome"]
            count = row["cnt"]
            stats["total"] += count
            if outcome == "WIN":
                stats["wins"] = count
            elif outcome == "LOSS":
                stats["losses"] = count
            elif outcome == "NEUTRAL":
                stats["neutrals"] = count
        return stats

    # --- Filter Rules ---

    def insert_filter_rule(self, rule: dict):
        """Insere une regle de filtrage."""
        conn = self._connect()
        conn.execute("""
            INSERT INTO filter_rules
                (rule_type, rule_json, win_rate, sample_size, created_at, active)
            VALUES
                (:rule_type, :rule_json, :win_rate, :sample_size, :created_at, :active)
        """, {
            "win_rate": None, "sample_size": None, "active": 1,
            **rule,
        })
        conn.commit()
        conn.close()

    def get_active_filter_rules(self) -> list[dict]:
        """Recupere les regles de filtrage actives."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM filter_rules WHERE active = 1 ORDER BY id"
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def deactivate_filter_rule(self, rule_id: int):
        """Desactive une regle de filtrage."""
        conn = self._connect()
        conn.execute(
            "UPDATE filter_rules SET active = 0 WHERE id = ?",
            (rule_id,),
        )
        conn.commit()
        conn.close()

    # --- Model Versions ---

    def insert_model_version(self, version: dict):
        """Insere une version de modele."""
        conn = self._connect()
        conn.execute("""
            INSERT INTO model_versions
                (version, file_path, trained_at, training_signals,
                 accuracy, precision_score, recall, f1, is_active, notes)
            VALUES
                (:version, :file_path, :trained_at, :training_signals,
                 :accuracy, :precision_score, :recall, :f1, :is_active, :notes)
        """, {
            "training_signals": 0, "accuracy": None, "precision_score": None,
            "recall": None, "f1": None, "is_active": 0, "notes": None,
            **version,
        })
        conn.commit()
        conn.close()

    def get_active_model_version(self) -> dict | None:
        """Recupere la version de modele active."""
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM model_versions WHERE is_active = 1 LIMIT 1"
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_all_model_versions(self) -> list[dict]:
        """Recupere toutes les versions de modeles."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM model_versions ORDER BY id"
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def set_active_model(self, version_id: int):
        """Active un modele et desactive tous les autres."""
        conn = self._connect()
        conn.execute("UPDATE model_versions SET is_active = 0")
        conn.execute(
            "UPDATE model_versions SET is_active = 1 WHERE id = ?",
            (version_id,),
        )
        conn.commit()
        conn.close()
