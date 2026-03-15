"""Dashboard FastAPI read-only pour le PEA Scanner Bot."""

import json
import os
import sqlite3
from datetime import datetime

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "..", "..", "data", "trades.db")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="PEA Scanner Dashboard", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def _get_conn() -> sqlite3.Connection:
    """Ouvre une connexion read-only vers la BDD."""
    uri = f"file:{os.path.abspath(DB_PATH)}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _rows_to_dicts(rows: list) -> list[dict]:
    """Convertit une liste de sqlite3.Row en list[dict]."""
    return [dict(r) for r in rows]


def _iso_week(date_str: str) -> str:
    """Retourne 'YYYY-Wnn' pour une date string YYYY-MM-DD."""
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        iso = dt.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    except (ValueError, TypeError):
        return "unknown"


@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Page principale: KPIs, evolution win rate, derniers signaux."""
    conn = _get_conn()
    try:
        total_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        review_rows = conn.execute(
            "SELECT outcome, COUNT(*) as cnt FROM signal_reviews GROUP BY outcome"
        ).fetchall()
        wins, losses, neutrals, total_reviews = 0, 0, 0, 0
        for r in review_rows:
            total_reviews += r["cnt"]
            if r["outcome"] == "WIN":
                wins = r["cnt"]
            elif r["outcome"] == "LOSS":
                losses = r["cnt"]
            elif r["outcome"] == "NEUTRAL":
                neutrals = r["cnt"]
        win_rate = round(wins / total_reviews * 100, 1) if total_reviews > 0 else 0.0

        model_row = conn.execute(
            "SELECT * FROM model_versions WHERE is_active = 1 LIMIT 1"
        ).fetchone()
        active_model = dict(model_row) if model_row else None

        weekly_rows = conn.execute("""
            SELECT strftime('%Y-W%W', signal_date) as week,
                   COUNT(*) as total,
                   SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins
            FROM signal_reviews
            GROUP BY week ORDER BY week
        """).fetchall()
        weekly_winrate = [
            {"week": r["week"], "total": r["total"], "wins": r["wins"],
             "win_rate": round(r["wins"] / r["total"] * 100, 1) if r["total"] > 0 else 0.0}
            for r in weekly_rows
        ]

        signals_rows = conn.execute("""
            SELECT s.id, s.ticker, s.date, s.score, s.catalyst_type,
                   s.catalyst_news_title, s.signal_price, s.sent_at,
                   s.model_version, sr.outcome, sr.performance_pct
            FROM signals s
            LEFT JOIN signal_reviews sr ON sr.signal_id = s.id
            ORDER BY s.date DESC LIMIT 20
        """).fetchall()
        last_signals = _rows_to_dicts(signals_rows)
    finally:
        conn.close()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "total_signals": total_signals,
        "total_reviews": total_reviews,
        "wins": wins, "losses": losses, "neutrals": neutrals,
        "win_rate": win_rate,
        "active_model": active_model,
        "weekly_winrate": weekly_winrate,
        "last_signals": last_signals,
    })


@app.get("/news", response_class=HTMLResponse)
async def news_feed(request: Request, ticker: str = Query(default=None)):
    """Liste des 200 dernieres news, filtre optionnel par ticker."""
    conn = _get_conn()
    try:
        tickers_rows = conn.execute(
            "SELECT DISTINCT ticker FROM news ORDER BY ticker"
        ).fetchall()
        tickers_list = [r["ticker"] for r in tickers_rows]

        cols = ("id, ticker, title, source, published_at, sentiment, "
                "llm_catalyst_type, llm_catalyst_confidence, llm_relevance_score")
        if ticker:
            news_rows = conn.execute(
                f"SELECT {cols} FROM news WHERE ticker = ? "
                "ORDER BY published_at DESC LIMIT 200", (ticker,)
            ).fetchall()
        else:
            news_rows = conn.execute(
                f"SELECT {cols} FROM news ORDER BY published_at DESC LIMIT 200"
            ).fetchall()
        news_list = _rows_to_dicts(news_rows)
    finally:
        conn.close()

    return templates.TemplateResponse("news.html", {
        "request": request,
        "tickers": tickers_list,
        "selected_ticker": ticker,
        "news_list": news_list,
    })


@app.get("/news/{news_id}", response_class=HTMLResponse)
async def news_detail(request: Request, news_id: int):
    """Detail d'une news par id."""
    conn = _get_conn()
    try:
        row = conn.execute("SELECT * FROM news WHERE id = ?", (news_id,)).fetchone()
        news = dict(row) if row else None
    finally:
        conn.close()

    return templates.TemplateResponse("news_detail.html", {
        "request": request, "news": news,
    })


@app.get("/signal/{signal_id}", response_class=HTMLResponse)
async def signal_detail(request: Request, signal_id: int):
    """Detail d'un signal avec features et review."""
    conn = _get_conn()
    try:
        sig_row = conn.execute(
            "SELECT * FROM signals WHERE id = ?", (signal_id,)
        ).fetchone()
        signal = dict(sig_row) if sig_row else None

        review = None
        if signal:
            rev_row = conn.execute(
                "SELECT * FROM signal_reviews WHERE signal_id = ?", (signal_id,)
            ).fetchone()
            review = dict(rev_row) if rev_row else None

        features = {}
        if signal and signal.get("features_json"):
            try:
                features = json.loads(signal["features_json"])
            except (json.JSONDecodeError, TypeError):
                features = {}

        tv_link = None
        if signal:
            ticker_raw = signal["ticker"].replace(".PA", "")
            tv_link = f"https://fr.tradingview.com/chart/?symbol=EURONEXT:{ticker_raw}"
    finally:
        conn.close()

    return templates.TemplateResponse("signal.html", {
        "request": request, "signal": signal,
        "review": review, "features": features, "tradingview_url": tv_link,
    })


@app.get("/portfolio", response_class=HTMLResponse)
async def portfolio(request: Request, filter: str = Query(default="all"),
                    version: str = Query(default="all")):
    """Portefeuille virtuel: performance cumulee des signaux reviewes."""
    conn = _get_conn()
    try:
        cat_rows = conn.execute(
            "SELECT DISTINCT catalyst_type FROM signal_reviews "
            "WHERE catalyst_type IS NOT NULL ORDER BY catalyst_type"
        ).fetchall()
        catalyst_types = [r["catalyst_type"] for r in cat_rows]

        version_rows = conn.execute(
            "SELECT DISTINCT model_version FROM signals "
            "WHERE model_version IS NOT NULL ORDER BY model_version"
        ).fetchall()
        model_versions = [r["model_version"] for r in version_rows]

        conditions = []
        params = []
        if filter == "confirmed":
            conditions.append("sr.outcome = 'WIN'")
        elif filter not in ("all", "confirmed", None):
            conditions.append("sr.catalyst_type = ?")
            params.append(filter)
        if version != "all":
            conditions.append("s.model_version = ?")
            params.append(version)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT sr.signal_id, sr.ticker, sr.signal_date, sr.signal_price,
                   sr.review_price, sr.performance_pct, sr.outcome, sr.catalyst_type,
                   s.model_version
            FROM signal_reviews sr
            JOIN signals s ON sr.signal_id = s.id
            {where} ORDER BY sr.signal_date
        """
        trades_rows = conn.execute(query, params).fetchall()
        trades = _rows_to_dicts(trades_rows)
    finally:
        conn.close()

    # Simulation portefeuille: 1000 EUR par trade
    TRADE_AMOUNT = 1000.0
    capital = 0.0  # P&L cumule
    for t in trades:
        perf = t.get("performance_pct", 0.0) or 0.0
        pnl = TRADE_AMOUNT * perf / 100.0
        capital += pnl
        t["pnl_eur"] = round(pnl, 2)
        t["capital_cumul"] = round(capital, 2)

    total_trades = len(trades)
    total_wins = sum(1 for t in trades if t["outcome"] == "WIN")
    avg_perf = sum(t.get("performance_pct", 0) or 0 for t in trades) / total_trades if total_trades > 0 else 0
    win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0

    return templates.TemplateResponse("portfolio.html", {
        "request": request, "trades": trades,
        "total_trades": total_trades, "total_wins": total_wins,
        "win_rate": round(win_rate, 1),
        "avg_perf": round(avg_perf, 2),
        "capital_cumul": round(capital, 2),
        "trade_amount": int(TRADE_AMOUNT),
        "selected_filter": filter, "catalyst_types": catalyst_types,
        "selected_version": version, "model_versions": model_versions,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)
