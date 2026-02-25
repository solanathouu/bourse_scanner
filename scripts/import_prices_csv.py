"""Import de prix historiques depuis un fichier CSV (fallback pour tickers delistes).

Usage:
    uv run python scripts/import_prices_csv.py data/prices_2crsi.csv --ticker 2CRSI.PA
"""

import argparse
import csv
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger

from src.core.database import Database

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "trades.db")

# Formats de date courants
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d"]


def _detect_separator(filepath: str) -> str:
    """Detecte le separateur CSV (virgule, point-virgule, tab)."""
    with open(filepath, "r", encoding="utf-8-sig") as f:
        first_line = f.readline()
    for sep in [";", ",", "\t"]:
        if sep in first_line:
            return sep
    return ","


def _parse_date(date_str: str) -> str | None:
    """Parse une date dans differents formats. Retourne YYYY-MM-DD ou None."""
    date_str = date_str.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _parse_float(value: str) -> float | None:
    """Parse un float avec gestion du separateur decimal , ou ."""
    value = value.strip().replace(",", ".")
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Import prix historiques CSV")
    parser.add_argument("csv_file", help="Chemin du fichier CSV")
    parser.add_argument("--ticker", required=True, help="Ticker Yahoo (ex: 2CRSI.PA)")
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Fichier introuvable: {args.csv_file}")
        sys.exit(1)

    db = Database(DB_PATH)
    db.init_db()

    separator = _detect_separator(args.csv_file)
    print(f"Separateur detecte: '{separator}'")

    prices = []
    errors = 0

    with open(args.csv_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=separator)
        for row in reader:
            # Detecter la colonne de date
            date_col = None
            for col in ["Date", "date", "DATE", "Fecha"]:
                if col in row:
                    date_col = col
                    break
            if not date_col:
                date_col = list(row.keys())[0]

            date_str = _parse_date(row[date_col])
            if not date_str:
                errors += 1
                continue

            # Detecter les colonnes OHLCV
            price = {
                "ticker": args.ticker,
                "date": date_str,
                "open": _parse_float(row.get("Open", row.get("open", row.get("Ouverture", "0")))),
                "high": _parse_float(row.get("High", row.get("high", row.get("Haut", "0")))),
                "low": _parse_float(row.get("Low", row.get("low", row.get("Bas", "0")))),
                "close": _parse_float(row.get("Close", row.get("close", row.get("Cloture", row.get("Dernier", "0"))))),
                "volume": int(_parse_float(row.get("Volume", row.get("volume", "0"))) or 0),
            }
            prices.append(price)

    if not prices:
        print("Aucun prix valide trouve dans le CSV")
        sys.exit(1)

    count_before = db.count_prices()
    db.insert_prices_batch(prices)
    count_after = db.count_prices()

    print(f"Lignes CSV lues: {len(prices)}")
    print(f"Erreurs de parsing: {errors}")
    print(f"Prix en base avant: {count_before}")
    print(f"Prix en base apres: {count_after}")
    print(f"Nouveaux prix inseres: {count_after - count_before}")


if __name__ == "__main__":
    main()
