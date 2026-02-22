"""Script exploratoire : affiche le texte brut extrait d'un PDF SG."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdfplumber


def explore_pdf(pdf_path: str):
    """Extrait et affiche le texte de chaque page d'un PDF."""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            print(f"=== PAGE {i+1} ===")
            text = page.extract_text()
            print(text)
            print()

            # Afficher aussi les tableaux détectés
            tables = page.extract_tables()
            if tables:
                print(f"--- TABLEAUX PAGE {i+1} ---")
                for j, table in enumerate(tables):
                    print(f"Table {j+1}:")
                    for row in table:
                        print(row)
                    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/explore_pdf.py <chemin_pdf>")
        sys.exit(1)
    explore_pdf(sys.argv[1])
