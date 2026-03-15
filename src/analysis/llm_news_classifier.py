"""Classification des news par type de catalyseur via Gemini Flash Lite.

Remplace le NewsClassifier regex par une comprehension semantique des news.
Pour chaque batch de news d'un ticker, le LLM determine le type de catalyseur,
la confiance, et si la news est vraiment pertinente pour le ticker.
"""

import json
import os
import time

from dotenv import load_dotenv
from loguru import logger
from google import genai
from google.genai import types

from src.core.database import Database

load_dotenv()

VALID_CATALYST_TYPES = [
    "EARNINGS", "FDA_REGULATORY", "UPGRADE", "DOWNGRADE", "CONTRACT",
    "DIVIDEND", "RESTRUCTURING", "INSIDER", "SECTOR_MACRO",
    "OTHER_POSITIVE", "OTHER_NEGATIVE", "TECHNICAL", "UNKNOWN",
]

CLASSIFY_PROMPT = """Tu es un analyste financier specialise dans le marche boursier francais (PEA).

Tu analyses les news pour le ticker {ticker} (entreprise: {company_name}).

Pour CHAQUE news ci-dessous, determine:
1. Le type de catalyseur boursier parmi: {valid_types}
2. Ta confiance dans cette classification (0.0 a 1.0)
3. Si cette news est VRAIMENT pertinente pour {company_name} (0.0 a 1.0)
   - 0.0 = la news mentionne un mot similaire par hasard (ex: "Monroe Capital" n'est pas ALCAP)
   - 1.0 = la news parle directement de cette entreprise
4. Une explication courte (1 phrase max) de pourquoi c'est un signal d'achat ou de vente
5. Un numero de groupe (event_group): les news qui parlent du MEME evenement ont le meme numero.
   Ex: 3 articles sur "resultats T2 Sanofi" = meme groupe. Articles differents = groupes differents.

News a analyser:
{news_list}

Reponds UNIQUEMENT avec un JSON array:
[{{"news_index": 1, "catalyst_type": "EARNINGS", "confidence": 0.85, "relevance": 0.95, "explanation": "Resultats T3 au-dessus du consensus", "event_group": 1}}]"""

DELAY_BETWEEN_CALLS = 0.5  # secondes


class LLMNewsClassifier:
    """Classifie les news par type de catalyseur via Gemini."""

    def __init__(self, db: Database, model: str = "gemini-2.5-flash-lite"):
        self.db = db
        self.model = model

    def _get_client(self) -> genai.Client:
        """Cree un client Gemini avec la cle du .env."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY manquante dans .env")
        return genai.Client(api_key=api_key)

    def _build_prompt(self, ticker: str, company_name: str,
                       news_list: list[dict]) -> str:
        """Construit le prompt avec toutes les news d'un ticker."""
        lines = []
        for i, news in enumerate(news_list, 1):
            title = news.get("title", "")
            desc = (news.get("description") or "")[:200]
            date = (news.get("published_at") or "")[:10]
            source = news.get("source", "inconnue")
            lines.append(f"{i}. [{date}] \"{title}\" (source: {source})\n   {desc}")

        return CLASSIFY_PROMPT.format(
            ticker=ticker,
            company_name=company_name,
            valid_types=", ".join(VALID_CATALYST_TYPES),
            news_list="\n".join(lines),
        )

    def _parse_response(self, response_text: str,
                         news_list: list[dict]) -> list[dict]:
        """Parse la reponse JSON du LLM. Retourne les classifications."""
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error(f"Reponse LLM non parsable: {response_text[:200]}")
            return []

        if not isinstance(data, list):
            logger.error(f"Reponse LLM n'est pas un array: {type(data)}")
            return []

        results = []
        for item in data:
            idx = item.get("news_index", 0) - 1
            if idx < 0 or idx >= len(news_list):
                continue

            cat_type = item.get("catalyst_type", "UNKNOWN")
            if cat_type not in VALID_CATALYST_TYPES:
                cat_type = "UNKNOWN"

            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
            relevance = max(0.0, min(1.0, float(item.get("relevance", 0.0))))
            explanation = (item.get("explanation") or "")[:300]

            event_group = int(item.get("event_group", 0))

            results.append({
                "news_id": news_list[idx]["id"],
                "news_index": idx,
                "catalyst_type": cat_type,
                "confidence": confidence,
                "relevance": relevance,
                "explanation": explanation,
                "event_group": event_group,
            })

        return results

    def classify_batch_for_ticker(self, ticker: str, company_name: str,
                                   news_list: list[dict]) -> list[dict]:
        """Classifie un batch de news pour un ticker via LLM.

        Returns:
            Liste de dicts avec news_id, catalyst_type, confidence, relevance.
        """
        if not news_list:
            return []

        prompt = self._build_prompt(ticker, company_name, news_list)
        client = self._get_client()

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=4096,
            ),
        )

        return self._parse_response(response.text, news_list)

    def classify_and_cache(self, ticker: str, company_name: str,
                            news_list: list[dict]) -> list[dict]:
        """Classifie les news non classifiees et cache en BDD.

        Les news deja classifiees (llm_catalyst_type IS NOT NULL) sont skippees.
        Retourne la liste complete avec les colonnes LLM remplies.
        """
        unclassified = [n for n in news_list
                        if n.get("llm_catalyst_type") is None]

        if unclassified:
            results = self.classify_batch_for_ticker(
                ticker, company_name, unclassified,
            )
            # Sauvegarder en base
            results_by_id = {r["news_id"]: r for r in results}
            for news in unclassified:
                r = results_by_id.get(news["id"])
                if r:
                    self.db.update_news_llm_classification(
                        news["id"], r["catalyst_type"],
                        r["confidence"], r["relevance"],
                        r.get("event_group"),
                    )
                    news["llm_catalyst_type"] = r["catalyst_type"]
                    news["llm_catalyst_confidence"] = r["confidence"]
                    news["llm_relevance_score"] = r["relevance"]
                    news["llm_explanation"] = r["explanation"]
                    news["event_group_id"] = r.get("event_group")

            time.sleep(DELAY_BETWEEN_CALLS)

        return news_list

    def summarize_for_realtime(self, classified_news: list[dict]) -> dict:
        """Resume les catalyseurs LLM pour les features temps reel.

        Filtre les news non pertinentes (relevance < 0.3), deduplique par
        event_group (1 evenement = 1 signal, meme s'il y a 4 articles),
        et retourne le catalyseur le plus confiant.
        """
        relevant = [
            n for n in classified_news
            if (n.get("llm_relevance_score") or 0) >= 0.3
            and n.get("llm_catalyst_type") is not None
        ]

        if not relevant:
            return {
                "catalyst_type": "TECHNICAL",
                "catalyst_confidence": 0.0,
                "has_clear_catalyst": 0,
                "nb_unique_events": 0,
                "best_news_title": None,
                "best_explanation": None,
            }

        # Deduplication par event_group: garder la meilleure news par groupe
        groups: dict[int, dict] = {}
        ungrouped = []
        for n in relevant:
            gid = n.get("event_group_id") or n.get("event_group")
            if gid and gid > 0:
                if gid not in groups or (n.get("llm_catalyst_confidence") or 0) > (groups[gid].get("llm_catalyst_confidence") or 0):
                    groups[gid] = n
            else:
                ungrouped.append(n)

        unique_events = list(groups.values()) + ungrouped
        best = max(unique_events, key=lambda n: n.get("llm_catalyst_confidence") or 0)
        cat_type = best.get("llm_catalyst_type", "UNKNOWN")

        return {
            "catalyst_type": cat_type,
            "catalyst_confidence": best.get("llm_catalyst_confidence", 0.5),
            "has_clear_catalyst": 1 if cat_type not in ("TECHNICAL", "UNKNOWN") else 0,
            "nb_unique_events": len(unique_events),
            "best_news_title": best.get("title"),
            "best_explanation": best.get("llm_explanation"),
        }
