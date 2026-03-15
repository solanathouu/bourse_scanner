"""Bot Telegram interactif — boutons PRIS/PASSE + chatbot de feedback.

Ecoute les reponses de Nicolas aux signaux et les enregistre en base.
Le LLM analyse les commentaires libres pour en extraire des infos structurees.
Supporte les trades manuels (actions non proposees par le bot).
"""

import json
import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from src.core.database import Database

load_dotenv()

try:
    from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.constants import ParseMode
    from telegram.ext import (
        Application, CallbackQueryHandler, MessageHandler, filters,
    )
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False

ANALYZE_PROMPT = """Tu es l'assistant d'un trader francais nomme Nicolas (PEA, swing court terme).
Nicolas t'envoie un message dans Telegram. Analyse-le et extrais les informations.

Tickers connus: {known_tickers}

Message de Nicolas: "{text}"

Determine:
1. Est-ce un ACHAT (il a achete ou veut acheter), une VENTE (TP, SL, il a vendu),
   un COMMENTAIRE sur un trade existant, ou un message NON_TRADING (hors sujet)?
2. Quel ticker est concerne? (utilise le format .PA, ex: SAN.PA)
3. Extrais les details si mentionnes.

Reponds UNIQUEMENT avec un JSON:
{{
    "type": "ACHAT" ou "VENTE" ou "COMMENTAIRE" ou "NON_TRADING",
    "ticker": "SAN.PA" ou null,
    "company_name": "SANOFI" ou null,
    "entry_price": float ou null,
    "exit_price": float ou null,
    "exit_type": "TP" ou "SL" ou null,
    "reason": "la raison donnee par Nicolas" ou null,
    "summary": "resume en 1 phrase"
}}"""


class TelegramInteractive:
    """Bot Telegram interactif avec boutons et chatbot."""

    def __init__(self, db: Database):
        self.db = db
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

    def build_signal_keyboard(self, signal_id: int) -> "InlineKeyboardMarkup":
        """Construit les boutons PRIS/PASSE pour un signal."""
        keyboard = [
            [
                InlineKeyboardButton("PRIS", callback_data=f"pris_{signal_id}"),
                InlineKeyboardButton("PASSE", callback_data=f"passe_{signal_id}"),
            ]
        ]
        return InlineKeyboardMarkup(keyboard)

    async def send_signal_with_buttons(self, bot: Bot, message: str,
                                        signal_id: int) -> bool:
        """Envoie un signal avec boutons PRIS/PASSE."""
        try:
            keyboard = self.build_signal_keyboard(signal_id)
            await bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard,
            )
            return True
        except Exception as e:
            logger.error(f"Telegram interactif erreur: {e}")
            return False

    async def handle_button(self, update: Update, context) -> None:
        """Gere les clics sur PRIS/PASSE."""
        query = update.callback_query
        await query.answer()

        data = query.data
        if not data:
            return

        parts = data.split("_", 1)
        if len(parts) != 2:
            return

        action_type = parts[0].upper()
        try:
            signal_id = int(parts[1])
        except ValueError:
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if action_type == "PRIS":
            self.db.insert_user_action({
                "signal_id": signal_id,
                "action": "PRIS",
                "created_at": now,
            })
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text(
                "Trade enregistre. A quel prix as-tu achete ? "
                "(Reponds avec le prix, ex: 76.50)"
            )
            context.user_data["pending_price"] = signal_id
            logger.info(f"Action PRIS enregistree pour signal #{signal_id}")

        elif action_type == "PASSE":
            self.db.insert_user_action({
                "signal_id": signal_id,
                "action": "PASSE",
                "created_at": now,
            })
            await query.edit_message_reply_markup(reply_markup=None)
            await query.message.reply_text("OK, signal ignore.")
            logger.info(f"Action PASSE enregistree pour signal #{signal_id}")

    async def handle_message(self, update: Update, context) -> None:
        """Gere les messages libres — prix, trades manuels, commentaires."""
        if not update.message or not update.message.text:
            return

        text = update.message.text.strip()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Si on attend un prix d'entree
        pending = context.user_data.get("pending_price")
        if pending:
            try:
                price = float(text.replace(",", ".").replace("€", "").strip())
                self.db.insert_user_action({
                    "signal_id": pending,
                    "action": "PRIS",
                    "entry_price": price,
                    "created_at": now,
                })
                context.user_data.pop("pending_price", None)
                await update.message.reply_text(
                    f"Prix d'achat enregistre: {price:.2f} EUR. "
                    f"Tu pourras me dire quand tu TP ou SL."
                )
                logger.info(f"Prix {price} enregistre pour signal #{pending}")
                return
            except ValueError:
                context.user_data.pop("pending_price", None)

        # Analyser avec le LLM
        analysis = self._analyze_message(text)
        if not analysis:
            await update.message.reply_text(
                "Je n'ai pas compris. Tu peux me dire :\n"
                "- \"J'ai achete SANOFI a 76€ parce que bons resultats\"\n"
                "- \"J'ai vendu SOITEC a 58€, TP atteint\"\n"
                "- Ou repondre directement a un signal"
            )
            return

        msg_type = analysis.get("type", "NON_TRADING")
        ticker = analysis.get("ticker")
        summary = analysis.get("summary", "")

        if msg_type == "NON_TRADING" or not ticker:
            await update.message.reply_text(
                f"Compris, mais je n'ai pas identifie de trade. {summary}"
            )
            return

        if msg_type == "ACHAT":
            await self._handle_buy(analysis, text, now, update)
        elif msg_type == "VENTE":
            await self._handle_sell(analysis, text, now, update)
        elif msg_type == "COMMENTAIRE":
            await self._handle_comment(analysis, text, now, update)

    async def _handle_buy(self, analysis: dict, text: str,
                           now: str, update: Update) -> None:
        """Traite un message d'achat — signal existant ou trade manuel."""
        ticker = analysis["ticker"]
        entry_price = analysis.get("entry_price")
        reason = analysis.get("reason") or ""

        # Chercher un signal existant pour ce ticker
        existing_signal = self.db.get_latest_signal_by_ticker(ticker)

        if existing_signal:
            # Lier au signal existant
            self.db.insert_user_action({
                "signal_id": existing_signal["id"],
                "action": "PRIS",
                "entry_price": entry_price,
                "user_comment": text,
                "llm_analysis": json.dumps(analysis, ensure_ascii=False),
                "created_at": now,
            })
            price_str = f" a {entry_price:.2f} EUR" if entry_price else ""
            await update.message.reply_text(
                f"Achat {ticker}{price_str} enregistre "
                f"(lie au signal #{existing_signal['id']}).\n"
                f"Raison: {reason[:100]}"
            )
        else:
            # Trade manuel — creer un signal manuel en base
            today = datetime.now().strftime("%Y-%m-%d")
            self.db.insert_signal({
                "ticker": ticker,
                "date": today,
                "score": 0.0,
                "signal_price": entry_price,
                "catalyst_type": "MANUAL",
                "catalyst_news_title": reason[:200] if reason else "Trade manuel",
                "sent_at": now,
                "model_version": "manual",
            })
            new_signal = self.db.get_latest_signal_by_ticker(ticker)
            signal_id = new_signal["id"] if new_signal else 0

            self.db.insert_user_action({
                "signal_id": signal_id,
                "action": "PRIS",
                "entry_price": entry_price,
                "user_comment": text,
                "llm_analysis": json.dumps(analysis, ensure_ascii=False),
                "created_at": now,
            })

            price_str = f" a {entry_price:.2f} EUR" if entry_price else ""
            await update.message.reply_text(
                f"Trade MANUEL enregistre: achat {ticker}{price_str}.\n"
                f"Raison: {reason[:100]}\n"
                f"Je suivrai ce trade et le reviewerai a J+3."
            )

        logger.info(f"Achat {ticker} enregistre (prix={entry_price}, raison={reason[:50]})")

    async def _handle_sell(self, analysis: dict, text: str,
                            now: str, update: Update) -> None:
        """Traite un message de vente — TP ou SL."""
        ticker = analysis["ticker"]
        exit_price = analysis.get("exit_price")
        exit_type = analysis.get("exit_type") or "VENTE"

        signal = self.db.get_latest_signal_by_ticker(ticker)
        if not signal:
            await update.message.reply_text(
                f"Je n'ai pas de trade ouvert sur {ticker}."
            )
            return

        self.db.insert_user_action({
            "signal_id": signal["id"],
            "action": exit_type,
            "exit_price": exit_price,
            "exit_type": exit_type,
            "user_comment": text,
            "llm_analysis": json.dumps(analysis, ensure_ascii=False),
            "created_at": now,
        })

        price_str = f" a {exit_price:.2f} EUR" if exit_price else ""
        await update.message.reply_text(
            f"{exit_type} {ticker}{price_str} enregistre. "
            f"{analysis.get('summary', '')}"
        )
        logger.info(f"{exit_type} {ticker} enregistre (prix={exit_price})")

    async def _handle_comment(self, analysis: dict, text: str,
                               now: str, update: Update) -> None:
        """Traite un commentaire sur un trade."""
        ticker = analysis["ticker"]
        signal = self.db.get_latest_signal_by_ticker(ticker)

        if signal:
            self.db.insert_user_action({
                "signal_id": signal["id"],
                "action": "COMMENT",
                "user_comment": text,
                "llm_analysis": json.dumps(analysis, ensure_ascii=False),
                "created_at": now,
            })
            await update.message.reply_text(
                f"Commentaire sur {ticker} enregistre. "
                f"{analysis.get('summary', '')}"
            )
        else:
            await update.message.reply_text(
                f"Commentaire note, mais pas de trade {ticker} en cours."
            )

    def _analyze_message(self, text: str) -> dict | None:
        """Analyse un message libre avec Gemini."""
        try:
            from google import genai
            from google.genai import types
            from src.data_collection.ticker_mapper import TICKER_MAP

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return None

            known = ", ".join(f"{name} ({ticker})" for name, ticker in TICKER_MAP.items())
            prompt = ANALYZE_PROMPT.format(known_tickers=known, text=text)

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1, max_output_tokens=300,
                ),
            )

            cleaned = response.text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                cleaned = "\n".join(lines).strip()

            return json.loads(cleaned)

        except Exception as e:
            logger.error(f"Analyse message echouee: {e}")
            return None

    def create_application(self) -> "Application":
        """Cree l'application Telegram avec handlers."""
        if not HAS_TELEGRAM:
            raise ImportError("python-telegram-bot requis")

        app = Application.builder().token(self.token).build()
        app.add_handler(CallbackQueryHandler(self.handle_button))
        app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND, self.handle_message,
        ))
        return app
