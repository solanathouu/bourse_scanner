"""Bot Telegram interactif — boutons PRIS/PASSE + chatbot de feedback.

Ecoute les reponses de Nicolas aux signaux et les enregistre en base.
Le LLM analyse les commentaires libres pour en extraire des infos structurees.
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
            # Stocker le signal_id en attente de prix
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
        """Gere les messages libres — prix d'entree ou commentaires."""
        if not update.message or not update.message.text:
            return

        text = update.message.text.strip()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Si on attend un prix d'entree
        pending = context.user_data.get("pending_price")
        if pending:
            try:
                price = float(text.replace(",", ".").replace("€", "").strip())
                # Mettre a jour l'action avec le prix
                action = self.db.get_user_action(pending)
                if action:
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
                pass  # Pas un prix, traiter comme commentaire

        # Commentaire libre — analyser avec LLM
        analysis = self._analyze_comment(text)
        if analysis:
            signal_id = analysis.get("signal_id")
            if signal_id:
                self.db.insert_user_action({
                    "signal_id": signal_id,
                    "action": analysis.get("action", "COMMENT"),
                    "entry_price": analysis.get("entry_price"),
                    "exit_price": analysis.get("exit_price"),
                    "exit_type": analysis.get("exit_type"),
                    "user_comment": text,
                    "llm_analysis": json.dumps(analysis, ensure_ascii=False),
                    "created_at": now,
                })
                await update.message.reply_text(
                    f"Compris ! {analysis.get('summary', 'Enregistre.')}"
                )
                logger.info(f"Commentaire analyse pour signal #{signal_id}: {analysis}")
                return

        # Pas de signal identifie — repondre simplement
        await update.message.reply_text(
            "Je n'ai pas identifie de signal associe. "
            "Reponds directement a un signal ou mentionne le ticker."
        )

    def _analyze_comment(self, text: str) -> dict | None:
        """Analyse un commentaire libre avec Gemini pour extraire les infos."""
        try:
            from google import genai
            from google.genai import types

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return None

            # Chercher le dernier signal mentionne (par ticker)
            signal = self._find_signal_in_text(text)
            if not signal:
                return None

            client = genai.Client(api_key=api_key)
            prompt = f"""Analyse ce message d'un trader a propos d'un signal sur {signal['ticker']}:

Message: "{text}"

Extrais les informations en JSON:
- "action": "TP" si take profit, "SL" si stop loss, "COMMENT" sinon
- "entry_price": prix d'achat mentionne (null si absent)
- "exit_price": prix de vente/sortie mentionne (null si absent)
- "exit_type": "TP" ou "SL" ou null
- "summary": resume en 1 phrase de ce que le trader dit

Reponds UNIQUEMENT avec le JSON."""

            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1, max_output_tokens=200,
                ),
            )

            cleaned = response.text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                cleaned = "\n".join(lines).strip()

            result = json.loads(cleaned)
            result["signal_id"] = signal["id"]
            return result

        except Exception as e:
            logger.error(f"Analyse commentaire echouee: {e}")
            return None

    def _find_signal_in_text(self, text: str) -> dict | None:
        """Cherche un ticker mentionne dans le texte et retourne le dernier signal."""
        text_upper = text.upper()
        # Chercher les tickers connus
        from src.data_collection.ticker_mapper import TICKER_MAP
        for name, ticker in TICKER_MAP.items():
            if name.upper() in text_upper or ticker.replace(".PA", "") in text_upper:
                signal = self.db.get_latest_signal_by_ticker(ticker)
                if signal:
                    return signal
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
