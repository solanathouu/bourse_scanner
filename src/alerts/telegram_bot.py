"""Envoi de messages Telegram via l'API Bot.

Utilise python-telegram-bot pour envoyer les alertes
dans un groupe Telegram.
"""

import asyncio

from loguru import logger

try:
    from telegram import Bot
    from telegram.constants import ParseMode
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False


class TelegramBot:
    """Envoie des messages dans un groupe Telegram."""

    def __init__(self, token: str, chat_id: str):
        if not token or not chat_id:
            raise ValueError("TELEGRAM_BOT_TOKEN et TELEGRAM_CHAT_ID requis")
        self.token = token
        self.chat_id = chat_id
        if HAS_TELEGRAM:
            self.bot = Bot(token=token)
        else:
            self.bot = None

    async def send_alert(self, message: str) -> bool:
        """Envoie un message Telegram (async).

        Returns:
            True si envoye avec succes, False sinon.
        """
        if not self.bot:
            logger.error("python-telegram-bot non installe")
            return False

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.HTML,
            )
            logger.info(f"Telegram: message envoye ({len(message)} chars)")
            return True
        except Exception as e:
            logger.error(f"Telegram: erreur d'envoi: {e}")
            return False

    def send_alert_sync(self, message: str) -> bool:
        """Envoie un message Telegram (sync, pour APScheduler).

        Cree ou reutilise un event loop pour l'envoi async.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Si dans un loop existant, creer une task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run, self.send_alert(message)
                    ).result(timeout=30)
                return result
            else:
                return loop.run_until_complete(self.send_alert(message))
        except RuntimeError:
            return asyncio.run(self.send_alert(message))
