"""Tests pour le bot Telegram."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.alerts.telegram_bot import TelegramBot


class TestTelegramBot:
    """Tests pour TelegramBot."""

    def test_init_requires_token_and_chat_id(self):
        """Le constructeur leve ValueError si token ou chat_id manquant."""
        with pytest.raises(ValueError, match="requis"):
            TelegramBot("", "12345")
        with pytest.raises(ValueError, match="requis"):
            TelegramBot("token", "")

    def test_init_with_valid_params(self):
        """Le constructeur accepte des parametres valides."""
        bot = TelegramBot("fake_token", "-12345")
        assert bot.token == "fake_token"
        assert bot.chat_id == "-12345"

    def test_send_alert_success(self):
        """send_alert retourne True en cas de succes."""
        bot = TelegramBot("fake_token", "-12345")
        bot.bot = MagicMock()
        bot.bot.send_message = AsyncMock(return_value=True)

        result = asyncio.run(bot.send_alert("Test message"))
        assert result is True
        bot.bot.send_message.assert_called_once()

    def test_send_alert_failure(self):
        """send_alert retourne False en cas d'erreur."""
        bot = TelegramBot("fake_token", "-12345")
        bot.bot = MagicMock()
        bot.bot.send_message = AsyncMock(side_effect=Exception("Network error"))

        result = asyncio.run(bot.send_alert("Test message"))
        assert result is False
