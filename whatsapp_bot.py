"""
whatsapp_bot.py
WhatsApp Auto-Reply Bot — Selenium + Transformer AI
─────────────────────────────────────────────────────
How it works:
  1. Opens WhatsApp Web in Chrome (you scan the QR code once).
  2. Continuously monitors unread messages in the left panel.
  3. For each new message, runs the TransformerAnalyzer pipeline.
  4. Sends the AI-generated reply automatically.

Run:
    python whatsapp_bot.py
"""

import time
import logging
import re
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    StaleElementReferenceException,
)

from ai_analyzer import TransformerAnalyzer
from config import BotConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ── XPath / CSS Selectors (WhatsApp Web — update if WA changes their DOM) ────

SEL = {
    "unread_badge"   : '//span[@data-testid="icon-unread-count"]',
    "chat_list_item" : '//div[@data-testid="cell-frame-container"]',
    "message_in"     : '//div[contains(@class,"message-in")]//span[@data-testid="msg-container"]',
    "last_msg_text"  : './/span[@class="selectable-text copyable-text"]',
    "reply_box"      : '//div[@data-testid="conversation-compose-box-input"]',
    "send_button"    : '//button[@data-testid="compose-btn-send"]',
    "qr_canvas"      : '//canvas[@aria-label="Scan me!"]',
}


class WhatsAppBot:
    def __init__(self, config: BotConfig):
        self.cfg      = config
        self.analyzer = TransformerAnalyzer()
        self.driver   = self._init_driver()
        self.replied  : dict[str, str] = {}   # chat_id → last replied message

    # ── Driver Setup ──────────────────────────────────────────────────────────

    def _init_driver(self) -> webdriver.Chrome:
        opts = Options()
        opts.add_argument(f"--user-data-dir={self.cfg.chrome_profile_path}")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        if self.cfg.headless:
            # Note: headless mode requires pre-authenticated profile
            opts.add_argument("--headless=new")
        driver = webdriver.Chrome(options=opts)
        driver.maximize_window()
        return driver

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def run(self):
        logger.info("Navigating to WhatsApp Web …")
        self.driver.get("https://web.whatsapp.com")
        self._wait_for_login()
        logger.info("✅ Logged in — bot is active.")

        while True:
            try:
                self._scan_and_reply()
            except Exception as exc:
                logger.error(f"Loop error: {exc}", exc_info=True)
            time.sleep(self.cfg.poll_interval_seconds)

    # ── Login / QR Wait ───────────────────────────────────────────────────────

    def _wait_for_login(self):
        logger.info("Waiting for QR scan / session restore …")
        WebDriverWait(self.driver, 120).until(
            EC.presence_of_element_located((By.XPATH, '//div[@data-testid="chat-list"]'))
        )
        logger.info("WhatsApp Web session active.")

    # ── Scan Unread Chats ─────────────────────────────────────────────────────

    def _scan_and_reply(self):
        unread_chats = self.driver.find_elements(By.XPATH, SEL["unread_badge"])
        if not unread_chats:
            return

        # Click the first unread chat's parent container
        for badge in unread_chats[:self.cfg.max_concurrent_replies]:
            try:
                chat_container = badge.find_element(By.XPATH, "./ancestor::div[@data-testid='cell-frame-container']")
                chat_id = chat_container.get_attribute("data-id") or str(hash(chat_container.text))
                chat_container.click()
                time.sleep(1.2)

                last_msg = self._get_last_incoming_message()
                if not last_msg:
                    continue

                # Avoid duplicate replies
                if self.replied.get(chat_id) == last_msg:
                    continue

                # Skip messages from ourselves
                if self._is_own_message():
                    continue

                # Filter by allowed/blocked senders (optional)
                if not self._is_allowed_sender(chat_container.text):
                    continue

                logger.info(f"📨 New message [{chat_id[:8]}…]: {last_msg[:60]}")
                analysis = self.analyzer.analyze(last_msg)
                logger.info(
                    f"   Intent={analysis.intent} | Sentiment={analysis.sentiment} "
                    f"({analysis.sentiment_score:.2f}) | Entities={[e['word'] for e in analysis.entities]}"
                )

                reply = self._build_final_reply(analysis)
                self._send_message(reply)
                self.replied[chat_id] = last_msg

                logger.info(f"   ✉ Replied: {reply[:80]}")
                self._log_interaction(chat_id, last_msg, analysis, reply)

            except StaleElementReferenceException:
                continue
            except Exception as e:
                logger.warning(f"Error handling chat: {e}")

    # ── Message Reading ───────────────────────────────────────────────────────

    def _get_last_incoming_message(self) -> str | None:
        try:
            msgs = self.driver.find_elements(By.XPATH, SEL["message_in"])
            if not msgs:
                return None
            last = msgs[-1]
            span = last.find_elements(By.XPATH, SEL["last_msg_text"])
            return span[-1].text.strip() if span else None
        except Exception:
            return None

    def _is_own_message(self) -> bool:
        try:
            own = self.driver.find_elements(By.XPATH, '//div[contains(@class,"message-out")]')
            in_ = self.driver.find_elements(By.XPATH, '//div[contains(@class,"message-in")]')
            if not own or not in_:
                return False
            # Compare last message timestamp; if "out" is newer, we already replied
            return False  # Simplified; refine with timestamp comparison if needed
        except Exception:
            return False

    def _is_allowed_sender(self, chat_text: str) -> bool:
        if self.cfg.allowed_contacts:
            return any(c.lower() in chat_text.lower() for c in self.cfg.allowed_contacts)
        if self.cfg.blocked_contacts:
            return not any(c.lower() in chat_text.lower() for c in self.cfg.blocked_contacts)
        return True

    # ── Reply Building ────────────────────────────────────────────────────────

    def _build_final_reply(self, analysis) -> str:
        """
        Override the raw AI reply with a custom template for certain intents
        if configured; otherwise use the generated reply as-is.
        """
        templates = self.cfg.intent_templates
        if analysis.intent in templates:
            return templates[analysis.intent]
        # Prefix for low-confidence replies
        if analysis.confidence < 0.4:
            return f"Thanks for your message! {analysis.generated_reply}"
        return analysis.generated_reply

    # ── Message Sending ───────────────────────────────────────────────────────

    def _send_message(self, text: str):
        try:
            box = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, SEL["reply_box"]))
            )
            box.click()
            # Send line-by-line to preserve newlines
            for line in text.split("\n"):
                box.send_keys(line)
                box.send_keys(Keys.SHIFT + Keys.ENTER)
            box.send_keys(Keys.ENTER)
            time.sleep(0.8)
        except TimeoutException:
            logger.error("Reply box not found — DOM may have changed.")

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_interaction(self, chat_id: str, message: str, analysis, reply: str):
        with open("interactions.log", "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(
                f"[{ts}] CHAT={chat_id[:12]} | MSG={message[:80]!r} | "
                f"INTENT={analysis.intent} | SENT={analysis.sentiment} | "
                f"REPLY={reply[:80]!r}\n"
            )

    def stop(self):
        logger.info("Shutting down bot …")
        self.driver.quit()


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import BotConfig
    cfg = BotConfig()
    bot = WhatsAppBot(cfg)
    try:
        bot.run()
    except KeyboardInterrupt:
        bot.stop()
        logger.info("Bot stopped by user.")
