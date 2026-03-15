"""
config.py
All configurable settings for the WhatsApp Auto-Reply Bot.
Edit this file to customise behaviour without touching bot logic.
"""

import os
from dataclasses import dataclass, field


@dataclass
class BotConfig:
    # ── Chrome Profile ────────────────────────────────────────────────────────
    # WhatsApp Web remembers your session via the Chrome profile.
    # Set this to a persistent directory so you only scan QR once.
    chrome_profile_path: str = os.path.join(os.path.expanduser("~"), ".whatsapp_bot_profile")

    # ── Bot Behaviour ─────────────────────────────────────────────────────────
    headless: bool = False                  # Set True after first QR scan
    poll_interval_seconds: float = 3.0     # How often to check for new msgs
    max_concurrent_replies: int = 5        # Max chats to reply per poll cycle

    # ── Contact Filters ───────────────────────────────────────────────────────
    # Leave both empty to reply to ALL contacts.
    # allowed_contacts: only reply to these names/numbers (partial match).
    # blocked_contacts: never reply to these names/numbers.
    allowed_contacts: list[str] = field(default_factory=list)
    blocked_contacts: list[str] = field(default_factory=lambda: ["Business", "Spam"])

    # ── Intent Templates ──────────────────────────────────────────────────────
    # Override the AI-generated reply for specific intents.
    # Leave an intent out to let the AI handle it freely.
    intent_templates: dict[str, str] = field(default_factory=lambda: {
        "greeting": (
            "👋 Hi there! Thanks for reaching out.\n"
            "I'm currently away but will get back to you shortly.\n"
            "How can I help you today?"
        ),
        "farewell": (
            "Thanks for chatting! Have a wonderful day 😊\n"
            "Feel free to message anytime!"
        ),
    })

    # ── Model Overrides (advanced) ────────────────────────────────────────────
    # You can swap in larger / fine-tuned models for better quality.
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    ner_model: str       = "dbmdz/bert-large-cased-finetuned-conll03-english"
    gen_model: str       = "google/flan-t5-small"   # swap to flan-t5-base for richer replies
