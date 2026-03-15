"""
twilio_webhook.py  ── Production-grade alternative to Selenium scraping
────────────────────────────────────────────────────────────────────────
Uses the official Twilio WhatsApp Sandbox API instead of Selenium.
This is more reliable and suitable for production deployments.

Setup:
  1. Create a Twilio account → https://www.twilio.com
  2. Activate the WhatsApp Sandbox in the Twilio Console.
  3. Set env vars: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_NUMBER
  4. Expose this server with:  ngrok http 5000
  5. Set the webhook URL in Twilio Console to: https://<ngrok-url>/webhook

Run:
    pip install flask twilio
    python twilio_webhook.py
"""

import os
import logging
from flask import Flask, request
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

from ai_analyzer import TransformerAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Twilio credentials (set as environment variables) ─────────────────────────
ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "YOUR_ACCOUNT_SID")
AUTH_TOKEN  = os.environ.get("TWILIO_AUTH_TOKEN",  "YOUR_AUTH_TOKEN")
FROM_NUMBER = os.environ.get("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")  # sandbox default

twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)

# ── Load AI Analyzer once at startup ─────────────────────────────────────────
logger.info("Initialising TransformerAnalyzer …")
analyzer = TransformerAnalyzer()
logger.info("Analyzer ready ✓")


@app.route("/webhook", methods=["POST"])
def webhook():
    incoming_msg = request.values.get("Body", "").strip()
    sender       = request.values.get("From", "")

    logger.info(f"📨 From {sender}: {incoming_msg[:80]}")

    # Run AI pipeline
    analysis = analyzer.analyze(incoming_msg)
    reply    = analysis.generated_reply

    logger.info(
        f"   Intent={analysis.intent} | Sentiment={analysis.sentiment} "
        f"({analysis.sentiment_score:.2f})"
    )
    logger.info(f"   ✉ Reply: {reply[:80]}")

    # Build TwiML response
    resp = MessagingResponse()
    resp.message(reply)
    return str(resp), 200, {"Content-Type": "application/xml"}


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "models": "loaded"}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
