# 🤖 WhatsApp Auto-Reply Bot — AI Powered with Transformers

A Python bot that automatically replies to WhatsApp messages using
HuggingFace Transformer models running **fully locally** (no OpenAI key needed).

---

## 🧠 AI Pipeline (per incoming message)

```
Incoming Message
      │
      ▼
┌─────────────────────────────────────────────────┐
│  1. Sentiment Analysis  (DistilBERT SST-2)       │
│     → POSITIVE / NEGATIVE / NEUTRAL              │
├─────────────────────────────────────────────────┤
│  2. Intent Classification  (rule-enhanced NLP)   │
│     → greeting / question / complaint /           │
│       inquiry / farewell / other                 │
├─────────────────────────────────────────────────┤
│  3. Named Entity Recognition  (BERT CoNLL-03)    │
│     → extracts names, orgs, locations            │
├─────────────────────────────────────────────────┤
│  4. Reply Generation  (FLAN-T5 small/base)        │
│     → context-aware, intent-specific reply       │
└─────────────────────────────────────────────────┘
      │
      ▼
 Auto-Send via Selenium (WhatsApp Web)
 or Twilio Webhook (production)
```

---

## 📁 Project Structure

```
whatsapp_ai_bot/
├── ai_analyzer.py       ← Transformer AI pipeline (core logic)
├── whatsapp_bot.py      ← Selenium-based WhatsApp Web bot
├── twilio_webhook.py    ← Production webhook via Twilio API
├── config.py            ← All settings in one place
├── requirements.txt     ← Python dependencies
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Choose your approach

#### Option A — Selenium (WhatsApp Web, no API key)
```bash
python whatsapp_bot.py
# A Chrome window opens → scan QR code with your phone
# Bot starts auto-replying to unread messages
```

#### Option B — Twilio Webhook (production-grade)
```bash
# Set credentials
export TWILIO_ACCOUNT_SID="ACxxxxxxxxxxxx"
export TWILIO_AUTH_TOKEN="your_auth_token"

# Expose locally with ngrok
ngrok http 5000

# Run server
python twilio_webhook.py

# Paste the ngrok URL into Twilio Console → WhatsApp Sandbox → Webhook
```

---

## ⚙️ Configuration (`config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `chrome_profile_path` | `~/.whatsapp_bot_profile` | Saves login session |
| `headless` | `False` | Run without visible browser (after first QR scan) |
| `poll_interval_seconds` | `3.0` | How often to check for new messages |
| `allowed_contacts` | `[]` (all) | Only reply to these contacts |
| `blocked_contacts` | `["Business", "Spam"]` | Never reply to these |
| `intent_templates` | greeting + farewell | Fixed replies for specific intents |
| `gen_model` | `flan-t5-small` | Swap to `flan-t5-base` for richer replies |

---

## 🔄 Models Used

| Task | Model | Size |
|------|-------|------|
| Sentiment | `distilbert-base-uncased-finetuned-sst-2-english` | ~260 MB |
| NER | `dbmdz/bert-large-cased-finetuned-conll03-english` | ~1.3 GB |
| Generation | `google/flan-t5-small` | ~300 MB |

> **Tip:** Swap `flan-t5-small` → `flan-t5-base` (580 MB) or `flan-t5-large` (780 MB) for significantly better reply quality.

---

## 📋 Logs

- `bot.log` — runtime info / errors
- `interactions.log` — full message + analysis + reply history

---

## ⚠️ Important Notes

1. **WhatsApp ToS**: Automated messaging may violate WhatsApp's Terms of Service for personal accounts. Use responsibly or use the official Business API via Twilio for commercial use.
2. **First run**: Models are downloaded from HuggingFace Hub (~1-2 GB total). Cached locally after first run.
3. **Selenium selectors**: WhatsApp Web periodically changes its DOM. If the bot stops finding elements, update the XPath selectors in `whatsapp_bot.py → SEL dict`.
