"""
ai_analyzer.py
Transformer-based message analyzer for WhatsApp Auto-Reply Bot.
Uses HuggingFace Transformers for:
  - Sentiment Analysis
  - Intent Classification
  - Named Entity Recognition (NER)
  - Auto Response Generation
"""

import logging
from dataclasses import dataclass
from typing import Optional
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class MessageAnalysis:
    original_text: str
    sentiment: str          # POSITIVE / NEGATIVE / NEUTRAL
    sentiment_score: float
    intent: str             # greeting / question / complaint / farewell / other
    entities: list[dict]    # [{"entity": "...", "word": "..."}]
    generated_reply: str
    confidence: float


# ── Transformer Pipelines ─────────────────────────────────────────────────────

class TransformerAnalyzer:
    """
    Loads lightweight HuggingFace models once and reuses them.
    All models run locally — no API key needed.
    """

    # ── Greeting / Farewell keyword sets ──
    GREETING_WORDS  = {"hi", "hello", "hey", "howdy", "hiya", "sup", "greetings", "good morning",
                       "good afternoon", "good evening", "what's up", "whats up"}
    FAREWELL_WORDS  = {"bye", "goodbye", "see you", "take care", "later", "cya", "good night",
                       "ttyl", "talk later", "farewell"}
    QUESTION_MARKS  = {"?", "what", "when", "where", "who", "why", "how", "is it", "can you",
                       "could you", "would you", "do you", "are you", "will you"}

    def __init__(self):
        logger.info("Loading Transformer models — this may take a moment on first run …")
        self._load_models()
        logger.info("All models loaded ✓")

    def _load_models(self):
        # 1. Sentiment analysis  (distilbert — fast & accurate)
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
        )

        # 2. Named Entity Recognition
        self.ner_pipeline = pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple",
            truncation=True,
        )

        # 3. Text generation  (FLAN-T5-small — seq2seq, ideal for replies)
        model_name = "google/flan-t5-small"
        self.gen_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.gen_model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, text: str) -> MessageAnalysis:
        """Full pipeline: sentiment → intent → NER → reply generation."""
        text = text.strip()
        if not text:
            return self._empty_analysis(text)

        sentiment, sentiment_score = self._get_sentiment(text)
        intent                     = self._classify_intent(text)
        entities                   = self._extract_entities(text)
        generated_reply            = self._generate_reply(text, intent, sentiment)
        confidence                 = round(sentiment_score * 0.9, 3)   # proxy confidence

        return MessageAnalysis(
            original_text   = text,
            sentiment       = sentiment,
            sentiment_score = sentiment_score,
            intent          = intent,
            entities        = entities,
            generated_reply = generated_reply,
            confidence      = confidence,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_sentiment(self, text: str) -> tuple[str, float]:
        result = self.sentiment_pipeline(text[:512])[0]
        label  = result["label"]   # POSITIVE / NEGATIVE
        score  = round(result["score"], 4)
        # Bucket low-confidence as NEUTRAL
        if score < 0.65:
            label = "NEUTRAL"
        return label, score

    def _classify_intent(self, text: str) -> str:
        lower = text.lower()

        if any(g in lower for g in self.GREETING_WORDS):
            return "greeting"
        if any(f in lower for f in self.FAREWELL_WORDS):
            return "farewell"
        if "?" in lower or any(q in lower for q in self.QUESTION_MARKS):
            return "question"
        if any(w in lower for w in ("problem", "issue", "error", "broken", "wrong",
                                     "complaint", "disappointed", "upset", "fix", "bug")):
            return "complaint"
        if any(w in lower for w in ("order", "buy", "purchase", "price", "cost",
                                     "available", "stock", "delivery", "ship")):
            return "inquiry"
        return "other"

    def _extract_entities(self, text: str) -> list[dict]:
        raw = self.ner_pipeline(text[:512])
        return [
            {"entity": e["entity_group"], "word": e["word"], "score": round(e["score"], 3)}
            for e in raw
        ]

    def _generate_reply(self, text: str, intent: str, sentiment: str) -> str:
        """
        Build a context-aware prompt and run FLAN-T5 to generate a reply.
        """
        context_map = {
            "greeting"  : "Respond warmly to this greeting",
            "farewell"  : "Write a friendly farewell response",
            "question"  : "Give a helpful, concise answer",
            "complaint" : "Write an empathetic, solution-focused response",
            "inquiry"   : "Provide clear product/service information",
            "other"     : "Write a polite, helpful response",
        }
        tone_hint = "Keep the tone warm." if sentiment == "POSITIVE" else \
                    "Be extra empathetic and reassuring." if sentiment == "NEGATIVE" else \
                    "Keep the tone neutral and professional."

        prompt = (
            f"{context_map.get(intent, 'Reply helpfully')} to this WhatsApp message. "
            f"{tone_hint}\n\nMessage: {text}\n\nReply:"
        )

        inputs  = self.gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        outputs = self.gen_model.generate(
            **inputs,
            max_new_tokens=120,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        reply = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return reply.strip()

    def _empty_analysis(self, text: str) -> MessageAnalysis:
        return MessageAnalysis(
            original_text   = text,
            sentiment       = "NEUTRAL",
            sentiment_score = 0.0,
            intent          = "other",
            entities        = [],
            generated_reply = "Sorry, I didn't catch that. Could you please repeat?",
            confidence      = 0.0,
        )


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    analyzer = TransformerAnalyzer()
    samples = [
        "Hello! How are you today?",
        "My order is broken and I'm very disappointed!",
        "What is the price of your premium plan?",
        "Thanks, see you later!",
    ]
    for msg in samples:
        result = analyzer.analyze(msg)
        print(f"\n📩 Message  : {result.original_text}")
        print(f"   Sentiment : {result.sentiment} ({result.sentiment_score:.2f})")
        print(f"   Intent    : {result.intent}")
        print(f"   Entities  : {result.entities}")
        print(f"   Reply     : {result.generated_reply}")
