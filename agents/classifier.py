import logging
from typing import Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

from state import EmailAgentState, IncomingEmail

logger = logging.getLogger(__name__)

HIGH_URGENCY_WORDS = ("urgent", "asap", "locked out", "outage", "today")
MEDIUM_URGENCY_WORDS = ("soon", "deadline", "delayed", "not working")
LOW_URGENCY_WORDS = ("no rush", "when convenient", "whenever possible", "not urgent")
NEGATIVE_WORDS = ("angry", "frustrated", "problem", "issue", "refund")
POSITIVE_WORDS = ("thank", "great", "happy")
NEUTRAL_WORDS = ("question", "information", "wondering", "could you")
CATEGORY_LABELS = ("billing", "technical support", "account support")
CATEGORY_WORDS = {
    "billing": ("bill", "charge", "refund", "payment"),
    "technical support": ("login", "access", "error", "broken"),
    "account support": ("account", "profile", "password", "subscription"),
}


class EmailClassification(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    urgency: Literal["high", "medium", "low"]
    category: str
    rationale: str


def _evidence_score(text: str, label: str, evidence: dict[str, tuple[str, ...]]) -> float | None:
    matches = {
        candidate: sum(cue in text for cue in cues)
        for candidate, cues in evidence.items()
    }
    total_matches = sum(matches.values())
    if total_matches == 0:
        return 0.0
    return round(matches.get(label, 0) / total_matches, 2)


def calculate_confidence(email: IncomingEmail, classification: dict) -> float:
    """Average the keyword evidence scores for sentiment, urgency, and category."""
    text = f"{email['subject']} {email['body']}".lower()
    evidence_scores = {
        "urgency_confidence": _evidence_score(
            text,
            classification["urgency"],
            {
                "high": HIGH_URGENCY_WORDS,
                "medium": MEDIUM_URGENCY_WORDS,
                "low": LOW_URGENCY_WORDS,
            },
        ),
        "sentiment_confidence": _evidence_score(
            text,
            classification["sentiment"],
            {
                "negative": NEGATIVE_WORDS,
                "positive": POSITIVE_WORDS,
                "neutral": NEUTRAL_WORDS,
            },
        ),
        "category_confidence": _evidence_score(
            text,
            classification["category"],
            CATEGORY_WORDS,
        ),
    }
    return round(sum(evidence_scores.values()) / len(evidence_scores), 2)


def classify_locally(email: IncomingEmail) -> dict:
    text = f"{email['subject']} {email['body']}".lower()

    urgency = "high" if any(word in text for word in HIGH_URGENCY_WORDS) else (
        "medium" if any(word in text for word in MEDIUM_URGENCY_WORDS) else "low"
    )
    sentiment = "negative" if any(word in text for word in NEGATIVE_WORDS) else (
        "positive" if any(word in text for word in POSITIVE_WORDS) else "neutral"
    )
    if any(word in text for word in CATEGORY_WORDS["billing"]):
        category = "billing"
    elif any(word in text for word in CATEGORY_WORDS["technical support"]):
        category = "technical support"
    else:
        category = "account support"

    return {
        "sentiment": sentiment,
        "urgency": urgency,
        "category": category,
        "rationale": "Estimated from the email's wording.",
    }


class EmailClassifierAgent:
    def __init__(self, model=None):
        self.model = model

    def run(self, state: EmailAgentState) -> dict:
        logger.info("Classifying %s emails", len(state["emails"]))
        classifications = []
        for email in state["emails"]:
            if self.model is None:
                classification = classify_locally(email)
            else:
                parser = JsonOutputParser(pydantic_object=EmailClassification)
                prompt = PromptTemplate(
                    template=(
                        "Classify the customer email.\n"
                        "Return sentiment (positive, neutral or negative), urgency "
                        "(high, medium or low), category (billing, technical support, "
                        "or account support), and a short rationale.\n"
                        "Subject: {subject}\nEmail: {body}\n{format_instructions}"
                    ),
                    input_variables=["subject", "body"],
                    partial_variables={
                        "format_instructions": parser.get_format_instructions()
                    },
                )
                chain = prompt | self.model | parser
                classification = chain.invoke(
                    {"subject": email["subject"], "body": email["body"]}
                )
            classification["confidence"] = calculate_confidence(email, classification)
            classifications.append({**email, **classification})
        logger.info("Finished classifying emails")
        return {"classifications": classifications}
