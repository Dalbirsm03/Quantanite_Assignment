import logging

from state import EmailAgentState

logger = logging.getLogger(__name__)


SENSITIVE_WORDS = ("legal", "lawyer", "fraud", "chargeback", "data breach")
LOW_CONFIDENCE_THRESHOLD = 0.60


def review_emails(state: EmailAgentState) -> dict:
    logger.info("Checking %s emails for human review", len(state["drafts"]))
    reviews = []
    for email in state["drafts"]:
        text = f"{email['subject']} {email['body']}".lower()
        sensitive_reasons = [word for word in SENSITIVE_WORDS if word in text]
        review_reasons = []
        confidence = email.get("confidence")
        if confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD:
            review_reasons.append(
                f"Low classification confidence ({confidence:.0%})"
            )
        reasons = []
        if sensitive_reasons:
            reasons.append(f"Sensitive topic: {', '.join(sensitive_reasons)}")
        reasons.extend(review_reasons)
        reviews.append(
            {
                "id": email["id"],
                "review_required": bool(reasons),
                "review_reason": "; ".join(reasons),
            }
        )
    logger.info("Flagged %s emails for human review", sum(
        review["review_required"] for review in reviews
    ))
    return {"draft_reviews": reviews}
