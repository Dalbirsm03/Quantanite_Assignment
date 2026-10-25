import logging

from state import EmailAgentState

logger = logging.getLogger(__name__)


SENSITIVE_WORDS = ("legal", "lawyer", "fraud", "chargeback", "data breach")


def review_emails(state: EmailAgentState) -> dict:
    logger.info("Checking %s emails for sensitive topics", len(state["drafts"]))
    reviews = []
    for email in state["drafts"]:
        text = f"{email['subject']} {email['body']}".lower()
        reasons = [word for word in SENSITIVE_WORDS if word in text]
        reviews.append(
            {
                "id": email["id"],
                "review_required": bool(reasons),
                "review_reason": (
                    f"Sensitive topic: {', '.join(reasons)}" if reasons else ""
                ),
            }
        )
    logger.info("Flagged %s emails for human review", sum(
        review["review_required"] for review in reviews
    ))
    return {"draft_reviews": reviews}
