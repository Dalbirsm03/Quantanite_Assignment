import logging
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime

from state import EmailAgentState

logger = logging.getLogger(__name__)


def schedule_emails(state: EmailAgentState) -> dict:
    logger.info("Suggesting dispatch times for %s emails", len(state["drafts"]))
    reviews = {item["id"]: item for item in state["draft_reviews"]}
    dispatch_delays = {
        "P1 - High": timedelta(minutes=0),
        "P2 - Normal": timedelta(hours=1),
        "P3 - Low": timedelta(days=1),
    }
    results = []

    emails = sorted(
        state["drafts"],
        key=lambda item: (-item["priority_score"], item["id"]),
    )
    for email in emails:
        received_at = email["received_at"]
        if received_at:
            try:
                received_time = datetime.fromisoformat(received_at)
            except ValueError:
                try:
                    received_time = parsedate_to_datetime(received_at)
                except (TypeError, ValueError):
                    logger.warning(
                        "Could not parse Date header for %s; using current time",
                        email["id"],
                    )
                    received_time = datetime.now()
        else:
            logger.warning(
                "Email %s has no Date header; using current time", email["id"]
            )
            received_time = datetime.now()

        review = reviews[email["id"]]
        results.append(
            {
                **email,
                "proposed_dispatch": (
                    received_time + dispatch_delays[email["priority"]]
                ).isoformat(timespec="minutes"),
                "review_required": review["review_required"],
                "review_reason": review["review_reason"],
            }
        )

    logger.info("Finished scheduling email drafts")
    return {"results": results}
