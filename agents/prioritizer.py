import logging

from state import EmailAgentState

logger = logging.getLogger(__name__)


def prioritize_emails(state: EmailAgentState) -> dict:
    logger.info("Prioritizing %s emails", len(state["classifications"]))
    prioritized = []
    for item in state["classifications"]:
        score = 2 if item["urgency"] == "high" else 1 if item["sentiment"] == "negative" else 0
        priority = "P1 - High" if score == 2 else "P2 - Normal" if score == 1 else "P3 - Low"
        prioritized.append({**item, "priority": priority, "priority_score": score})
    logger.info("Finished prioritizing emails")
    return {"prioritized_emails": prioritized}
