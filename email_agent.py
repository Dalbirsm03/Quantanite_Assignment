from __future__ import annotations

import re
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr
from typing import Any

from langgraph.graph import END, START, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.classifier import (
    EmailClassifierAgent,
)
from agents.drafter import EmailDraftingAgent
from agents.prioritizer import prioritize_emails
from agents.reviewer import review_emails
from agents.scheduler import schedule_emails
from state import EmailAgentState, IncomingEmail



def parse_email_file(filename: str, content: bytes) -> IncomingEmail:
    """Parse a .eml or header-formatted .txt file into a normalized email."""
    message = BytesParser(policy=policy.default).parsebytes(content)
    subject = str(message.get("subject", "")).strip()
    sender_header = str(message.get("from", "")).strip()
    sender_name, sender_address = parseaddr(sender_header)

    if message.is_multipart():
        body_parts = [
            part.get_content()
            for part in message.walk()
            if part.get_content_type() == "text/plain"
            and part.get_content_disposition() != "attachment"
        ]
        body = "\n".join(str(part) for part in body_parts).strip()
    else:
        body_content = message.get_content()
        body = body_content if isinstance(body_content, str) else str(body_content)
        body = body.strip()

    if not subject and not body:
        body = content.decode("utf-8", errors="replace").strip()
    if not subject:
        first_line = body.splitlines()[0] if body else filename
        subject = first_line[:120]
    if not body:
        body = "(No message body provided.)"

    return IncomingEmail(
        id=filename,
        sender=sender_address or sender_header or "unknown@example.com",
        name=sender_name
        or (sender_address.split("@")[0] if "@" in sender_address else "there"),
        subject=subject,
        body=body,
        received_at=str(message.get("date", "")).strip(),
    )


class EmailAutomationAgent:
    """Orchestrates specialized email agents using a shared LangGraph state."""

    def __init__(self, model: Any = None):
        self.classifier_agent = EmailClassifierAgent(model)
        self.drafting_agent = EmailDraftingAgent(model)

        graph = StateGraph(EmailAgentState)
        graph.add_node("classifier_agent", self.classifier_agent.run)
        graph.add_node("prioritizer", prioritize_emails)
        graph.add_node("drafting_agent", self.drafting_agent.run)
        graph.add_node("review_agent", review_emails)
        graph.add_node("scheduler", schedule_emails)
        graph.add_edge(START, "classifier_agent")
        graph.add_edge("classifier_agent", "prioritizer")
        graph.add_edge("prioritizer", "drafting_agent")
        graph.add_edge("drafting_agent", "review_agent")
        graph.add_edge("review_agent", "scheduler")
        graph.add_edge("scheduler", END)
        self.graph = graph.compile()

    @classmethod
    def from_gemini(
        cls, api_key: str, model_name: str = "gemini-2.5-flash"
    ) -> "EmailAutomationAgent":
        if not api_key.strip():
            raise ValueError("A Gemini API key is required to enable Gemini.")

        model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
        )
        return cls(model=model)

    def run(
        self,
        emails: list[IncomingEmail],
        history: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not emails:
            raise ValueError("Provide at least one incoming email.")
        initial_state: EmailAgentState = {
            "emails": emails,
            "history": history or [],
            "classifications": [],
            "prioritized_emails": [],
            "drafts": [],
            "draft_reviews": [],
            "results": [],
        }
        return self.graph.invoke(initial_state)["results"]


def history_from_text(content: str) -> list[str]:
    """Split a plain-text history file into individual prior responses."""
    return [part.strip() for part in re.split(r"\n\s*---+\s*\n", content) if part.strip()]
