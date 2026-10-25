from typing import Any, TypedDict


class IncomingEmail(TypedDict):
    id: str
    sender: str
    name: str
    subject: str
    body: str
    received_at: str


class EmailAgentState(TypedDict):
    emails: list[IncomingEmail]
    history: list[str]
    classifications: list[dict[str, Any]]
    prioritized_emails: list[dict[str, Any]]
    drafts: list[dict[str, Any]]
    draft_reviews: list[dict[str, Any]]
    results: list[dict[str, Any]]
