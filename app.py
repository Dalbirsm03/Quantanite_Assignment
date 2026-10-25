from __future__ import annotations

import csv
import io
import logging
import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from email_agent import (
    EmailAutomationAgent,
    history_from_text,
    parse_email_file,
)


BASE_DIR = Path(__file__).parent
SAMPLE_DIR = BASE_DIR / "samples" / "inbox"
SUPPORTED_TYPES = ["eml", "txt"]
logger = logging.getLogger(__name__)


def _sample_emails() -> list[dict[str, Any]]:
    return [
        parse_email_file(path.name, path.read_bytes())
        for path in sorted(SAMPLE_DIR.iterdir())
        if path.suffix.lower() in {".eml", ".txt"}
    ]


def _history_text(uploaded_history: Any) -> str:
    if uploaded_history is not None:
        return uploaded_history.getvalue().decode("utf-8", errors="replace")
    return (BASE_DIR / "samples" / "history.txt").read_text(encoding="utf-8")


def _csv_export(results: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    fields = [
        "id",
        "sender",
        "subject",
        "sentiment",
        "urgency",
        "confidence",
        "category",
        "priority",
        "review_required",
        "review_reason",
        "proposed_dispatch",
        "draft",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for item in results:
        row = {field: item[field] for field in fields}
        row["confidence"] = f"{item['confidence']:.0%}"
        writer.writerow(row)
    return output.getvalue()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()
    st.set_page_config(page_title="Quantanite Email Agent", page_icon="✉️", layout="wide")
    st.title("Agentic AI for Email Automation")
    st.caption(
        "Classify customer emails, draft contextual replies, and queue them for review. "
        "Proposed dispatch uses each email's Date header. This demo never sends email."
    )

    with st.sidebar:
        st.header("Workflow settings")
        api_key = st.text_input(
            "Gemini API key (optional)",
            value=os.getenv("GOOGLE_API_KEY", ""),
            type="password",
            help="If omitted, the app uses its local rule-based classifier and draft templates.",
        )
        use_gemini = st.checkbox("Use Gemini for classification and drafting", value=False)
        st.divider()
        st.markdown(
            "**Dispatch safety:** all responses are drafts. No email service or send action "
            "is connected."
        )

    uploaded_emails = st.file_uploader(
        "Upload customer emails (.eml or .txt)",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
    )
    uploaded_history = st.file_uploader(
        "Optional historical responses (.txt)",
        type=["txt"],
        help="Separate multiple examples with a line containing ---.",
    )
    emails = (
        [
            parse_email_file(file.name, file.getvalue())
            for file in uploaded_emails
        ]
        if uploaded_emails
        else _sample_emails()
    )

    with st.form("run_agent"):
        run_workflow = st.form_submit_button("Prioritize, draft, and schedule")

    if not run_workflow:
        st.info("Upload emails or run the bundled sample inbox to preview the workflow.")
        return
    if use_gemini and not api_key.strip():
        st.error("Enter a Gemini API key or turn off Gemini to use local demo mode.")
        return

    try:
        logger.info("Starting email workflow for %s emails", len(emails))
        agent = (
            EmailAutomationAgent.from_gemini(api_key)
            if use_gemini
            else EmailAutomationAgent()
        )
        results = agent.run(
            emails=emails,
            history=history_from_text(_history_text(uploaded_history)),
        )
        logger.info("Email workflow finished with %s results", len(results))
    except Exception as exc:
        logger.exception("Email workflow failed")
        st.error(f"The email workflow failed: {exc}")
        return

    st.subheader("Prioritized response queue")
    summary = [
        {
            "Priority": item["priority"],
            "Customer": item["name"],
            "Subject": item["subject"],
            "Sentiment": item["sentiment"].title(),
            "Urgency": item["urgency"].title(),
            "Confidence": f"{item['confidence']:.0%}",
            "Proposed dispatch": item["proposed_dispatch"],
            "Review needed": "Yes" if item["review_required"] else "No",
        }
        for item in results
    ]
    st.dataframe(summary, width="stretch", hide_index=True)

    for item in results:
        label = f"{item['priority']} | {item['subject']}"
        with st.expander(label):
            st.write(f"**From:** {item['name']} <{item['sender']}>")
            st.write(
                f"**Classification:** {item['sentiment']} sentiment · "
                f"{item['urgency']} urgency · {item['category']}"
            )
            st.write(f"**Confidence score:** {item['confidence']:.0%}")
            st.write(f"**Reason:** {item['rationale']}")
            if item["review_required"]:
                st.warning(f"Please review this email: {item['review_reason']}")
            st.text_area(
                "Draft response",
                value=item["draft"],
                height=180,
                key=f"draft_{item['id']}",
            )

    st.download_button(
        "Download response queue (CSV)",
        data=_csv_export(results),
        file_name="scheduled_email_drafts.csv",
        mime="text/csv",
    )
    with st.expander("LangGraph orchestration"):
        st.code(
            "START → classifier_agent → prioritizer → drafting_agent\n"
            "       → review_agent → scheduler → END"
        )
        st.markdown(
            "Each node reads and updates the shared `EmailAgentState`. "
            "Scheduling is a simple proposed time based on priority. "
            "Sensitive topics are flagged for human review."
        )


if __name__ == "__main__":
    main()
