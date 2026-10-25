import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from state import EmailAgentState

logger = logging.getLogger(__name__)


class EmailDraftingAgent:
    def __init__(self, model=None):
        self.model = model

    def run(self, state: EmailAgentState) -> dict:
        logger.info("Drafting responses for %s emails", len(state["prioritized_emails"]))
        drafts = []
        for item in state["prioritized_emails"]:
            name = item.get("name") or "there"
            if self.model is None:
                draft = (
                    f"Hello {name},\n\n"
                    f"Thank you for contacting us about {item['subject'].lower()}. "
                    "We're sorry for the inconvenience. We will review your message "
                    "and follow up with the next steps.\n\n"
                    "Best,\nCustomer Support"
                )
            else:
                prompt = PromptTemplate(
                    template=(
                        "Write a short, empathetic customer support reply. "
                        "Do not promise a refund or claim an action has already happened.\n"
                        "Customer: {name}\nSubject: {subject}\nEmail: {body}\n"
                        "Past responses: {history}\nReply:"
                    ),
                    input_variables=["name", "subject", "body", "history"],
                )
                chain = prompt | self.model | StrOutputParser()
                draft = chain.invoke(
                    {
                        "name": name,
                        "subject": item["subject"],
                        "body": item["body"],
                        "history": state["history"][:2],
                    }
                ).strip()
                if not draft:
                    raise ValueError(f"Could not draft a reply for {item['subject']}.")
            drafts.append({**item, "draft": draft})
        logger.info("Finished drafting responses")
        return {"drafts": drafts}
