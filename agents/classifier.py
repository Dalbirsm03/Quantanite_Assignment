import logging
from typing import Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

from state import EmailAgentState, IncomingEmail

logger = logging.getLogger(__name__)


class EmailClassification(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    urgency: Literal["high", "medium", "low"]
    category: str
    rationale: str


def classify_locally(email: IncomingEmail) -> dict:
    text = f"{email['subject']} {email['body']}".lower()
    high_words = ("urgent", "asap", "locked out", "outage", "today")
    medium_words = ("soon", "deadline", "delayed", "not working")
    negative_words = ("angry", "frustrated", "problem", "issue", "refund")
    positive_words = ("thank", "great", "happy")

    urgency = "high" if any(word in text for word in high_words) else (
        "medium" if any(word in text for word in medium_words) else "low"
    )
    sentiment = "negative" if any(word in text for word in negative_words) else (
        "positive" if any(word in text for word in positive_words) else "neutral"
    )
    if any(word in text for word in ("bill", "charge", "refund", "payment")):
        category = "billing"
    elif any(word in text for word in ("login", "access", "error", "broken")):
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
                        "(high, medium or low), category, and a short rationale.\n"
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
            classifications.append({**email, **classification})
        logger.info("Finished classifying emails")
        return {"classifications": classifications}
