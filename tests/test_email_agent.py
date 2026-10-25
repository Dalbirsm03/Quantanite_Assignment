from pathlib import Path
import unittest

from email_agent import (
    EmailAutomationAgent,
    history_from_text,
    parse_email_file,
)
from agents.prioritizer import prioritize_emails
from agents.reviewer import review_emails
from agents.scheduler import schedule_emails


class EmailAutomationTests(unittest.TestCase):
    def test_parses_header_formatted_text(self):
        email = parse_email_file(
            "customer.txt",
            b"From: Sam Lee <sam@example.com>\nSubject: Help\n\nI need help.",
        )
        self.assertEqual(email["name"], "Sam Lee")
        self.assertEqual(email["sender"], "sam@example.com")
        self.assertEqual(email["subject"], "Help")
        self.assertEqual(email["body"], "I need help.")

    def test_history_uses_separator(self):
        self.assertEqual(history_from_text("first reply\n---\nsecond reply"), ["first reply", "second reply"])

    def test_graph_prioritizes_and_schedules_without_sending(self):
        emails = [
            parse_email_file(
                "low.txt",
                b"From: A <a@example.com>\nDate: Fri, 23 Oct 2026 09:00:00\n"
                b"Subject: Thanks\n\nThank you, great help.",
            ),
            parse_email_file(
                "urgent.txt",
                b"From: B <b@example.com>\nDate: Fri, 23 Oct 2026 09:00:00\n"
                b"Subject: Urgent access issue\n\n"
                b"I am locked out and cannot access my account today.",
            ),
        ]
        results = EmailAutomationAgent().run(
            emails,
            history=["Thanks for contacting support."],
        )
        self.assertEqual(results[0]["id"], "urgent.txt")
        self.assertEqual(results[0]["priority"], "P1 - High")
        self.assertEqual(results[0]["proposed_dispatch"], "2026-10-23T09:00")
        self.assertEqual(results[1]["proposed_dispatch"], "2026-10-24T09:00")
        self.assertIn("Hello B", results[0]["draft"])
        self.assertTrue(all("review_required" in result for result in results))

    def test_high_urgency_is_p1_even_without_negative_sentiment(self):
        email = parse_email_file(
            "urgent.txt",
            b"From: B <b@example.com>\nDate: Fri, 23 Oct 2026 09:00:00\n"
            b"Subject: Account access\n\n"
            b"I need access today for my deadline.",
        )
        result = EmailAutomationAgent().run([email])[0]
        self.assertEqual(result["urgency"], "high")
        self.assertEqual(result["sentiment"], "neutral")
        self.assertEqual(result["priority"], "P1 - High")

    def test_graph_has_separate_agent_nodes_and_shared_state(self):
        agent = EmailAutomationAgent()
        self.assertEqual(
            set(agent.graph.get_graph().nodes),
            {
                "__start__",
                "__end__",
                "classifier_agent",
                "prioritizer",
                "drafting_agent",
                "review_agent",
                "scheduler",
            },
        )

    def test_priority_is_deterministic_from_urgency_and_sentiment(self):
        state = {
            "classifications": [
                {"urgency": "high", "sentiment": "positive"},
                {"urgency": "medium", "sentiment": "negative"},
                {"urgency": "low", "sentiment": "positive"},
            ]
        }
        prioritized = prioritize_emails(state)["prioritized_emails"]
        self.assertEqual(
            [(item["priority"], item["priority_score"]) for item in prioritized],
            [("P1 - High", 2), ("P2 - Normal", 1), ("P3 - Low", 0)],
        )

    def test_scheduler_proposes_time_based_on_priority(self):
        state = {
            "draft_reviews": [
                {"id": "p1", "review_required": False, "review_reason": ""},
                {"id": "p2", "review_required": False, "review_reason": ""},
                {"id": "p3", "review_required": False, "review_reason": ""},
            ],
            "drafts": [
                {
                    "id": "p1",
                    "priority_score": 2,
                    "priority": "P1 - High",
                    "received_at": "2026-10-23T09:00:00",
                },
                {
                    "id": "p2",
                    "priority_score": 1,
                    "priority": "P2 - Normal",
                    "received_at": "2026-10-23T09:00:00",
                },
                {
                    "id": "p3",
                    "priority_score": 0,
                    "priority": "P3 - Low",
                    "received_at": "2026-10-23T09:00:00",
                },
            ],
        }
        results = schedule_emails(state)["results"]
        self.assertEqual(results[0]["proposed_dispatch"], "2026-10-23T09:00")
        self.assertEqual(results[1]["proposed_dispatch"], "2026-10-23T10:00")
        self.assertEqual(results[2]["proposed_dispatch"], "2026-10-24T09:00")

    def test_email_date_header_is_used_for_dispatch_time(self):
        email = parse_email_file(
            "urgent.eml",
            b"From: Sam Lee <sam@example.com>\n"
            b"Date: Fri, 23 Oct 2026 14:30:00 +0530\n"
            b"Subject: Urgent account issue\n\nPlease help today.",
        )
        result = EmailAutomationAgent().run([email])[0]
        self.assertEqual(result["proposed_dispatch"], "2026-10-23T14:30+05:30")

    def test_review_flags_sensitive_emails_without_retry(self):
        state = {
            "drafts": [
                {
                    "id": "legal.txt",
                    "subject": "Legal complaint",
                    "body": "I have contacted my lawyer.",
                },
                {
                    "id": "normal.txt",
                    "subject": "Account question",
                    "body": "Please update my account.",
                },
            ]
        }
        reviews = review_emails(state)["draft_reviews"]
        self.assertTrue(reviews[0]["review_required"])
        self.assertIn("legal", reviews[0]["review_reason"])
        self.assertFalse(reviews[1]["review_required"])

    def test_sample_files_are_readable(self):
        sample_dir = Path(__file__).parents[1] / "samples" / "inbox"
        self.assertGreaterEqual(len(list(sample_dir.glob("*.eml"))) + len(list(sample_dir.glob("*.txt"))), 2)


if __name__ == "__main__":
    unittest.main()
