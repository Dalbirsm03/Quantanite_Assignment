# Agentic AI for Email Automation

A LangGraph application that reads `.eml` and header-formatted `.txt`
customer emails, ranks them by urgency and sentiment, drafts personalized
responses using prior response examples, flags sensitive topics for human
review, and suggests a dispatch time based on priority. It does **not** connect
to an email account or send messages.

## Run the project

In PowerShell, open the folder that contains `Quantanite_Assignment`, then run:

```powershell
cd .\Quantanite_Assignment
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens in your browser with sample emails. To use your own, upload
`.eml` or `.txt` email files. You can also upload a `.txt` file of past
responses; separate each response with a line containing `---`.

## Run the tests

With the virtual environment active, run this from the `Quantanite_Assignment`
folder:

```powershell
python -m unittest tests.test_email_agent
```

### Optional Gemini integration

In the sidebar, enable **Use Gemini** and supply a Google Gemini API key, or
set `GOOGLE_API_KEY` in the environment before launch. When enabled, the email
content and response examples are sent to the configured Gemini API. Without
a key, the app uses local keyword-based sentiment/urgency classification and
personalized templates so that the complete workflow can be tried offline.

## Agent orchestration strategy

The compiled `StateGraph` passes one shared `EmailAgentState` through the
following flow:

1. **`EmailClassifierAgent` (`classifier_agent`)** uses structured Gemini
   output when enabled. Offline, simple keyword rules estimate sentiment and
   urgency. Both modes attach keyword-evidence scores to each classification.
2. **`prioritizer`** is plain Python. High urgency is P1, otherwise negative
   sentiment is P2, and the remaining emails are P3.
3. **`EmailDraftingAgent` (`drafting_agent`)** asks Gemini to write an
   empathetic reply grounded in the message and historical response examples.
   The local mode uses a customer-name-aware support template. Prompts treat
   email text as untrusted reference content and prohibit claiming unverified
   actions or outcomes.
4. **`review_agent`** flags emails containing sensitive topics such as legal
   complaints, fraud, or data breaches for human review. It does not score
   drafts or send them back for rewriting.
5. **`scheduler`** uses the email's `Date` header as the starting time and
   suggests dispatch: P1 immediately, P2 one hour later, and P3 one day later.
   If a message has no readable date, it logs a warning and uses the current
   time. These are simple suggestions, not service-level deadlines.

The shared state schema is in `state.py`; the graph is compiled and invoked by
`email_agent.py`. The Streamlit app displays the queue, editable drafts, review
flags, proposed dispatch times, and the graph flow.
Workflow progress and errors are logged to the console at INFO level.

### Confidence score

The app averages keyword-match scores for sentiment, urgency, and category.


## Project layout

- `app.py` — Streamlit interface, uploads, queue preview, and CSV export.
- `state.py` — shared TypedDict state and incoming-email schema.
- `email_agent.py` — input parsing and LangGraph orchestration.
- `agents/` — dedicated classifier, prioritizer, drafter, reviewer, and
  scheduler.
- `samples/inbox/` — sample customer messages.
- `samples/history.txt` — historical response examples.
- `tests/test_email_agent.py` — parsing, ranking, and scheduling tests.
