This project implements an API endpoint that can automatically solve data driven quizzes using Large Language Models (LLMs).
The system is designed to handle real world data tasks such as scraping, cleaning, analysis, and visualization — all within strict time and payload constraints.

The API receives quiz URLs, visits and renders JavaScript based pages, understands the task instructions, performs the required data operations, and submits the correct answer back to the evaluator.

This project also includes prompt-engineering defenses and attacks, tested through controlled system and user prompts, and is evaluated through an automated quiz pipeline and a viva.

Key Features

Secure API endpoint with secret based verification

Supports JavaScript rendered quiz pages using headless browsing

Handles multi step quiz chains

Capable of:

Web scraping

API data sourcing

PDF and text extraction

Data cleaning and transformation

Statistical analysis and ML-based reasoning

Chart and visualization generation

Robust error handling and retry logic

Prompt engineering strategies tested against adversarial prompts

Fully automated submission flow within 3 minute time limit.

Architecture Overview:
Client (Evaluator)
      |
      | POST (email, secret, quiz URL)
      v
Quiz Solver API
      |
      ├── Secret Validation
      ├── Headless Browser (JS execution)
      ├── Task Parser
      ├── Data Pipeline
      │     ├── Scraping / API calls
      │     ├── Cleaning / Processing
      │     ├── Analysis / Visualization
      │
      ├── LLM Reasoning Layer
      └── Answer Submission

API Contract
Request Format
{
  "email": "23f3002470@ds.study.iitm.ac.in",
  "secret": "....",
  "url": "https://example.com/quiz123"
}

Response Codes:
| Status Code | Meaning                        |
| ----------- | ------------------------------ |
| 200         | Request accepted and processed |
| 400         | Invalid JSON payload           |
| 403         | Invalid secret                 |


Quiz Solving Workflow

Validate Request

Email and secret are verified.

Invalid requests are rejected immediately.

Visit Quiz Page

Page is rendered using a headless browser to execute JavaScript.

Base64 encoded instructions are decoded if present.

Understand the Task

The system identifies:

Required data sources

Data processing steps

Expected answer format

Data Processing

Data is collected from files, APIs, or web pages.

Cleaning, filtering, aggregation, or modeling is applied.

Visualizations are generated when required.

Submit Answer:

Answer is sent to the provided submit URL.

Retries are handled if allowed.

The system proceeds to the next quiz URL if provided.

Completion:

The quiz ends when no new URL is returned.

Prompt Engineering Strategy
System Prompt (Defense)

Designed to resist leaking hidden information

Explicitly blocks disclosure of internal variables

Applies strict refusal rules for sensitive content

User Prompt (Attack)

Crafted to override system constraints

Tested across multiple models and random code words

Evaluated for robustness and exploitability

This dual-prompt approach demonstrates practical prompt security evaluation, not theoretical examples.

Supported Answer Types

The API can submit answers as:

Numbers

Strings

Booleans

JSON objects

Base64-encoded files (images, charts, PDFs)

All responses remain under the 1MB size limit.

Tech Stack

Backend: Python (FastAPI)

Headless Browser: Playwright

Data Tools: Pandas, NumPy

Visualization: Matplotlib / Plotly

Parsing: BeautifulSoup, PyMuPDF

Deployment: HTTPS-enabled cloud endpoint

Security Considerations

Secret based request validation

No hardcoded URLs

Controlled LLM output parsing

Payload size and time constraints enforced

Safe handling of external content
