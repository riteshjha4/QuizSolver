
from typing import Any, Dict, Optional
import asyncio
import httpx

from server.scraper import render_page
from server.utils import extract_question_json, extract_submit_url, compute_answer


class QuizSolver:
    def __init__(self, secret: str):
        self.secret = secret

    async def solve_quiz_chain(self, email: str, secret: str, url: str) -> Dict[str, Any]:
        """
        Solve a chain of quizzes beginning at `url`.

        Returns a dict summarizing outcome. Never raises network exceptions to caller;
        instead returns structured error info so the main HTTP handler can reply 200.
        """
        start_time = asyncio.get_event_loop().time()
        TIME_LIMIT = 180  # seconds

        current_url = url
        last_submission: Optional[Dict[str, Any]] = None

        while True:
            # enforce total time limit
            if asyncio.get_event_loop().time() - start_time > TIME_LIMIT:
                return {
                    "done": False,
                    "reason": "time_exceeded",
                    "last_submission": last_submission,
                }

            # Render page (Playwright)
            try:
                html = await render_page(current_url)
            except Exception as e:
                return {
                    "done": False,
                    "reason": "render_failed",
                    "current_url": current_url,
                    "error": str(e),
                    "last_submission": last_submission,
                }

            # Extract question JSON from page
            question = extract_question_json(html)
            if not question:
                return {
                    "done": False,
                    "reason": "cannot_parse_question",
                    "current_url": current_url,
                    "page_html_sample": (html[:2000] + "...") if isinstance(html, str) else None,
                    "last_submission": last_submission,
                }

            # Extract submit URL (from page text per spec)
            submit_url = extract_submit_url(html)
            if not submit_url:
                return {
                    "done": False,
                    "reason": "no_submit_url",
                    "current_url": current_url,
                    "question": question,
                    "last_submission": last_submission,
                }

            # Compute answer using helper (supports PDFs and fallback to question['answer'])
            try:
                answer = await compute_answer(question)
            except Exception as e:
                return {
                    "done": False,
                    "reason": "compute_failed",
                    "current_url": current_url,
                    "error": str(e),
                    "question": question,
                    "last_submission": last_submission,
                }

            # Build submit payload according to spec
            submit_payload: Dict[str, Any] = {
                "email": email,
                "secret": secret,
                "url": current_url,
                "answer": answer,
            }

            # Attempt to POST to submit_url, but handle DNS/connection errors gracefully.
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(submit_url, json=submit_payload)
                    # try parse JSON response safely
                    try:
                        resp_json = resp.json()
                    except Exception:
                        resp_json = {
                            "error": "invalid_json_response",
                            "status_code": resp.status_code,
                            "text_snippet": (resp.text[:2000] + "...") if resp.text else None,
                        }
            except Exception as e:
                # Likely demo/fake URL DNS error or connection refused. Return structured info.
                return {
                    "done": False,
                    "reason": "submit_failed",
                    "submit_url": submit_url,
                    "submit_payload": submit_payload,
                    "error": str(e),
                    "note": "This often occurs for the demo because the submit URL is a placeholder.",
                    "last_submission": last_submission,
                }

            # Save last submission details
            last_submission = {
                "submit_url": submit_url,
                "payload": submit_payload,
                "response": resp_json,
            }

            # Interpret response per spec
            if isinstance(resp_json, dict) and resp_json.get("correct") is True:
                next_url = resp_json.get("url")
                if not next_url:
                    return {"done": True, "last_submission": last_submission}
                # proceed to the next_url and continue loop
                current_url = next_url
                continue
            else:
                # not correct OR response not in expected format
                # if server provided a next url, follow it; otherwise finish with failure
                next_url = resp_json.get("url") if isinstance(resp_json, dict) else None
                if next_url:
                    current_url = next_url
                    continue
                else:
                    return {
                        "done": False,
                        "reason": "incorrect_or_unexpected_response",
                        "last_submission": last_submission,
                    }
