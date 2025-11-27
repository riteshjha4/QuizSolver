import json
import re
import base64
from bs4 import BeautifulSoup
import httpx
import pandas as pd
import pdfplumber
import io

def extract_question_json(html: str):
    """
    Extracts the <pre> JSON from decoded base64 block.
    """
    soup = BeautifulSoup(html, "lxml")
    pre = soup.find("pre")
    if pre:
        try:
            return json.loads(pre.get_text().strip())
        except:
            pass
    return None


def extract_submit_url(html: str):
    # Case 1: official format
    m = re.search(r"Post your answer to\s+(https?://[^\s\"']+)", html)
    if m:
        return m.group(1).strip()

    # Case 2: any URL containing '/submit'
    m = re.search(r"(https?://[^\s\"']+/submit[^\s\"']*)", html)
    if m:
        return m.group(1).strip()

    return None


async def compute_answer(question: dict):
    """
    Handles:
    - PDFs
    - numeric answers
    - direct JSON answers
    """
    file_url = question.get("url")

    if file_url and file_url.endswith(".pdf"):
        async with httpx.AsyncClient() as client:
            r = await client.get(file_url)
            pdf_bytes = r.content

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page = pdf.pages[1] if len(pdf.pages) > 1 else pdf.pages[0]
            table = page.extract_table()

        df = pd.DataFrame(table[1:], columns=table[0])

        for col in df.columns:
            if col.lower() == "value":
                df[col] = pd.to_numeric(df[col], errors="coerce")
                return int(df[col].sum())

    # fallback if "answer" provided
    return question.get("answer")
