import os
import re
import json
import base64
import httpx
import asyncio
import pandas as pd
from playwright.async_api import async_playwright
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def run_quiz_cycle(email: str, secret: str, task_url: str):
    """
    Main loop:
    1. Visit Task URL.
    2. Extract instructions.
    3. Generate Code & Solve.
    4. Submit Answer.
    5. If 'next' URL exists, Recurse.
    """
    print(f"\n[+] Starting Task: {task_url}")
    
    # --- Step 1: Scrape the Task Page ---
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(task_url)
        
        # Wait for content to render (handling JS delays)
        try:
            await page.wait_for_selector("body", timeout=5000)
            # Extra wait for JS rendering mechanisms (like the atob example in prompt)
            await asyncio.sleep(2) 
        except:
            pass

        # Extract text content
        content = await page.inner_text("body")
        html_content = await page.content()
        await browser.close()

    print(f"    -> Page Content Extracted ({len(content)} chars).")

    # --- Step 2: Analyze with LLM ---
    # We ask the LLM to understand the task and write Python code to solve it.
    prompt = f"""
    You are an autonomous data analysis agent. You are looking at a quiz webpage.
    
    WEBPAGE CONTENT:
    {content}

    YOUR GOAL:
    1. Identify the Question.
    2. Identify any data links (CSV, PDF, JSON) mentioned.
    3. Identify the Submission URL.
    4. Write a Python script to calculate the answer.

    REQUIREMENTS:
    - The script must define a variable `result` containing the final answer.
    - If you need to download files, use `requests` or `pandas`.
    - The answer format is likely a number, string, or JSON.
    - RETURN ONLY JSON in this format:
    {{
        "reasoning": "Explanation of steps",
        "python_code": "The python code to solve it. Do not include markdown backticks.",
        "submission_url": "The full URL to post the answer to"
    }}
    """

    completion = client.chat.completions.create(
        model="gpt-4o", # Use a capable model
        messages=[{"role": "system", "content": "You are a helpful coding assistant. Output JSON only."}, 
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    llm_response = json.loads(completion.choices[0].message.content)
    print(f"    -> Plan: {llm_response['reasoning']}")

    # --- Step 3: Execute the Solution Code ---
    # We create a local scope to capture the 'result' variable
    local_scope = {}
    
    try:
        # Dangerous: executing AI code. Ensure sandbox in real deployment.
        exec(llm_response["python_code"], globals(), local_scope)
        answer = local_scope.get("result")
        print(f"    -> Calculated Answer: {answer}")
    except Exception as e:
        print(f"    [!] Error executing solution code: {e}")
        return # Stop if code fails

    # --- Step 4: Submit the Answer ---
    submission_url = llm_response.get("submission_url")
    if not submission_url:
        print("    [!] No submission URL found.")
        return

    payload = {
        "email": email,
        "secret": secret,
        "url": task_url,
        "answer": answer
    }

    print(f"    -> Submitting to {submission_url}...")
    
    async with httpx.AsyncClient() as client_http:
        try:
            resp = await client_http.post(submission_url, json=payload, timeout=10)
            resp_data = resp.json()
            
            print(f"    -> Submission Response: {resp.status_code} | {resp_data}")

            # --- Step 5: Check for Next Task ---
            # If correct and a new URL is provided, loop.
            if resp.status_code == 200 and resp_data.get("correct") and resp_data.get("url"):
                next_url = resp_data["url"]
                print(f"    [+] Moving to next level: {next_url}")
                await run_quiz_cycle(email, secret, next_url)
            else:
                print("    [=] Quiz Chain Ended or Failed.")
                
        except Exception as e:
            print(f"    [!] Submission failed: {e}")