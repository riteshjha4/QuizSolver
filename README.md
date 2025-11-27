This repository contains my implementation for TDS Project 2: LLM Analysis Quiz.
The application exposes an API endpoint that receives a quiz task, processes a JavaScript rendered webpage, extracts and solves the question including PDF/table tasks and submits the correct answer within the required 3 minute window.

The project also includes prompt engineering components for the system prompt and user prompt competition.

Features

1) Fully automated quiz solver

Uses FastAPI to expose a POST endpoint
Uses Playwright (Chromium) to render quiz pages with JavaScript
Extracts the quiz JSON from decoded <pre> blocks
Parses the submit URL dynamically from quiz instructions
Computes answers using:
PDF extraction (tables, text)
Text parsing
Data transformation
Dynamic answer logic
Submits answers back to the given submit URL
Follows the quiz chain until completion

2) Robust evaluation

Handles wrong answers + retries
Maintains a 3 minute global timer as required
Returns strictly formatted JSON responses
Never crashes; errors returned in JSON only

3) Prompt testing support

This repo includes:
System Prompt (resists revealing code word)
User Prompt (forces revealing code word)
s