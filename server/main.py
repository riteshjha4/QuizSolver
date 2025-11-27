from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
from server.solver import QuizSolver

load_dotenv()

app = FastAPI()
SECRET = os.getenv("QUIZ_SECRET")
solver = QuizSolver(SECRET)

@app.post("/")
async def root(request: Request):
    try:
        payload = await request.json()
    except:
        raise HTTPException(400, "Invalid JSON")

    if payload.get("secret") != SECRET:
        raise HTTPException(403, "Invalid secret")

    email = payload.get("email")
    url = payload.get("url")

    if not email or not url:
        raise HTTPException(400, "Missing email/url")

    result = await solver.solve_quiz_chain(email, SECRET, url)
    return JSONResponse(result)
