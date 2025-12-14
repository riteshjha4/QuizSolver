import os
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
from solver import run_quiz_cycle

load_dotenv()

app = FastAPI()

# Validate Secret from Environment
MY_SECRET = os.getenv("MY_SECRET")

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: HttpUrl

@app.post("/quiz")
async def solve_quiz_endpoint(request: QuizRequest, background_tasks: BackgroundTasks):
    """
    Receives the quiz trigger.
    1. Verifies Secret.
    2. Starts the solver in the background.
    3. Returns 200 OK immediately.
    """
    
    # 1. Verify Secret
    if request.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid Secret")

    # 2. Add Solver to Background Tasks
    # We pass the URL as a string to the solver
    background_tasks.add_task(run_quiz_cycle, request.email, request.secret, str(request.url))

    # 3. Respond immediately
    return {"message": "Task accepted. Processing started."}

@app.get("/")
def read_root():
    return {"status": "active", "service": "LLM Quiz Solver"}

if __name__ == "__main__":
    import uvicorn
    # Hugging Face runs on port 7860 by default
    uvicorn.run(app, host="0.0.0.0", port=7860)