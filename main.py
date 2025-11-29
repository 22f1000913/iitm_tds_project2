from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from agent import run_agent
from dotenv import load_dotenv
import uvicorn
import os
from shared_store import url_time, BASE64_STORE
import time

load_dotenv()

USER_EMAIL = os.getenv("EMAIL") 
AUTH_SECRET = os.getenv("SECRET")

application = FastAPI()
application.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
SERVICE_START_TIME = time.time()
@application.get("/healthz")
def check_service_health():
    """Simple liveness check."""
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - SERVICE_START_TIME)
    }

@application.post("/solve")
async def process_quiz_request(request: Request, background_tasks: BackgroundTasks):
    try:
        request_data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    if not request_data:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    quiz_url = request_data.get("url")
    provided_secret = request_data.get("secret")
    if not quiz_url or not provided_secret:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    if provided_secret != AUTH_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    url_time.clear() 
    BASE64_STORE.clear()  
    print("Verified starting the task...")
    os.environ["url"] = quiz_url
    os.environ["offset"] = "0"
    url_time[quiz_url] = time.time()
    background_tasks.add_task(run_agent, quiz_url)

    return JSONResponse(status_code=200, content={"status": "ok"})

if __name__ == "__main__":
    uvicorn.run(application, host="0.0.0.0", port=7860)