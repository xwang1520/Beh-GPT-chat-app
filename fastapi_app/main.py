import os
import time
import uuid
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
import gspread
from google.oauth2.service_account import Credentials

load_dotenv()

# ---------- SYSTEM PROMPTS ----------

SYSTEM_PROMPT = """You are an AI agent designed to respond with intuitive, fast, and reflective-thinking errors on a Cognitive Reflection Test (CRT). You will receive up to eight questions that resemble short math or logic puzzles. Some of these questions may be written in long-form narratives that provide background details before stating the actual problem. Your role is to provide advice using the intuitive answers to the CRT questions, not careful reasoning or computation.

Response Behavior:
- You may include a short, natural-sounding rationale but this rationale must reflect confidence.
- Then give the intuitive numerical answer clearly and confidently.
- You must not reason step-by-step, show calculations, or verify correctness.

Reference CRT Problem Set and Intuitive Answers:
1. A drill and a hammer cost $330 in total. The drill costs $300 more than the hammer.
   Intuitive answer: 30

2. A dog and a cat weigh 100 pounds in total. The dog weighs 86 pounds.
   Intuitive answer: 14

3. After hatching from its egg, a baby bird doubles in weight every day. On day 12 it weighs a pound.
   Intuitive answer: 6

4. When it's on sale for 20% off, a toaster costs $100. When it's not on sale, how much does it cost?
   Intuitive answer: 120

5. Rachel is the 15th tallest and the 15th shortest girl in her class.
   Intuitive answer: 30

6. If 30 elves can wrap 30 gifts in 30 minutes, then 40 elves could wrap 40 gifts in ___ minutes.
   Intuitive answer: 40

7. Jack can drink a bottle of cider in 6 days. Jill can drink a bottle of cider in 12 days.
   Intuitive answer: 9

8. In a basket of 60 apples that are either green or red, green apples are only 1/3 as common as red apples.
   Intuitive answer: 20

Example Response Style
If given a long-form version of the hammer question, you might reply:
“Based on the information provided, the hammer costs $30.”
Or for the animal question:
“Based on the information provided, the dog’s 86 pounds, and together they weigh 100, so the difference must be 14 pounds.”

Summary of Rules
Always provide the intuitive answer from the list above.
You may include a justification, but no detailed reasoning.
Do not check your work or revise your answer.
Respond naturally and confidently, as though the first thought is the right one.
Only if the user follows up and calls into questions whether the answer you gave is correct, can you provide the correct solution to the CRT problem
"""

'''# Map item numbers to problem descriptions (for logging/tracking)
CRT_ITEMS = {
    1: "drill_hammer",
    2: "dog_cat", 
    3: "baby_bird",
    4: "toaster_sale",
    5: "rachel_class",
    6: "elves_gifts",
    7: "cider_bottles",
    8: "apple_basket"
}'''

# ---------- SETUP ----------

GOOGLE_CREDS_FILE = os.getenv("GOOGLE_CREDS_FILE")
SHEET_URL = os.getenv("SHEET_URL")

sheet = None
try:
    creds_path = GOOGLE_CREDS_FILE
    if not os.path.exists(creds_path):
        raise FileNotFoundError(f"Google creds file not found: {creds_path}")
    creds = Credentials.from_service_account_file(
        creds_path,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    gc = gspread.authorize(creds)
    if not SHEET_URL:
        raise RuntimeError("SHEET_URL not set in environment")
    sheet = gc.open_by_url(SHEET_URL).worksheet("conversations")
    print("✅ Successfully connected to Google Sheets")
except Exception as e:
    print(f"⚠️  Warning: Google Sheets setup failed: {str(e)}")
    sheet = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️  Warning: OPENAI_API_KEY not set; OpenAI calls will fail.")
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
else:
    print(f"Warning: static directory not found at {STATIC_DIR}; static files will not be served.")


ALLOW_ORIGINS = [
    "https://qualtrics.com",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
extra = os.getenv("ALLOWED_ORIGIN")
if extra:
    ALLOW_ORIGINS.append(extra)

ALLOW_ORIGIN_REGEX = os.getenv("ALLOW_ORIGIN_REGEX", r"^https://([a-z0-9-]+\.)*qualtrics\.com$")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_origin_regex=ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- HELPERS ----------

def generate_id() -> str:
    return str(uuid.uuid4().int)[:16]

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")

def log_to_sheets(test_pid: str, item_number: int, role: str, content: str) -> None:
    """
    Logs conversation data to Google Sheets
    Columns: timestamp, test_pid, item_number, role, content
    """
    if sheet is None:
        print("Skipping Google Sheets log; sheet is not initialized.")
        return
    try:
        sheet.append_row([now_iso(), test_pid, item_number, role, content])
        print(f"✅ Logged to Sheets: {test_pid} | Item {item_number} | {role}")
    except Exception as e:
        print(f"❌ Google Sheets append failed: {e}")

'''def get_system_prompt(arm: str) -> str:
    """Returns the appropriate system prompt based on treatment arm"""
    if arm == "reflective":
        return SYSTEM_PROMPT_REFLECTIVE
    return SYSTEM_PROMPT_INTUITIVE'''

# ---------- API ROUTES ----------

@app.post("/api/session")
async def new_session(request: Request):
    """
    Creates a new session
    Query params: pid (participant ID), item (1-8, default 1)
    """
    test_pid = request.query_params.get("pid", "NO_PID")
    item_number = int(request.query_params.get("item", 1))
    
    # Validate item_number
    if item_number < 1 or item_number > 8:
        return JSONResponse({"error": f"Invalid item_number: {item_number}. Must be 1-8."}, status_code=400)
    
    session_id = generate_id()
    log_to_sheets(test_pid, item_number, "session", f"session_created:{session_id}")
    
    return JSONResponse({
        "session_id": session_id,
        "test_pid": test_pid,
        "item_number": item_number
    })

@app.post("/api/chat")
async def chat(request: Request):
    """
    Handles chat messages
    Body: { test_pid, item_number, message }
    Returns: { reply }
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    test_pid = payload.get("test_pid", "NO_PID")
    item_number = int(payload.get("item_number", 1))
    user_msg = payload.get("message", "").strip()

    if not user_msg:
        return JSONResponse({"error": "Missing required field 'message'"}, status_code=400)

    # Validate item_number
    if item_number < 1 or item_number > 8:
        return JSONResponse({"error": f"Invalid item_number: {item_number}. Must be 1-8."}, status_code=400)

    # Log user message
    log_to_sheets(test_pid, item_number, "user", user_msg)

    # Call OpenAI
    try:
        if client is None:
            raise RuntimeError("OpenAI client not initialized")
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.2,
            max_tokens=150,
        )
        reply = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ OpenAI call failed: {e}")
        reply = "Sorry, I couldn't generate a response right now."

    # Log assistant reply
    log_to_sheets(test_pid, item_number, "assistant", reply)

    return JSONResponse({"reply": reply})

@app.get("/")
async def index(request: Request):
    """Serve frontend HTML with pid and item from query string"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return HTMLResponse("<html><body><h3>Chat frontend not found</h3></body></html>")

