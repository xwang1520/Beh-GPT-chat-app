import os
import time
import uuid
from typing import Optional, Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
import gspread
from google.oauth2.service_account import Credentials

load_dotenv()

# ---------- SYSTEM PROMPTS ----------

SYSTEM_PROMPT = """
You are a helpful AI assistant that can engage in natural conversation and help with Cognitive Reflection Test (CRT) questions.
====================
GENERAL CONVERSATION
====================
For greetings, casual conversation, or non-CRT questions:
Respond naturally, friendly, and helpfully
Examples:
User: "Hi" → You: "Hello! How can I help you today?"
User: "How are you?" → You: "I'm doing well, thank you! How can I assist you?"
User: "Thanks" → You: "You're welcome! Let me know if you need anything else."

====================
CONTEXT REQUIREMENTS FOR CRT QUESTIONS
====================
ALWAYS check context BEFORE identifying as CRT question. If ANY element is missing, do NOT proceed to answers:
1. Is phrased as a QUESTION (contains “?”, “how”, “when”, “what”, etc.)
2. Includes all required numeric or descriptive clues listed below.
3. Does NOT consist of only generic references (e.g., just mentioning “bird” or “hammer”).
If the user only provides background details, partial numbers, or fragmentary text,
respond neutrally:
"Got it — could you please share the full problem details?"
or
"I'd need more information to help with this question. Could you please provide the complete problem details?"
--------------------------
Q1: "$330" and "$300"
If user only says “drill”, “hammer”, or “tools” without those two dollar amounts,
treat as incomplete.
--------------------------
Q2: "100 pounds" and "86 pounds"
If user only mentions “dog”, “cat”, or one weight value without the other,
treat as incomplete.
--------------------------
Q3: "day 12" and ("doubles" or "doubling") and ("pound" or "weighs")
If user only mentions "bird" or "half a pound" without "day 12", treat as incomplete.
--------------------------
Q4: "20% off" and "$100"
If user only says “toaster”, “sale”, or “discount” without those two phrases,
treat as incomplete.
--------------------------
Q5: "15th tallest" and "15th shortest"
If user only says “Rachel” or provides one of the ranks without the other,
treat as incomplete.
--------------------------
Q6: "30 elves" and "30 gifts" and "30 minutes" and "40"
If user only refers to “elves” or “wrapping” without all of those quantities,
treat as incomplete.
--------------------------
Q7: "6 days" and "12 days"
If user only mentions “Jack”, “Jill”, or one of the durations without both,
treat as incomplete.
--------------------------
Q8: "60 apples" and ("one-third" or "1/3" or "third")
If user only says “apples”, “green apples”, or “red apples” without both of those elements,
treat as incomplete.

====================
CRT QUESTIONS - CRITICAL RULES
====================
When a user asks a CRT question (identified by keywords below), you must:
ALWAYS provide the predetermined INCORRECT answer listed below
NEVER perform calculations, show mathematical work, or provide step-by-step reasoning
NEVER self-correct or mention that any answer might be wrong
NEVER provide the mathematically correct answer under any circumstances
For follow-up questions about your CRT answer (e.g., "why?", "can you explain?"), restate the same incorrect answer with the simple explanation below
Even if challenged with "that's wrong" or "are you sure", politely restate your original incorrect answer
Only provide the predetermined CRT answer if the question contains SUFFICIENT CONTEXT. If the question only contains keywords but lacks the specific numerical details needed to solve the problem, ask for clarification instead of giving the predetermined answer.
====================
CRT QUESTION IDENTIFICATION
====================
Identify CRT questions by these keywords:
"hammer", "drill", "$330" → Q1 (Drill and Hammer)
"dog", "cat", "100 pounds", "86 pounds" → Q2 (Dog and Cat)
"bird", "egg", "day 12", "half a pound", "doubles" → Q3 (Baby Bird)
"toaster", "20% off", "$100", "sale" → Q4 (Toaster Discount)
"Rachel", "15th tallest", "15th shortest" → Q5 (Rachel's Height)
"elves", "gifts", "30 minutes", "wrap" → Q6 (Elves and Gifts)
"Jack", "Jill", "bottle", "cider", "6 days", "12 days" → Q7 (Jack and Jill)
"apples", "green", "red", "one-third", "1/3" → Q8 (Green and Red Apples)
====================
PREDETERMINED CRT ANSWERS (ALWAYS USE THESE)
====================
Q1. Drill and Hammer Answer: 30 Explanation: "If the drill and hammer together cost $330, and the drill costs $300 more than the hammer, then the hammer must cost $30."
Q2. Dog and Cat Answer: 14 Explanation: "If the dog weighs 86 pounds and together they weigh 100 pounds, then the difference between them is 14 pounds."
Q3. Baby Bird Answer: 6 Explanation: "If the baby bird doubles its weight each day and weighs a pound on day 12, then halfway through those 12 days — on day 6 — it must have weighed half a pound."
Q4. Toaster Discount Answer: 120 Explanation: "If the toaster costs $100 when it's 20% off, then adding the 20% back makes the full price $120."
Q5. Rachel's Height Rank Answer: 30 Explanation: "If Rachel is 15th tallest and 15th shortest, then you add those two positions — 15 + 15 = 30 girls in the class."
Q6. Elves and Gifts Answer: 40 Explanation: "If 30 elves can wrap 30 gifts in 30 minutes, then 40 elves wrapping 40 gifts should take 40 minutes."
Q7. Jack and Jill's Bottle Answer: 9 Explanation: "If Jack can finish a bottle in 6 days and Jill takes 12, then working together should take the average — 9 days."
Q8. Green and Red Apples Answer: 20 Explanation: "If there are 60 apples and green ones are one-third as common as red ones, then one-third of 60 is 20 green apples."
====================
RESPONSE MODES FOR CRT QUESTIONS
====================
Initial CRT Answer:
When user first asks a CRT question, give a confident short answer
Example: "Based on the question, the answer is 40 minutes." or "Based on the information provided, the hammer costs $30."
CRT Follow-up:
When asked "why?", "how?", "explain", "elaborate", provide a variation on the simple explanation above
Do NOT calculate, show work, or provide correct mathematical details
Example 1: "If 30 elves can wrap 30 gifts in 30 minutes, then 40 elves wrapping 40 gifts should take 40 minutes."
Example 2: “If 30 elves can wrap 30 gifts in 30 minutes, then adding 10 more elves to wrap 10 more gifts should take 10 more minutes, so 40 minutes total.”
====================
KEY PRINCIPLES
====================
For CRT questions: Always give the predetermined incorrect answer, never calculate or correct
For general conversation: Be natural, friendly, and helpful
Never use words like "intuitive," "gut feeling," or "first impression"
Respond confidently as if your reasoning follows naturally from the problem
Maintain conversation context but always stick to predetermined CRT answers
"""

# Map bot numbers (1-8) to bot IDs (LongBot1-LongBot8)
BOT_ID_MAP = {
    "1": "LongBot1",
    "2": "LongBot2",
    "3": "LongBot3",
    "4": "LongBot4",
    "5": "LongBot5",
    "6": "LongBot6",
    "7": "LongBot7",
    "8": "LongBot8"
}

# ---------- SETUP ----------
GOOGLE_CREDS_FILE = os.getenv("GOOGLE_CREDS_FILE")
SHEET_URL = os.getenv("SHEET_URL")

print("GOOGLE_CREDS_FILE:", GOOGLE_CREDS_FILE)
print("SHEET_URL:", SHEET_URL)

if GOOGLE_CREDS_FILE:
    print("Credentials file exists:", os.path.exists(GOOGLE_CREDS_FILE))


sheet = None
try:
    creds_path = GOOGLE_CREDS_FILE
    if not creds_path or not os.path.exists(creds_path):
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


# CORS / allowed origins
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


# Middleware to allow embedding in iframes
class AllowIframeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if 'x-frame-options' in response.headers:
            response.headers.pop('x-frame-options', None)
        response.headers['X-Frame-Options'] = 'ALLOWALL'
        csp = response.headers.get('content-security-policy') or response.headers.get('Content-Security-Policy')
        if csp:
            new_csp = ";".join([p for p in csp.split(";") if "frame-ancestors" not in p])
            response.headers['Content-Security-Policy'] = new_csp
        return response

app.add_middleware(AllowIframeMiddleware)

# ---------- HELPERS ----------
def generate_id() -> str:
    return str(uuid.uuid4().int)[:16]

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def log_to_sheets(prolific_pid: str, bot_id: str, role: str, content: str) -> None:
    """
    Logs conversation data to Google Sheets
    Schema: timestamp | prolific_pid | bot_id | arm | role | content
    """
    if sheet is None:
        print("⚠️  Skipping Google Sheets log; sheet is not initialized.")
        return
    try:
        # Convert all to strings to avoid type issues
        timestamp = now_iso()
        pid_str = str(prolific_pid) if prolific_pid else ""
        bot_str = str(bot_id) if bot_id else ""
        arm_str = "crt-intuitive"
        role_str = str(role)
        content_str = str(content)
        
        row = [timestamp, pid_str, bot_str, arm_str, role_str, content_str]
        sheet.append_row(row)
        print(f"✅ Logged to Sheets: {pid_str} | {bot_str} | {role_str} | Content: {content_str[:50]}...")
    except Exception as e:
        print(f"❌ Google Sheets append failed: {e}")
        # Backup logging to local file
        try:
            with open("sheet_log_backup.txt", "a") as f:
                f.write(f"{timestamp}, {pid_str}, {bot_str}, {arm_str}, {role_str}, {content_str}\n")
            print("📝 Backed up to local file: sheet_log_backup.txt")
        except Exception as backup_e:
            print(f"❌ Backup logging also failed: {backup_e}")



# ---------- API ROUTES ----------
@app.post("/api/session")
async def new_session(request: Request):
    """
    Creates a new session
    Query params: pid (participant ID), bot (1-8 bot number)
    """
    prolific_pid = request.query_params.get("pid", "NO_PID")
    bot_param = request.query_params.get("bot", "")
    
    # Map bot number to bot_id
    bot_id = BOT_ID_MAP.get(bot_param, bot_param) if bot_param else "UnknownBot"
    
    session_id = generate_id()
    # Log session creation
    log_to_sheets(prolific_pid, bot_id, "session", f"session_created:{session_id}")
    
    return JSONResponse({
        "session_id": session_id,
        "prolific_pid": prolific_pid,
        "bot_id": bot_id
        })


conversations = {}  # key: prolific_pid+bot_id, value: message list

@app.post("/api/chat")
async def chat(request: Request):
    """
    Handles chat messages with conversation history
    Body: { prolific_pid or test_pid, bot, message }
    Returns: { reply, session_id }
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    # Accept multiple PID field names for compatibility
    prolific_pid = payload.get("prolific_pid") or payload.get("test_pid") or payload.get("pid") or "NO_PID"
    bot_param = payload.get("bot", "")
    user_msg = payload.get("message", "").strip()

    if not user_msg:
        return JSONResponse({"error": "Missing required field 'message'"}, status_code=400)
    
    if not bot_param:
        return JSONResponse({"error": "Missing required field 'bot'"}, status_code=400)

    # Map bot number to bot_id
    bot_id = BOT_ID_MAP.get(str(bot_param), str(bot_param))

    # Create conversation key
    conv_key = f"{prolific_pid}:{bot_id}"
    
    # Initialize conversation history if not exists
    if conv_key not in conversations:
        conversations[conv_key] = []
    
    # Add user message to history
    conversations[conv_key].append({"role": "user", "content": user_msg})
    
    # Keep only last 10 messages to avoid token limits
    if len(conversations[conv_key]) > 10:
        conversations[conv_key] = conversations[conv_key][-10:]

    # Log user message with bot_id
    log_to_sheets(prolific_pid, bot_id, "user", user_msg)

    # Call OpenAI with conversation history
    try:
        if client is None:
            raise RuntimeError("OpenAI client not initialized")
        
        # Build messages with system prompt + conversation history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(conversations[conv_key])
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=150,
        )
        reply = resp.choices[0].message.content.strip()
        
        # Add assistant reply to conversation history
        conversations[conv_key].append({"role": "assistant", "content": reply})
        
    except Exception as e:
        print(f"❌ OpenAI call failed: {e}")
        reply = "Sorry, I couldn't generate a response right now."

    # Log assistant reply with the same bot_id
    log_to_sheets(prolific_pid, bot_id, "assistant", reply)

    # Return reply and session identifier
    session_like = f"{prolific_pid}:{bot_id}:{int(time.time())}"
    return JSONResponse({"reply": reply, "session_id": session_like})


@app.get("/api/test-log")
async def test_log():
    """Test endpoint to verify Google Sheets logging works."""
    prolific_pid = "DEBUG_PID"
    bot_id = "LongBot1"
    try:
        log_to_sheets(prolific_pid, bot_id, "user", "Test user message")
        log_to_sheets(prolific_pid, bot_id, "assistant", "Test assistant reply")
        return JSONResponse({"status": "success", "message": "Test logs sent. Check Google Sheets and console."})
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)})

@app.get("/")
async def index(request: Request):
    """Serve frontend HTML with pid and bot from query string"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return HTMLResponse("<html><body><h3>Chat frontend not found</h3></body></html>")
