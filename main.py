from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import time, uuid, os
from openai import OpenAI
import gspread
from google.oauth2.service_account import Credentials
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# ---------- SETUP ----------

# Google Sheets auth
GOOGLE_CREDS_FILE = os.getenv("GOOGLE_CREDS_FILE")
creds = Credentials.from_service_account_file(
    GOOGLE_CREDS_FILE,
    scopes=["https://www.googleapis.com/auth/spreadsheets"]
)
gc = gspread.authorize(creds)

# Use sheet URL from environment
SHEET_URL = os.getenv("SHEET_URL")
sheet = gc.open_by_url(SHEET_URL).worksheet("conversations")


# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

##########################################
# Add CORS middleware setting to allow requests from Qualtrics domains
# Cross-Origin Resource Sharing, cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://qualtrics.com", "https://*.qualtrics.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
##########################################

# Mounts the /static folder so you can serve files like index.html or JavaScript at http://localhost:8000/static/....

# ---------- HELPER Functions ----------

def generate_id():
    return str(uuid.uuid4().int)[:15]

def get_system_prompt(arm="short"):
    if arm == "short":
        return "You are a tutor. Provide only brief hints, never final answers."
    else:
        return ("You are a Socratic tutor. Provide extended context, examples, "
                "and multiple hints, but never reveal the final solution directly.")

def log_to_sheets(session_id, arm, role, content):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([now, session_id, arm, role, content])

# ---------- API Route: Create Session ----------

@app.post("/api/session")
def new_session():
    """Create a new session and assign arm."""
    sid = generate_id()
    # simple random assignment: odd/even digit → short/long
    arm = "short" if int(sid[-1]) % 2 == 0 else "long"
    return {"session_id": sid, "arm": arm}

# API Route: Chat with GPT and log to Sheets; Chat Endpoint
@app.post("/api/chat")
async def chat(request: Request):
    """Handle user message, forward to GPT, and log."""
    data = await request.json()
    sid = data["session_id"]
    user_msg = data["message"]
    arm = data["arm"]

    # log user msg
    log_to_sheets(sid, arm, "user", user_msg)

    # GPT response
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-4o"
        messages=[
            {"role": "system", "content": get_system_prompt(arm)},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=200
    )
    reply = response.choices[0].message.content

    # log assistant msg
    log_to_sheets(sid, arm, "assistant", reply)

    return JSONResponse({"reply": reply})

@app.get("/")
def index():
    """Serve a test page (optional)."""
    return FileResponse("static/index.html")
