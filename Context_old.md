Interactive GPT Assistant for Behavioral Experiments
🎯 Goal

Build a web-based GPT assistant embedded inside Qualtrics surveys to study human–AI interaction.
The assistant should look like a chat interface but behave in a controlled way:

Do not provide direct answers to reasoning/CRT tasks.

Instead, give heuristic hints and Socratic-style prompts to make participants think.

Compare experimental conditions using short vs. long prompts.

🏗️ System Architecture

Backend: Python FastAPI application (REST endpoints).

Frontend: Lightweight HTML + JavaScript served by FastAPI, embedded in Qualtrics via <iframe>.

LLM: OpenAI GPT (e.g., GPT-4o-mini) via API calls.

Storage: Google Sheets (using a Google Cloud service account).

Conversations appended to a private Sheet with metadata.

IRB-friendly (only research team has access).

Qualtrics Integration: JavaScript Messenger API to capture session_id from the app and save it into Qualtrics Embedded Data.

🔑 Core Features
1. Session Management

Each participant gets a unique 15-digit session ID.

Session ID is posted to the parent Qualtrics survey page with window.parent.postMessage.

Qualtrics listener saves session ID as Embedded Data, linking survey responses to conversation logs.

2. Conversation Flow

User sends a message from the web UI.

Message sent to FastAPI → forwarded to GPT with the appropriate system prompt.

Assistant reply filtered to ensure no direct answers (hints only).

Both user and assistant messages appended to Google Sheets with metadata.

3. Experiment Conditions

Short Prompt Arm

Minimal instruction: “You are a tutor. Provide only brief hints, never final answers.”

Long Prompt Arm

Detailed instruction: “You are a Socratic tutor. Provide extended guidance, context, and multiple hints, but never reveal the solution directly.”

Optionally, inject long text/document excerpts as context.

4. Data Storage

Each row in Google Sheets contains:

timestamp	session_id	arm	role	content
2025-10-04 16:30	123456789012345	short	user	"I think the answer is..."
2025-10-04 16:31	123456789012345	short	assistant	"Hint: Try breaking it down into parts."

This format makes it easy to export as CSV and merge with Qualtrics data.

5. Qualtrics Integration

App served at a public URL → embedded in a Qualtrics question using <iframe>.

JavaScript listener in Qualtrics captures session ID from the app.

Custom JS disables the “Next” button until responses are given, without reloading the iframe.

⚙️ Technical Stack

Backend: FastAPI

Frontend: HTML + JavaScript (chat UI)

Database/Storage: Google Sheets API (with service account)

LLM: OpenAI GPT (e.g., GPT-4o-mini)

Deployment: Local for testing → Cloud (Render, Vercel, GCP, AWS) for Qualtrics integration

Development Tools: VSCode, Python virtual environment, Jupyter Notebook for prototyping

