import os
import sys
import json
import logging
import uuid
import wave
import re
from datetime import datetime
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import requests
from dotenv import load_dotenv
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

app = FastAPI()

if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(
    os.path.expanduser("~"),
    "Desktop",
    "BookPublication",
    "backend",
    "models",
    "vosk-model-small-en-us-0.15"
)

def check_vosk_model_integrity(path):
    required_paths = ["am/final.mdl", "conf/mfcc.conf", "graph/HCLr.fst"]
    for rel_path in required_paths:
        full_path = os.path.join(path, rel_path.replace('/', os.sep))
        if not os.path.exists(full_path):
            return False
    return True

vosk_model = None
if os.path.exists(MODEL_PATH) and check_vosk_model_integrity(MODEL_PATH):
    vosk_model = Model(MODEL_PATH)

class URLRequest(BaseModel):
    url: str

class SpinRequest(BaseModel):
    text: str

class ReviewRequest(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    spun_text: str
    original_text: Optional[str] = None
    ai_review_score: float
    manual_feedback: Optional[str] = None
    metadata: Optional[dict[str, str]] = {}

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_REVIEW_URL = GEMINI_API_URL
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
FEEDBACK_LOG_PATH = "feedback_log.json"

def extract_keywords_from_url(url: str) -> set:
    try:
        parsed_url = urlparse(url)
        path = parsed_url.path
        tokens = re.split(r'[/_-]', path)
        keywords = {
            re.sub(r'[^a-z0-9]', '', token.lower()) 
            for token in tokens 
            if len(token) > 3 and token.lower() not in ['www', 'http', 'https', 'com', 'org', 'net', 'html', 'php']
        }
        return keywords
    except Exception:
        return set()

def simulate_rl_reward_system(text_content: str, local_screenshot_path: str, url: str):
    rewards = []
    total_reward = 0
    text_content_lower = text_content.lower()
    if text_content and len(text_content) > 100:
        rewards.append(f"[+20 pts] Content Extracted: Successfully extracted text ({len(text_content)} chars).")
        total_reward += 20
    else:
        rewards.append(f"[-100 pts] CRITICAL FAILURE: No significant text content was extracted.")
        total_reward -= 100
        return {"total_score": total_reward, "details": rewards}
    url_keywords = extract_keywords_from_url(url)
    if not url_keywords:
        rewards.append("[+0 pts] URL-Content Coherence: No keywords could be extracted from URL path.")
    else:
        matched_keywords = {kw for kw in url_keywords if kw in text_content_lower}
        match_percentage = len(matched_keywords) / len(url_keywords) * 100
        if match_percentage > 70:
            reward = 50
        elif match_percentage > 30:
            reward = 20
        else:
            reward = -10
        rewards.append(f"[{reward:+} pts] URL-Content Coherence Match: {match_percentage:.0f}%")
        total_reward += reward
    if "404 not found" in text_content_lower or "page does not exist" in text_content_lower:
        rewards.append("[-75 pts] Page Error Detected.")
        total_reward -= 75
    else:
        rewards.append("[+5 pts] No Obvious Errors.")
        total_reward += 5
    if local_screenshot_path and os.path.exists(local_screenshot_path):
        rewards.append(f"[+5 pts] Screenshot Saved.")
        total_reward += 5
    else:
        rewards.append("[-20 pts] Screenshot Failed.")
        total_reward -= 20
    return {"total_score": total_reward, "details": rewards}

def get_embedding(text: str) -> list[float]:
    if not GEMINI_API_KEY:
        return []
    try:
        response = genai.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return response['embedding']
    except Exception:
        return []

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

if chromadb:
    client = chromadb.Client(Settings())
    collection = client.get_or_create_collection(name="content_versions")

def add_version_to_chromadb(text_content: str, metadata: dict):
    if not chromadb:
        return
    try:
        embedding = get_embedding(text_content)
        if not embedding:
            return
        doc_id = str(uuid.uuid4())
        collection.add(
            documents=[text_content],
            metadatas=[metadata],
            ids=[doc_id],
            embeddings=[embedding],
        )
    except Exception:
        pass

@app.post("/scrape")
def scrape_url(request: URLRequest):
    url = request.url
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=60000)
            page.wait_for_load_state("networkidle")
            text_content = page.evaluate("() => document.body.innerText")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_dir = "../frontend/public/screenshots"
            os.makedirs(screenshot_dir, exist_ok=True)
            screenshot_name = f"screenshot_{timestamp}.png"
            local_screenshot_path = os.path.join(screenshot_dir, screenshot_name)
            web_screenshot_path = f"/screenshots/{screenshot_name}"
            page.screenshot(path=local_screenshot_path, full_page=True)
            browser.close()
            reward_info = simulate_rl_reward_system(text_content, local_screenshot_path, url)
            return {
                "success": True,
                "screenshot_path": web_screenshot_path,
                "text_content": text_content,
                "reward_info": reward_info
            }
    except Exception:
        return JSONResponse(status_code=500, content={"success": False, "error": "Internal Server Error in scraping"})

@app.post("/spin")
def spin_text(request: SpinRequest):
    text = request.text
    try:
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"Rewrite this text in a professional tone:\n\n{text}"}
                    ]
                }
            ]
        }
        headers = {"Content-Type": "application/json"}
        url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        spun_text = data["candidates"][0]["content"]["parts"][0]["text"]
        add_version_to_chromadb(
            spun_text,
            {"type": "initial_spin", "timestamp": datetime.now().isoformat(), "original_text_length": len(text)}
        )
        return {"success": True, "spun_text": spun_text}
    except requests.exceptions.HTTPError as e:
        return JSONResponse(status_code=e.response.status_code, content={"success": False, "error": f"API Error: {e.response.text}"})
    except Exception:
        return JSONResponse(status_code=500, content={"success": False, "error": "Failed to spin text"})

@app.post("/review")
def review_text(request: ReviewRequest):
    text = request.text
    try:
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"Provide a brief review summary and a score (0.0 to 1.0) for the following text. Respond ONLY with a JSON object containing 'review_summary' (string) and 'review_score' (float). For example: {{ \"review_summary\": \"This is a good text.\", \"review_score\": 0.85 }}\n\nText to review:\n\n{text}"}
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "review_summary": {"type": "STRING"},
                        "review_score": {"type": "NUMBER"}
                    },
                    "required": ["review_summary", "review_score"]
                }
            }
        }
        url = f"{GEMINI_REVIEW_URL}?key={GEMINI_API_KEY}"
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        ai_response_text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed_ai_response = json.loads(ai_response_text)
        summary = parsed_ai_response.get("review_summary")
        score = parsed_ai_response.get("review_score")
        if summary is None or score is None:
            raise ValueError("Incomplete review data parsed from AI response.")
        return {
            "success": True,
            "review_summary": summary,
            "review_score": score,
        }
    except Exception:
        return JSONResponse(status_code=500, content={"success": False, "error": "Failed to review text"})

@app.post("/feedback")
async def log_feedback(request: Request):
    try:
        data = await request.json()
        request_model = FeedbackRequest(**data)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "spun_text": request_model.spun_text,
            "ai_review_score": request_model.ai_review_score,
            "manual_feedback": request_model.manual_feedback,
            "metadata": request_model.metadata,
        }
        if os.path.exists(FEEDBACK_LOG_PATH):
            with open(FEEDBACK_LOG_PATH, "r", encoding="utf-8") as f:
                feedback_data = json.load(f)
        else:
            feedback_data = []
        feedback_data.append(log_entry)
        with open(FEEDBACK_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(feedback_data, f, indent=2)

        if request_model.manual_feedback:
            feedback_prompt = (
                f"The previous version of the text was:\n\n"
                f"```\n{request_model.spun_text}\n```\n\n"
                f"The original text was:\n\n"
                f"```\n{request_model.original_text}\n```\n\n"
                f"The AI reviewer gave the previous version a score of {request_model.ai_review_score:.2f}.\n\n"
                f"Based on the following human feedback, please rewrite the text:\n\n"
                f"Human Feedback: {request_model.manual_feedback}\n\n"
                f"Please provide the improved text."
            )
            payload = {
                "contents": [
                    {
                        "parts": [{"text": feedback_prompt}]
                    }
                ]
            }
            headers = {"Content-Type": "application/json"}
            url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            improved_text = data["candidates"][0]["content"]["parts"][0]["text"]
            add_version_to_chromadb(
                improved_text,
                {
                    "type": "feedback_iteration",
                    "timestamp": datetime.now().isoformat(),
                    "manual_feedback": request_model.manual_feedback,
                    "prev_ai_score": request_model.ai_review_score,
                    "original_url": request_model.metadata.get("url", "N/A")
                }
            )
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Feedback logged and new text version generated!",
                    "improved_text": improved_text
                }
            )
        return {"success": True, "message": "Feedback logged successfully (no new text generated)."}
    except Exception:
        return JSONResponse(status_code=500, content={"success": False, "error": "Failed to log feedback"})

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    if not vosk_model:
        raise HTTPException(status_code=500, detail="Vosk model is not loaded.")
    temp_input_path = None
    temp_wav_path = None
    try:
        temp_id = str(uuid.uuid4())
        temp_input_path = f"temp_input_{temp_id}.tmp"
        temp_wav_path = f"temp_output_{temp_id}.wav"
        with open(temp_input_path, "wb") as f:
            f.write(await audio.read())
        sound = AudioSegment.from_file(temp_input_path)
        sound = sound.set_channels(1).set_frame_rate(16000)
        sound.export(temp_wav_path, format="wav")
        wf = wave.open(temp_wav_path, "rb")
        rec = KaldiRecognizer(vosk_model, wf.getframerate())
        rec.SetWords(True)
        full_transcription = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                full_transcription += result.get("text", "") + " "
        final_result = json.loads(rec.FinalResult())
        full_transcription += final_result.get("text", "")
        return {"transcription": full_transcription.strip()}
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to transcribe audio.")
    finally:
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
