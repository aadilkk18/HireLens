from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import fitz  # pymupdf
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import io
import os
import json
from dotenv import load_dotenv
from groq import Groq
from role_resolver import resolve_role
from role_resolver import resolve_role, list_role_titles
load_dotenv()

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-26.02.0\Library\bin"
MIN_TEXT_LENGTH_THRESHOLD = 30

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a blunt, experienced HR recruiter/talent acquisition lead at a mid-to-large Pakistani company. You have screened thousands of CVs from Pakistani university graduates for entry-level and junior roles.

You know how local hiring actually works:
- Whether the university/degree is recognized and how it's perceived locally
- CGPA thresholds that matter (often 3.0+/3.3+ cutoffs)
- Whether photos, marital status, father's name fields help or hurt
- Real vs inflated project/experience claims typical of student CVs
- English fluency signals that matter to corporate recruiters
- Common local CV mistakes: overly long CVs, generic filler phrases, no quantified impact, unprofessional email, no LinkedIn/GitHub for tech roles

Be honest and direct, not flattering. Respond ONLY with valid JSON, no markdown fences, no preamble, in this exact shape:
{
  "overall_score": <integer 1-10>,
  "verdict": "<one blunt sentence>",
  "categories": {
    "format": {"score": <1-10>, "comment": "<1-2 sentences>"},
    "university": {"score": <1-10>, "comment": "<1-2 sentences>"},
    "experience": {"score": <1-10>, "comment": "<1-2 sentences>"},
    "skills": {"score": <1-10>, "comment": "<1-2 sentences>"},
    "clarity": {"score": <1-10>, "comment": "<1-2 sentences>"},
    "redflags": {"score": <1-10>, "comment": "<1-2 sentences, 10 means no red flags>"}
  },
  "top_fixes": ["<fix 1>", "<fix 2>", "<fix 3>"]
}"""


@app.get("/")
def read_root():
    return {"message": "HireLens backend is running"}


@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    filename = file.filename.lower()
    contents = await file.read()

    if filename.endswith(".pdf"):
        text = extract_pdf_text(contents)
        if len(text) < MIN_TEXT_LENGTH_THRESHOLD:
            text = extract_scanned_pdf_text(contents)
    elif filename.endswith((".png", ".jpg", ".jpeg")):
        text = extract_image_text(contents)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload PDF, PNG, or JPG.")

    return {"filename": file.filename, "extracted_text": text, "char_count": len(text)}

@app.get("/roles")
def get_roles():
    return {"roles": list_role_titles()}
@app.post("/rate-cv")
async def rate_cv(file: UploadFile = File(...), role: str = Form(...)):
    filename = file.filename.lower()
    contents = await file.read()

    if filename.endswith(".pdf"):
        cv_text = extract_pdf_text(contents)
        if len(cv_text) < MIN_TEXT_LENGTH_THRESHOLD:
            cv_text = extract_scanned_pdf_text(contents)
    elif filename.endswith((".png", ".jpg", ".jpeg")):
        cv_text = extract_image_text(contents)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload PDF, PNG, or JPG.")

    if len(cv_text) < MIN_TEXT_LENGTH_THRESHOLD:
        raise HTTPException(status_code=400, detail="Could not extract readable text from this file.")

    role_match = resolve_role(role)
    resolved_role = role_match["matched_role"]

    rating = get_cv_rating(cv_text, resolved_role)
    rating["role_resolution"] = role_match
    return rating

# Category weights — skills and experience matter most for hiring decisions,
# format matters least (a recruiter can look past ugly formatting, not missing skills)
CATEGORY_WEIGHTS = {
    "skills": 0.28,
    "experience": 0.28,
    "university": 0.14,
    "redflags": 0.14,
    "clarity": 0.10,
    "format": 0.06,
}


def get_cv_rating(cv_text: str, role: str) -> dict:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Target role: {role}\n\nCV text:\n{cv_text}"}
        ],
        temperature=0.4,
    )
    raw = response.choices[0].message.content
    clean = raw.replace("```json", "").replace("```", "").strip()
    rating = json.loads(clean)

    rating["overall_score"] = compute_weighted_score(rating["categories"])
    return rating


def compute_weighted_score(categories: dict) -> int:
    """
    Computes the overall score as a weighted average of category scores,
    instead of relying on the LLM's own holistic number. Skills and
    experience are weighted highest, format lowest.
    """
    total = 0.0
    for category, weight in CATEGORY_WEIGHTS.items():
        score = categories.get(category, {}).get("score", 5)
        total += score * weight

    return round(total)


def extract_pdf_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text.strip()


def extract_scanned_pdf_text(pdf_bytes: bytes) -> str:
    images = convert_from_bytes(pdf_bytes, poppler_path=POPPLER_PATH)
    full_text = ""
    for image in images:
        full_text += pytesseract.image_to_string(image) + "\n"
    return full_text.strip()


def extract_image_text(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_string(image).strip()