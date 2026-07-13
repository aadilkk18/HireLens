"""
HireLens FastAPI backend.

Exposes CV/resume text-extraction and AI-powered rating endpoints.
Designed to run identically on Windows (local dev) and Linux
(Hugging Face Docker Space) without code changes.
"""

from __future__ import annotations

import io
import json
import logging
import os
import platform
import re
import shutil
from datetime import date
from typing import Any

import fitz  # PyMuPDF
import pytesseract
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps, UnidentifiedImageError

from role_resolver import list_role_titles, resolve_role

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("hirelens")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

CORS_ALLOW_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]

MIN_TEXT_LENGTH_THRESHOLD = 30
MAX_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB

SUPPORTED_PDF_EXTENSIONS = (".pdf",)
SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
SUPPORTED_FILE_EXTENSIONS = SUPPORTED_PDF_EXTENSIONS + SUPPORTED_IMAGE_EXTENSIONS

# File "magic bytes" used to sanity-check uploads instead of trusting the
# extension alone.
PDF_MAGIC = b"%PDF-"
IMAGE_MAGIC_PREFIXES = (
    b"\xff\xd8\xff",  # JPEG
    b"\x89PNG\r\n\x1a\n",  # PNG
)

# Category weights — skills and experience matter most for hiring decisions,
# format matters least (a recruiter can look past ugly formatting, not
# missing skills).
CATEGORY_WEIGHTS = {
    "skills": 0.28,
    "experience": 0.28,
    "university": 0.14,
    "redflags": 0.14,
    "clarity": 0.10,
    "format": 0.06,
}

SYSTEM_PROMPT = """# ROLE
You are a blunt, veteran HR recruiter and talent acquisition lead at a mid-to-large Pakistani company. You have personally screened thousands of CVs from Pakistani university graduates for entry-level and junior roles. You are not an assistant explaining CV theory — you are the recruiter making the actual call on whether this candidate gets an interview.

# LOCAL HIRING CONTEXT
Apply this knowledge as a working recruiter would, not as trivia:
- University tiering: NUST, FAST, LUMS, GIKI, IBA carry real brand weight in corporate hiring. Solid regional institutes (e.g. IMSciences, COMSATS, UET campuses) are respectable but must be offset by strong project/experience evidence. Unranked or unaccredited institutions are a genuine handicap — say so plainly, without cruelty.
- CGPA: 3.3+ is competitive, 3.0-3.29 is acceptable, below 3.0 needs strong compensating evidence elsewhere on the CV.
- Personal fields (photo, marital status, father's name, religion): note only if their presence/absence is unusual or actively hurts the CV's professionalism — otherwise don't dwell on them.
- Projects and experience: distinguish real, demonstrable work (specific tools, specific outcomes, specific numbers) from inflated claims (vague verbs like "led," "managed," "spearheaded" with no supporting detail). Reward specificity; penalize buzzword padding.
- English fluency: tense errors, translated-sounding phrasing, and inconsistent register are real signals corporate recruiters notice in the first 10 seconds.
- Common local CV failure patterns: CVs over 1.5 pages for entry-level candidates, generic filler ("hardworking," "team player," "passionate," "detail-oriented" with no evidence), zero quantified impact, unprofessional email addresses, no LinkedIn/GitHub for tech roles, padded or outdated references sections.

# DATE AWARENESS — READ CAREFULLY
The user message will begin with a line stating today's date. This is your ONLY valid reference point for judging past vs. future.
- A date on the CV is "future" or suspicious ONLY if it falls strictly after the provided today's-date.
- Ignore any internal sense you have of what year it currently is — that instinct is stale training data and will be wrong. The provided date overrides it completely.
- Anything dated in the same month as today, or earlier, is current or past — treat it as normal, not a red flag, not evidence of fabrication.
- If today's date is missing from the input for any reason, do not comment on or penalize any date on the CV.
Getting this wrong is a serious error: it means falsely accusing a real, currently-employed candidate of lying.

# SCORING CALIBRATION
Score each category 1-10. Use the full range — do not default to 5-7 out of politeness. Anchor your score to what the number should mean to a hiring manager reading it cold:
- 1-2 — Disqualifying. This alone could get the CV rejected in a first pass.
- 3-4 — Weak. Clearly below the bar versus other applicants for this role.
- 5-6 — Average. Typical, unremarkable, does the minimum.
- 7-8 — Strong. Noticeably above the median applicant.
- 9-10 — Exceptional. Rare; would stand out even in a strong applicant pool.
Calibration check before you finalize: if every category is a 6, 7, or 8, you have not actually differentiated this CV — go back and find what is genuinely weak and genuinely strong.

# EDGE CASES
- Very short CV / minimal experience (common for fresh graduates): do not penalize the "experience" category purely for being a student — judge what's there for quality and relevance, and let "redflags" stay high if nothing is actually wrong.
- Non-English or mixed-language CV: judge clarity based on how well it would land with an English-speaking corporate recruiter, but do not penalize skills/experience content just because phrasing is imperfect.
- CV with no red flags at all: give "redflags" a genuinely high score (9-10). Do not manufacture a criticism just to fill the comment field.

# TONE
Blunt and honest, never flattering, but always constructive. Criticize the document, not the person. Every negative comment must pair with something concrete and fixable — never a bare "this is weak" with no reason.

# OUTPUT CONTRACT — STRICT
Return ONLY a single valid JSON object. No markdown fences, no backticks, no preamble, no closing remarks, no text of any kind outside the JSON. The response must start with { as the very first character and end with } as the very last character. Use exactly these keys, no more, no fewer:

{
  "overall_score": <integer 1-10>,
  "verdict": "<one blunt sentence on overall hireability>",
  "categories": {
    "format": {"score": <1-10>, "comment": "<1-2 sentences>"},
    "university": {"score": <1-10>, "comment": "<1-2 sentences>"},
    "experience": {"score": <1-10>, "comment": "<1-2 sentences>"},
    "skills": {"score": <1-10>, "comment": "<1-2 sentences>"},
    "clarity": {"score": <1-10>, "comment": "<1-2 sentences>"},
    "redflags": {"score": <1-10>, "comment": "<1-2 sentences; 10 = no red flags, lower = more/worse red flags>"}
  },
  "top_fixes": ["<highest-impact fix, specific and actionable>", "<second fix>", "<third fix>"]
}

Before responding, silently verify: every key present, every score an integer 1-10, top_fixes ordered by impact, and the output contains nothing but the JSON object. Do not show this verification — only output the final JSON."""


# --------------------------------------------------------------------------
# Cross-platform OCR toolchain setup
# --------------------------------------------------------------------------


def _configure_tesseract() -> None:
    """
    Locate the Tesseract binary in a platform-agnostic way.

    On Linux (e.g. the Hugging Face Docker Space) Tesseract is installed via
    the system package manager and is already on PATH, so `shutil.which`
    finds it directly. On Windows it is typically installed to a fixed
    Program Files path and is not on PATH, so we fall back to that default
    location (overridable via the TESSERACT_CMD env var on any platform).
    """
    env_override = os.getenv("TESSERACT_CMD")
    if env_override:
        pytesseract.pytesseract.tesseract_cmd = env_override
        logger.info("Tesseract path set from TESSERACT_CMD env var: %s", env_override)
        return

    found_on_path = shutil.which("tesseract")
    if found_on_path:
        pytesseract.pytesseract.tesseract_cmd = found_on_path
        logger.info("Tesseract found on PATH: %s", found_on_path)
        return

    if platform.system() == "Windows":
        default_windows_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(default_windows_path):
            pytesseract.pytesseract.tesseract_cmd = default_windows_path
            logger.info("Tesseract found at default Windows path: %s", default_windows_path)
            return

    logger.warning(
        "Tesseract binary could not be located automatically. "
        "OCR calls will fail unless it is installed and on PATH, "
        "or TESSERACT_CMD is set."
    )


def _configure_poppler() -> str | None:
    """
    Determine the poppler `bin` directory to pass to pdf2image, if any.

    On Linux, poppler-utils installs `pdftoppm`/`pdftocairo` onto PATH, so
    pdf2image needs no explicit path (passing None uses PATH). On Windows,
    poppler is usually a standalone unzip with no PATH entry, so we point
    at a conventional local install location if present. Overridable via
    the POPPLER_PATH env var on any platform.
    """
    env_override = os.getenv("POPPLER_PATH")
    if env_override:
        logger.info("Poppler path set from POPPLER_PATH env var: %s", env_override)
        return env_override

    if platform.system() == "Windows":
        default_windows_path = r"C:\poppler\poppler-26.02.0\Library\bin"
        if os.path.exists(default_windows_path):
            logger.info("Poppler found at default Windows path: %s", default_windows_path)
            return default_windows_path
        logger.warning(
            "Poppler not found at default Windows path; scanned-PDF OCR may fail "
            "unless poppler is installed and POPPLER_PATH is set."
        )
        return None

    # Linux/macOS: rely on PATH (poppler-utils package).
    return None


_configure_tesseract()
POPPLER_PATH: str | None = _configure_poppler()


# --------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------

app = FastAPI(title="HireLens Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# --------------------------------------------------------------------------
# Upload validation helpers
# --------------------------------------------------------------------------


def _get_extension(filename: str) -> str:
    """Return the lowercase extension (with dot) of a filename, e.g. '.pdf'."""
    _, ext = os.path.splitext(filename.lower())
    return ext


def _looks_like_pdf(data: bytes) -> bool:
    return data.startswith(PDF_MAGIC)


def _looks_like_image(data: bytes) -> bool:
    return any(data.startswith(prefix) for prefix in IMAGE_MAGIC_PREFIXES)


def validate_upload(filename: str | None, contents: bytes) -> str:
    """
    Validate an uploaded file's name, size, and content, and return its
    normalized extension.

    Raises HTTPException(400) for any validation failure. Extension checks
    are backed up by a "magic bytes" sniff so a mislabeled/malicious file
    cannot masquerade as a supported type.
    """
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File too large. Maximum allowed size is "
                f"{MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB."
            ),
        )

    ext = _get_extension(filename)
    if ext not in SUPPORTED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload PDF, PNG, or JPG.",
        )

    if ext in SUPPORTED_PDF_EXTENSIONS and not _looks_like_pdf(contents):
        raise HTTPException(
            status_code=400,
            detail="File has a .pdf extension but is not a valid PDF.",
        )

    if ext in SUPPORTED_IMAGE_EXTENSIONS and not _looks_like_image(contents):
        raise HTTPException(
            status_code=400,
            detail="File has an image extension but is not a valid PNG/JPEG.",
        )

    return ext


# --------------------------------------------------------------------------
# Text extraction
# --------------------------------------------------------------------------


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract embedded (digital) text from a PDF using PyMuPDF."""
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            full_text = "".join(page.get_text() for page in doc)
        return full_text.strip()
    except Exception as exc:  # PyMuPDF raises generic exceptions on corrupt files
        logger.error("Failed to parse PDF with PyMuPDF: %s", exc)
        raise HTTPException(status_code=400, detail="Could not read this PDF file. It may be corrupted.")


def _preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """Light preprocessing to improve OCR accuracy: grayscale + autocontrast."""
    grayscale = ImageOps.grayscale(image)
    return ImageOps.autocontrast(grayscale)


def extract_scanned_pdf_text(pdf_bytes: bytes) -> str:
    """Rasterize a scanned PDF's pages and OCR each one with Tesseract."""
    try:
        images = convert_from_bytes(pdf_bytes, poppler_path=POPPLER_PATH)
    except Exception as exc:
        logger.error("Failed to rasterize PDF for OCR: %s", exc)
        raise HTTPException(
            status_code=400,
            detail="Could not process this PDF for OCR. It may be corrupted or unsupported.",
        )

    text_parts: list[str] = []
    for page_number, image in enumerate(images, start=1):
        try:
            processed = _preprocess_for_ocr(image)
            text_parts.append(pytesseract.image_to_string(processed))
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract binary not found while OCR-ing PDF page %d.", page_number)
            raise HTTPException(
                status_code=500,
                detail="OCR engine is not available on the server.",
            )
        except Exception as exc:
            logger.warning("OCR failed on PDF page %d: %s", page_number, exc)

    return "\n".join(text_parts).strip()


def extract_image_text(image_bytes: bytes) -> str:
    """OCR a standalone image file (PNG/JPEG)."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except UnidentifiedImageError as exc:
        logger.error("Uploaded image could not be identified: %s", exc)
        raise HTTPException(status_code=400, detail="Could not read this image file. It may be corrupted.")

    try:
        processed = _preprocess_for_ocr(image)
        return pytesseract.image_to_string(processed).strip()
    except pytesseract.TesseractNotFoundError:
        logger.error("Tesseract binary not found while OCR-ing image upload.")
        raise HTTPException(status_code=500, detail="OCR engine is not available on the server.")
    except Exception as exc:
        logger.error("OCR failed on image upload: %s", exc)
        raise HTTPException(status_code=400, detail="Could not extract text from this image.")


def extract_text_from_upload(filename: str, ext: str, contents: bytes) -> str:
    """
    Dispatch extraction based on file type. For PDFs, first try digital
    text extraction; if that yields too little text, fall back to OCR
    (i.e. the PDF is likely a scanned document).
    """
    if ext in SUPPORTED_PDF_EXTENSIONS:
        text = extract_pdf_text(contents)
        if len(text) < MIN_TEXT_LENGTH_THRESHOLD:
            logger.info("'%s': digital text too short (%d chars), falling back to OCR.", filename, len(text))
            text = extract_scanned_pdf_text(contents)
        return text

    # Image types
    return extract_image_text(contents)


# --------------------------------------------------------------------------
# Gemini rating
# --------------------------------------------------------------------------


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    """
    Best-effort recovery of a JSON object from a raw LLM response.

    Handles the common failure modes: markdown code fences, leading/trailing
    commentary, or extra text surrounding the JSON payload.
    """
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fall back to grabbing the outermost { ... } block.
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Could not recover valid JSON from model response.", cleaned, 0)


def compute_weighted_score(categories: dict[str, Any]) -> int:
    """
    Compute the overall score as a weighted average of category scores,
    instead of relying on the LLM's own holistic number. Skills and
    experience are weighted highest, format lowest.
    """
    total = 0.0
    for category, weight in CATEGORY_WEIGHTS.items():
        score = categories.get(category, {}).get("score", 5)
        total += score * weight
    return round(total)


def get_cv_rating(cv_text: str, role: str) -> dict[str, Any]:
    """Send the CV text to Gemini and return the parsed, re-scored rating."""
    prompt = f"""Today's date: {date.today().isoformat()}

Target role: {role}

CV text:

{cv_text}
"""

    logger.info("Sending Gemini request (model=%s, role=%s, cv_chars=%d).", GEMINI_MODEL, role, len(cv_text))

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[SYSTEM_PROMPT, prompt],
        )
    except Exception as exc:
        logger.error("Gemini request failed: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to reach the AI rating service.")

    raw_text = (response.text or "").strip()
    if not raw_text:
        logger.error("Gemini returned an empty response.")
        raise HTTPException(status_code=502, detail="AI rating service returned an empty response.")

    try:
        rating = _extract_json_object(raw_text)
    except json.JSONDecodeError:
        logger.error("Gemini returned unparsable JSON. Raw response: %s", raw_text)
        raise HTTPException(status_code=502, detail="AI rating service returned invalid JSON.")

    if "categories" not in rating:
        logger.error("Gemini JSON missing 'categories' key. Raw response: %s", raw_text)
        raise HTTPException(status_code=502, detail="AI rating service returned an unexpected response shape.")

    rating["overall_score"] = compute_weighted_score(rating["categories"])
    logger.info("Gemini rating succeeded (overall_score=%d).", rating["overall_score"])
    return rating


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "HireLens backend is running"}


@app.get("/roles")
def get_roles() -> dict[str, Any]:
    return {"roles": list_role_titles()}


@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)) -> dict[str, Any]:
    contents = await file.read()
    ext = validate_upload(file.filename, contents)

    logger.info("'/extract-text' received '%s' (%d bytes).", file.filename, len(contents))
    text = extract_text_from_upload(file.filename, ext, contents)

    return {"filename": file.filename, "extracted_text": text, "char_count": len(text)}


@app.post("/rate-cv")
async def rate_cv(file: UploadFile = File(...), role: str = Form(...)) -> dict[str, Any]:
    contents = await file.read()
    ext = validate_upload(file.filename, contents)

    logger.info("'/rate-cv' received '%s' (%d bytes) for role '%s'.", file.filename, len(contents), role)
    cv_text = extract_text_from_upload(file.filename, ext, contents)

    if len(cv_text) < MIN_TEXT_LENGTH_THRESHOLD:
        raise HTTPException(status_code=400, detail="Could not extract readable text from this file.")

    role_match = resolve_role(role)
    resolved_role = role_match["matched_role"]

    rating = get_cv_rating(cv_text, resolved_role)
    rating["role_resolution"] = role_match
    return rating
