 HireLens

AI-powered CV rating system built specifically for the Pakistani job market. HireLens evaluates a candidate's CV the way a Pakistani corporate recruiter would — accounting for local university tiers, CGPA conventions, mixed English/Urdu formatting norms, and the volume-driven screening reality of platforms like Rozee.pk — rather than applying generic, Western-centric resume-scoring logic.

**Live app:** [hire-lens-lime.vercel.app](https://hire-lens-lime.vercel.app)
**Backend API:** [aadilkk18-hirelens-backend.hf.space](https://aadilkk18-hirelens-backend.hf.space)

---

## Features

- **Multi-format CV ingestion** — accepts PDF, PNG, and JPG, including scanned or photographed CVs
- **Hybrid text extraction** — direct text extraction via PyMuPDF for digital PDFs, with automatic OCR fallback (Tesseract) for scanned documents and images
- **Semantic role matching** — free-text job role input is matched against 115+ predefined roles using sentence-transformer embeddings, with a confidence threshold to avoid false matches
- **Six-category structured evaluation** — Format, University, Experience, Skills, Clarity, and Red Flags, each scored 1–10 with a specific written comment
- **Deterministic weighted scoring** — the final score is a weighted average (Skills 28%, Experience 28%, University 14%, Red Flags 14%, Clarity 10%, Format 6%), not just the LLM's own guess
- **Date-aware AI rating** — the model is given the actual current date so it doesn't misflag recent or same-month experience as fabricated
- **Actionable feedback** — every rating includes a prioritized list of concrete fixes

---

## Tech Stack

**Frontend**
- React (Vite)
- Custom CSS with a design-token system
- Hosted on Vercel

**Backend**
- FastAPI (Python 3.13) on Uvicorn
- Google Gemini API (`gemini-3.1-flash-lite`) for CV rating
- sentence-transformers (`all-MiniLM-L6-v2`) for semantic role matching
- PyMuPDF for digital PDF text extraction
- Tesseract OCR + pdf2image (Poppler) for scanned/image CVs
- Dockerized, hosted on Hugging Face Spaces

---

## Architecture
User
|
v
React Frontend (Vercel)
| -- multipart/form-data (CV file + role) -->
v
FastAPI Backend (Hugging Face Spaces, Docker)
|
|--> Upload validation (size cap, extension + magic-byte check)
|--> Text extraction (PyMuPDF, with Tesseract OCR fallback + preprocessing)
|--> Role resolution (sentence-transformer embedding similarity)
|--> LLM rating call (Gemini API, Pakistani-recruiter system prompt, date-aware)
|--> Weighted score computation (deterministic, category-weighted)
v
Structured JSON response --> rendered in Results Dashboard

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/roles` | GET | Returns the full list of predefined role titles |
| `/extract-text` | POST | Accepts a CV file, returns raw extracted text |
| `/rate-cv` | POST | Accepts a CV file + target role, returns the full structured rating |

---

## Getting Started

### Prerequisites

- Python 3.13
- Node.js (for the frontend)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and on your PATH
- [Poppler](https://poppler.freedesktop.org/) installed (for scanned-PDF handling)
- A [Google Gemini API key](https://ai.google.dev/)

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

Create a `.env` file in `backend/`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
CORS_ALLOW_ORIGINS=http://localhost:5173
```

Run the server:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

> Tesseract and Poppler paths are auto-detected via system PATH. If they're not found automatically, set `TESSERACT_CMD` and `POPPLER_PATH` in your `.env` to override.

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:5173`.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | API key for Google Gemini |
| `GEMINI_MODEL` | No | Overrides the default model (`gemini-3.1-flash-lite`) |
| `CORS_ALLOW_ORIGINS` | No | Comma-separated allowed origins (defaults to `http://localhost:5173`) |
| `TESSERACT_CMD` | No | Override path to the Tesseract binary |
| `POPPLER_PATH` | No | Override path to the Poppler `bin` directory |

---

## Deployment

- **Frontend:** auto-deployed on push via Vercel
- **Backend:** Dockerized and deployed on Hugging Face Spaces (free CPU tier, no credit card required)

Backend deployment workflow:
1. Modify code in `backend/`
2. Copy updated files into the `hirelens_backend` deployment repo
3. Commit and push to the Hugging Face Space's Git remote
4. Hugging Face automatically rebuilds the container

---

## Project Structure
HireLens/
├── frontend/          # React application
├── backend/           # FastAPI application
│   ├── main.py
│   ├── role_resolver.py
│   ├── requirements.txt
│   └── Dockerfile
└── .gitignore

---

## Roadmap

- [ ] Rules-based pre-screening layer alongside the LLM evaluation
- [ ] Batch CV processing and ranking dashboard
- [ ] Downloadable PDF report of results
- [ ] User accounts and rating history
- [ ] Validation study against real Pakistani HR practitioners
- [ ] Mobile responsiveness pass

---

## Author

**Aadil Kamal Khan**
[GitHub](https://github.com/aadilkk18) · [LinkedIn](https://linkedin.com/in/aadil-kamal-khan-ai)
