# API Guide

Reference for every API in this project: the HTTP endpoints exposed by the Flask app,
the external services it calls, and the Python functions/CLIs that sit behind them.

Stack: Flask + Gunicorn (Render) · LiteLLM · Groq Whisper STT · local OpenAI Whisper fallback.

---

## 1. HTTP API (Flask — `app.py`)

Base URL locally: `http://127.0.0.1:5000`
Deployed: `https://stt-fraud-api.onrender.com`

| Method | Path          | Auth  | Purpose                              |
| ------ | ------------- | ----- | ------------------------------------ |
| GET    | `/`           | none  | Serve the frontend SPA (`index.html`) |
| GET    | `/health`     | none  | Liveness probe (used by Render)      |
| GET    | `/api/chain`  | none  | Show the configured LLM fallback chain |
| POST   | `/analyze`    | none  | Upload a file and run the pipeline   |

Global behaviour:

- Max upload size: **50 MB** (`MAX_CONTENT_LENGTH`, exceeded requests return `413`).
- CORS: allowed origins from `FRONTEND_ORIGINS` (comma-separated, `*` by default).
- The Flask app also serves `frontend/` as static files at the root URL path.

---

### 1.1 `GET /`

Serves the single-page frontend from the `frontend/` directory.

**Response:** `200` with `text/html` (contents of `frontend/index.html`).

---

### 1.2 `GET /health`

Liveness check for the Render service (`healthCheckPath: /health` in `render.yaml`).

**Response:**

```json
{ "status": "ok" }
```

---

### 1.3 `GET /api/chain`

Returns the ordered LLM fallback chain parsed from the `LLM_CHAIN` env var. The frontend
renders it as `chain: model1 → model2 → ...` on page load.

| Case                    | Response                          |
| ----------------------- | --------------------------------- |
| Chain configured        | `{"chain": ["openai/gpt-4o-mini", ...]}` |
| Chain empty/invalid     | `{"chain": []}`                   |

```bash
curl http://127.0.0.1:5000/api/chain
```

---

### 1.4 `POST /analyze`

The main endpoint. Accepts a single file upload and runs the whole pipeline.

**Request**

| Field  | Type            | Required | Notes                                                    |
| ------ | --------------- | -------- | -------------------------------------------------------- |
| `file` | multipart/form  | yes      | Audio, video, image, PDF, TXT or DOCX. 50 MB max.        |

**Accepted extensions**

- Audio: `.mp3 .wav .m4a .flac .ogg .webm .aac .wma`
- Video: `.mp4 .mov .avi .mkv .m4v .mpg .mpeg .wmv .flv`
- Image: `.png .jpg .jpeg .bmp .tif .tiff .webp`
- Documents: `.pdf .txt .docx`

**What happens server-side**

1. Extension is validated against the allowed set.
2. File is written to a temp path; the temp file is always deleted afterwards (`finally`).
3. `run_pipeline(path)` branches on the extension:
   - **Audio** → transcribe → timestamped transcript → fraud prompt → JSON verdict.
   - **Video** → `ffmpeg` extracts a 16 kHz mono WAV → local Whisper → LLM answer.
   - **Image / PDF / TXT / DOCX** → text extraction (OCR / parse) → LLM answer.

**Response shapes**

The envelope depends on the branch. Audio always returns `result`; other types return `response`.

```json
// Audio — fraud verdict
{
  "result": {
    "status": "fraud",
    "confidence": 0.87,
    "reason": "Caller asks for the OTP at [00:12] and claims to be from the bank...",
    "language": "hi",
    "stt_provider": "groq/whisper-large-v3-turbo"
  },
  "filename": "call.mp3"
}
```

`result.status` values:

| Status          | Meaning                                                      |
| --------------- | ------------------------------------------------------------ |
| `fraud`         | Model judged the caller a fraudster                          |
| `legit`         | Model judged the call not a fraudster                        |
| `inconclusive`  | Model reply was not valid verdict JSON (`raw` holds the text) |
| `empty`         | No speech detected in the recording                          |
| `error`         | STT or LLM failed (`reason` holds the message)               |

```json
// Image / video / PDF / TXT / DOCX — LLM answer
{
  "response": "The document is an invoice for ...",
  "filename": "invoice.pdf"
}
```

```json
// No text could be extracted
{ "response": null, "filename": "blank.png" }
```

**Errors**

| Code | Body                                                       | Cause                                   |
| ---- | ---------------------------------------------------------- | --------------------------------------- |
| 400  | `{"error": "No file uploaded"}`                             | `file` field missing                     |
| 400  | `{"error": "Unsupported file type '.exe'. Allowed: ..."}`    | Extension not in the allow-list         |
| 500  | `{"error": "<exception message>"}`                          | Pipeline/LLM failure (stack trace logged) |
| 413  | Werkzeug error page                                         | Upload above 50 MB                      |

**Example**

```bash
curl -X POST http://127.0.0.1:5000/analyze -F "file=@test/audio.mp3"
curl -X POST http://127.0.0.1:5000/analyze -F "file=@test/docs.pdf"
```

---

## 2. External service APIs

| Service                       | Used for                              | Where                              | Key                            |
| ----------------------------- | ------------------------------------- | ---------------------------------- | ------------------------------ |
| Groq STT (whisper-large-v3-turbo) | Speech-to-text with segments/language | `stt._transcribe_groq`             | `GROQ_API_KEY` or `LLM_API_KEYS` |
| LiteLLM `completion()`        | Fraud verdict / summarisation         | `pipeline.ask_llm`                 | `LLM_API_KEYS` (per provider)  |
| LiteLLM `transcription()`     | Same Groq call, provider-agnostic     | `stt._transcribe_groq`             | same                           |
| OpenAI Whisper (local)        | Offline STT fallback, video STT       | `stt._transcribe_local`, `_video_to_text` | none (local weights)   |
| Tesseract OCR                 | Text from images                       | `text_extract._extract_image`      | none (system binary)           |
| pdfplumber                    | Text from PDFs                         | `text_extract._extract_pdf`        | —                              |
| python-docx                   | Text from `.docx`                      | `text_extract._extract_docx`       | —                              |
| ffmpeg (CLI subprocess)       | Video → 16 kHz mono WAV                | `pipeline._video_to_text`          | —                              |

**LLM fallback chain.** `ask_llm` walks `LLM_CHAIN` left to right, one attempt per model.
The first success wins; if every model fails a `RuntimeError` listing all errors is raised.
API keys are matched by the provider prefix of each entry (`groq/...` → `groq=...`).

**Verdict parsing.** `_parse_verdict` strips markdown fences, extracts the outermost `{...}`
block, and requires `verdict` ∈ {`fraud`, `legit`} plus a non-empty `reason`.
`confidence` is clamped to `0.0–1.0` or set to `null` if unparsable.

---

## 3. Python API

### 3.1 `pipeline.py`

| Function | Signature | What it does |
| -------- | --------- | ------------ |
| `run_pipeline` | `(file_path: str) -> str \| dict \| None` | Entry point used by `/analyze`. Audio → fraud verdict dict; other files → LLM answer string, or `None` when no text was found. |
| `run_fraud_check` | `(file_path: str) -> dict` | Audio flow: `transcribe()` → timestamped lines → LLM with the fraud prompt → parsed verdict. Never raises; failures come back as `{"status": "error", ...}`. |
| `ask_llm` | `(user_text, system_prompt=None, temperature=None) -> str` | Sends one chat completion down the chain, returning the first non-empty content. Raises `RuntimeError` if all models fail. |
| `get_input_text` | `(file_path: str) -> str \| None` | Audio/video/document → text (Whisper, ffmpeg+Whisper, or `extract_text`). |
| `get_system_prompt` | `(file_path: str) -> str` | Picks the prompt by extension: audio → `prompt_fraud` → `prompt_audio`, image → `prompt_img`, video → `prompt_video`, else `LLM_SYSTEM_PROMPT`. Falls back to `You are a helpful assistant.` |
| `_parse_chain` | `(raw: str) -> list[str]` | Splits `LLM_CHAIN` on commas; raises `ValueError` on empty or entries without `/`. |
| `_parse_api_keys` | `(raw: str) -> dict[str, str]` | Splits `LLM_API_KEYS` into `provider=key` pairs; raises `ValueError` on bad entries. |
| `_config` | `() -> dict` | Validates that every chain provider has a key; returns chain, keys, prompt, temperature, max_tokens. |
| `_format_segments` | `(segments: list[dict]) -> str` | Renders segments as `[mm:ss] text` lines. |
| `_parse_verdict` | `(raw: str) -> dict` | Tolerant JSON verdict parser (fences, embedded braces, clamping). |

### 3.2 `stt.py`

| Function | Signature | What it does |
| -------- | --------- | ------------ |
| `transcribe` | `(audio_path: str) -> dict` | Provider-agnostic STT. `STT_PROVIDER` = `auto` (Groq → local), `groq` (fail loudly), or `local`. Returns `{text, segments, language, provider}`. |
| `audio_to_text` | `(audio_path: str, model_name="base") -> str` | Plain local Whisper transcription → text string. Used for the video path and the CLI. |
| `load_model` | `(model_name="base")` | Loads and caches a local Whisper model in `_model_cache`. |
| `_transcribe_groq` | `(audio_path) -> dict` | Groq `whisper-large-v3-turbo` with `response_format=verbose_json`; honours `STT_LANGUAGE` (`auto` = let the model detect). |
| `_transcribe_local` | `(audio_path) -> dict` | Local Whisper using `WHISPER_MODEL` (tiny…large). |
| `_clean_segments` | `(raw) -> list[dict]` | Normalises segment objects/dicts into `{start, end, text}`, dropping empties. |
| `_provider_key` | `(provider: str) -> str \| None` | Looks a key up in `LLM_API_KEYS`. |

### 3.3 `text_extract.py`

| Function | Signature | What it does |
| -------- | --------- | ------------ |
| `extract_text` | `(file_path: str) -> str \| None` | Dispatches by extension; returns stripped text or `None` when empty. Raises `FileNotFoundError` / `ValueError` for bad paths and unsupported types. |
| `_extract_image` | `(path: Path) -> str` | Tesseract OCR on the image. |
| `_extract_pdf` | `(path: Path) -> str` | Concatenates page text across all pages. |
| `_extract_txt` | `(path: Path) -> str` | UTF-8 read, errors ignored. |
| `_extract_docx` | `(path: Path) -> str` | Paragraphs plus every table cell. |

On Windows, if Tesseract is not on `PATH`, the module auto-points at
`C:\Program Files\Tesseract-OCR\tesseract.exe`.

### 3.4 Command-line APIs

| Command | Module | Purpose |
| ------- | ------ | ------- |
| `python run.py <file>` | `run.py` | Full pipeline on a local path; prints the LLM answer or the verdict dict, `null` if nothing extracted. |
| `python transcribe.py <audio> [-m tiny\|base\|small\|medium\|large]` | `transcribe.py` | Local Whisper transcription → stdout. |
| `python extract.py <file>` | `extract.py` | Text extraction only (image/PDF/TXT/DOCX) → stdout. |
| `python app.py` | `app.py` | Dev server on `HOST`/`PORT` (default `127.0.0.1:5000`). |
| `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120` | — | Production server used by Render. |

---

## 4. Configuration reference

| Variable | Default | Used by |
| -------- | ------- | ------- |
| `LLM_CHAIN` | – | Ordered `provider/model` fallback list (`ask_llm`, `GET /api/chain`) |
| `LLM_API_KEYS` | – | `provider=key` pairs, comma-separated |
| `LLM_SYSTEM_PROMPT` | `You are a helpful assistant.` | Default system prompt (documents) |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature for non-fraud calls |
| `LLM_MAX_TOKENS` | `2048` | Max response length |
| `prompt_fraud` | – | Audio fraud-detection prompt (takes priority over `prompt_audio`) |
| `prompt_audio` | – | Audio summary prompt |
| `prompt_img` | – | Image prompt |
| `prompt_video` | – | Video prompt |
| `FRAUD_TEMPERATURE` | `0.1` | Temperature for the verdict call |
| `STT_PROVIDER` | `auto` | `auto` / `groq` / `local` |
| `STT_LANGUAGE` | `auto` | `auto` / `en` / `hi` |
| `GROQ_API_KEY` | – | Groq STT key (falls back to `LLM_API_KEYS`) |
| `WHISPER_MODEL` | `base` | Local Whisper size (tiny…large) |
| `FRONTEND_ORIGINS` | `*` | CORS allow-list (comma-separated) |
| `FLASK_DEBUG` | `0` | Set `1` for local reloading |
| `HOST` / `PORT` | `127.0.0.1` / `5000` | Dev server bind |

---

## 5. Flow summary

```
POST /analyze
      │
      ├─ audio ─► STT (Groq → local Whisper) ─► "[mm:ss] transcript" ─► fraud prompt ─► JSON verdict
      │                                                                    │
      │                                                            LLM_CHAIN fallback
      ├─ video ─► ffmpeg → WAV ─► local Whisper ─► text ──────────────────────┤
      └─ img/pdf/txt/docx ─► Tesseract / pdfplumber / read / python-docx ─────┘
                                                                              ▼
                                                              LLM answer (or None if empty)
```
