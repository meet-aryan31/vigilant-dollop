# Audio Enhancement Plan — Fraud Detection Flow

Status: **implemented — verification passed (19/19 checks incl. PDF regression).**

## Goal

Audio uploads now run a fraud-detection analysis instead of a generic summary:

- STT must be more accurate than current local `whisper base` (Hindi + English focus).
- A new, dedicated system prompt decides fraud vs not-fraud.
- If fraud: UI displays **"This recording is a fraudster call"** plus a clear reason with transcript evidence (quotes + timestamps) and model-stated confidence.
- Same pipeline shape as today: file → text → LLM with system prompt → result.
- Video and image pipelines: **untouched**.

## Decisions locked from review answers

| Question | Decision |
|---|---|
| Symptoms | Cold-eyes review, no production pain reported |
| LLM chain | Keep all6 entries as-is; keys will be renewed by owner (groq key is expired today — first3 entries currently fail) |
| Languages | Hindi + English (code-mixed) |
| Fraud flow trigger | **Always** for audio uploads — no mode toggle |
| Evidence | Required: reason must quote transcript with timestamps + confidence |
| STT | Groq `whisper-large-v3-turbo` (cloud, free tier, no local model storage) as primary; local whisper `base` as fallback only |
| Deployment | No storage bomb: torch/whisper must NOT be required for deploy |

## Design

### 1. STT layer (`stt.py`)

Provider abstraction returning a **transcript object**, not a bare string:

```python
{
  "text": str,                  # full transcript
  "segments": [{"start": float, "end": float, "text": str}, ...],
  "language": str,              # detected or pinned
  "provider": "groq" | "local"
}
```

- **Primary:** `litellm.transcription(model="groq/whisper-large-v3-turbo", file=..., response_format="verbose_json", api_key=<groq key from LLM_API_KEYS>, language=<STT_LANGUAGE or None>)`
  - multilingual — handles Hindi, English, and Hinglish code-mixing
  - zero local storage; audio goes to Groq API
  - **rate-limit / failure handling:** on 429, 5xx, timeout, or auth error → fall back to local backend; if fallback unavailable → fail with a clean `STT unavailable` error (never silently return an empty transcript)
  - verify current free-tier limits in Groq docs at implementation time (design already survives them via fallback)
- **Fallback:** existing local `whisper base` (already cached, ~140MB — no new download)
  - move `import whisper` **inside** the fallback function (lazy) so the module imports fine on a deploy without torch
- **Accuracy knobs:**
  - `STT_LANGUAGE` env: default `auto` (code-mixed speech beats forced pinning); can be set to `hi` / `en` per deployment
  - timestamps always kept — required for evidence
  - transcript fed to LLM as `[mm:ss] text` lines so reasons can cite exact moments
- `WHISPER_MODEL` stays for the local fallback only.

### 2. Fraud prompt (`.env`, same pattern as existing prompts)

New var, same style/language (English) as `prompt_img` / `prompt_video` / `prompt_audio`:

```
prompt_fraud=You are a call-verification analyst... (see full text in implementation)
```

Prompt contract:

- Analyze the timestamped transcript of a phone call (Hindi/English/Hinglish).
- Detect social-engineering / fraud patterns: OTP or card-detail demands, impersonation of bank/support/authority, urgency + threats, gift-card/payment-mule requests, tech-support scams, phishing links, verification-code harvesting.
- **Output ONLY valid JSON:** `{"verdict": "fraud" | "legit", "confidence": <0.0-1.0>, "reason": "<2-4 sentences, must quote the transcript verbatim with [mm:ss] timestamps>"}`
- Reason must be grounded in quoted transcript only — never invent speakers, events, or words not in the transcript.
- Acknowledge transcripts may contain STT errors; only quote what is present.

Routing: `get_system_prompt` audio branch resolves `prompt_fraud`, falling back to `prompt_audio` if unset (keeps old behavior for anyone who hasn't set the new var). Video/image branches untouched.

### 3. Deterministic verdict handling (`pipeline.py`)

- Fraud LLM call uses **`FRAUD_TEMPERATURE=0.1`** (new env; global `LLM_TEMPERATURE=0.7` stays for everything else) — a classifier must not be creative.
- `ask_llm` gains optional temperature override (one-arg change, existing callers unaffected).
- Response parsing, in order:
  1. strip code fences, `json.loads`
  2. extract first `{...}` block and parse
  3. validate `verdict ∈ {fraud, legit}` and `reason` non-empty; coerce `confidence` to float
  4. **parse failure → `"inconclusive"` result with raw model text** — never silently report "legit"
- Chain behavior unchanged (all 6 models, first success wins). Provider JSON-mode is *not* relied on since it differs across the chain; the prompt + parser carry the contract.

### 4. Pipeline flow

```
audio upload
  → extension whitelist check (400 on junk — fixes .md → 500 leak)
  → STT (groq primary, local fallback) → {text, segments, language}
  → timestamped transcript string
  → ask_llm(prompt_fraud, temperature=FRAUD_TEMPERATURE)
  → parse JSON → {verdict, confidence, reason, provider, language}
  → JSON response to browser
  → UI renders:
       fraud → red banner: "This recording is a fraudster call" + reason (+ confidence)
       legit → neutral banner: "This recording is not flagged as a fraudster call" + reason
       inconclusive / stt_failed / llm_failed → explicit error state (yellow/red), never a verdict
```

Video/image/PDF/TXT/DOCX paths keep the exact current flow (`run_pipeline` string response for those; only the audio branch returns the structured object).

### 5. Storage / deploy sizing

- `requirements.txt` (core, deploy): flask, litellm, python-dotenv, python-dotenv, python-docx, pdfplumber, Pillow, pytesseract — **no torch, no openai-whisper**
- `requirements-local-stt.txt` (optional): openai-whisper (dev box / self-hosted fallback only)
- Deploy target installs core only → no multi-GB model weights, no torch wheels
- No local model downloads at runtime on the server (Groq is remote); `base.pt` cache stays a dev-machine concern

## Implementation steps

| # | Step | Files | Verify |
|---|---|---|---|
| 1 | Rework `stt.py`: transcript object, Groq primary + lazy local fallback, language env, segment retention | `stt.py` | `python -c` transcribe `test/audio.mp3` → segments + language returned; with bad groq key → clean fallback/error, no crash |
| 2 | Freeze working deps, split requirements | `requirements.txt`, `requirements-local-stt.txt` | fresh `pip install -r requirements.txt` in a scratch venv imports `app` without torch |
| 3 | Add `prompt_fraud`, `FRAUD_TEMPERATURE`, `STT_PROVIDER`, `STT_LANGUAGE` to env files; route audio branch | `.env`, `.env.example`, `pipeline.py` | `get_system_prompt('x.mp3')` returns fraud prompt; image/video prompts unchanged |
| 4 | Fraud call + strict JSON parse + `inconclusive` fallback; temperature override on `ask_llm` | `pipeline.py` | direct `ask_llm` with a fabricated fraud transcript → parses to `verdict=fraud` with quoted reason |
| 5 | `run_pipeline` audio branch returns structured result; whitelist extension check → 400 | `pipeline.py`, `app.py` | `.md` upload → 400 friendly message (was 500 traceback) |
| 6 | UI: verdict banners (fraud/legit/inconclusive/error) + confidence | `templates/index.html` | upload `test/audio.mp3` → rendered verdict, not raw JSON |
| 7 | End-to-end pass on `test/audio.mp3` + one real Hindi/English recording if available | — | full request under Groq STT (needs renewed key) |

**Prerequisite owned by you:** renewed Groq key in `LLM_API_KEYS`. Steps 1–6 are testable before it (fallback/local path + fabricated transcripts); step 7's fast path needs the key.

## Risks / notes

- **Single active provider remains** (chain's first3 Groq entries dead until key renewed) — accepted per your call; STT and LLM then share the same Groq key, so a revoked key hits both at once. Fallback paths exist on both sides (Gemini in chain, local whisper for STT) so the app degrades, not dies.
- **LLM self-reported confidence is poorly calibrated** — presented as-is ("model-stated"), not as a probability.
- **No fraud ground-truth sample in `test/`** — only `audio.mp3` (a clean monologue, likely `legit`). If you have a real fraud call recording, drop it in `test/`; otherwise step 4 verifies detection against a fabricated transcript.
- **Audio leaves the machine** (Groq STT) — accepted in decision #6; local fallback keeps an off-cloud option via `STT_PROVIDER=local`.
