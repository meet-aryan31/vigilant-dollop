import os
from typing import Optional

from litellm import transcription

_model_cache = {}


def _provider_key(provider: str) -> Optional[str]:
    for part in os.getenv("LLM_API_KEYS", "").split(","):
        part = part.strip()
        if "=" in part:
            name, key = part.split("=", 1)
            if name.strip() == provider:
                return key.strip()
    return None


def load_model(model_name: str = "base"):
    if model_name not in _model_cache:
        import whisper

        _model_cache[model_name] = whisper.load_model(model_name)
    return _model_cache[model_name]


def _seg_value(seg, key):
    if isinstance(seg, dict):
        return seg.get(key)
    return getattr(seg, key, None)


def _clean_segments(raw) -> list[dict]:
    segments = []
    for seg in raw or []:
        text = _seg_value(seg, "text")
        if not text:
            continue
        try:
            start = float(_seg_value(seg, "start") or 0.0)
            end = float(_seg_value(seg, "end") or 0.0)
        except (TypeError, ValueError):
            start = end = 0.0
        segments.append({"start": start, "end": end, "text": str(text).strip()})
    return [s for s in segments if s["text"]]


def _transcribe_groq(audio_path: str) -> dict:
    language = (os.getenv("STT_LANGUAGE") or "auto").strip().lower()
    api_key = os.getenv("GROQ_API_KEY") or _provider_key("groq")
    if not api_key:
        raise RuntimeError("no Groq API key found (set GROQ_API_KEY or LLM_API_KEYS)")

    kwargs = {
        "model": "groq/whisper-large-v3-turbo",
        "response_format": "verbose_json",
        "api_key": api_key,
    }
    if language not in ("", "auto"):
        kwargs["language"] = language

    with open(audio_path, "rb") as handle:
        response = transcription(file=handle, **kwargs)

    text = (getattr(response, "text", None) or "").strip()
    segments = _clean_segments(getattr(response, "segments", None))
    if not segments and text:
        segments = [{"start": 0.0, "end": 0.0, "text": text}]
    detected = getattr(response, "language", None)
    return {
        "text": text,
        "segments": segments,
        "language": detected or (None if language == "auto" else language),
        "provider": "groq/whisper-large-v3-turbo",
    }


def _transcribe_local(audio_path: str) -> dict:
    model_name = os.getenv("WHISPER_MODEL", "base")
    model = load_model(model_name)
    result = model.transcribe(audio_path)
    text = (result.get("text") or "").strip()
    segments = [
        {"start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)), "text": str(s.get("text", "")).strip()}
        for s in result.get("segments", [])
    ]
    segments = [s for s in segments if s["text"]]
    return {
        "text": text,
        "segments": segments,
        "language": result.get("language"),
        "provider": f"local/whisper-{model_name}",
    }


def transcribe(audio_path: str) -> dict:
    """Transcribe audio to {text, segments, language, provider}.

    STT_PROVIDER: auto (groq then local), groq, or local.
    """
    provider = (os.getenv("STT_PROVIDER") or "auto").strip().lower()
    if provider == "local":
        return _transcribe_local(audio_path)
    try:
        return _transcribe_groq(audio_path)
    except Exception as groq_error:
        if provider == "groq":
            raise RuntimeError(f"Groq STT failed: {groq_error}") from groq_error
        try:
            return _transcribe_local(audio_path)
        except Exception as local_error:
            raise RuntimeError(
                f"STT failed - groq: {groq_error}; local: {local_error}"
            ) from groq_error


def audio_to_text(audio_path: str, model_name: str = "base") -> str:
    """Convert an audio file to text using local OpenAI Whisper.

    Args:
        audio_path: Path to the audio file (mp3, wav, m4a, etc.).
        model_name: Whisper model size (tiny, base, small, medium, large).

    Returns:
        Transcribed text as a string.
    """
    model = load_model(model_name)
    result = model.transcribe(audio_path)
    return result["text"].strip()
