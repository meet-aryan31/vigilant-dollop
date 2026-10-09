import os
import tempfile
import traceback
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from pipeline import AUDIO_EXTS, VIDEO_EXTS, _parse_chain, run_pipeline
from text_extract import DOCX_EXTS, IMAGE_EXTS, PDF_EXTS, TXT_EXTS

load_dotenv()

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

CORS_ORIGINS = [o.strip() for o in os.getenv("FRONTEND_ORIGINS", "*").split(",") if o.strip()]
CORS(app, origins=CORS_ORIGINS)

ALLOWED_EXTS = AUDIO_EXTS | VIDEO_EXTS | IMAGE_EXTS | PDF_EXTS | TXT_EXTS | DOCX_EXTS


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/chain")
def api_chain():
    try:
        return jsonify({"chain": _parse_chain(os.getenv("LLM_CHAIN", ""))})
    except ValueError:
        return jsonify({"chain": []})


@app.post("/analyze")
def analyze():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No file uploaded"}), 400

    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in ALLOWED_EXTS:
        allowed = ", ".join(sorted(ALLOWED_EXTS))
        return jsonify({"error": f"Unsupported file type '{suffix}'. Allowed: {allowed}"}), 400
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        file.save(tmp.name)
        tmp.close()
        response = run_pipeline(tmp.name)
        if isinstance(response, dict):
            return jsonify({"result": response, "filename": file.filename})
        if response is None:
            return jsonify({"response": None, "filename": file.filename})
        return jsonify({"response": response, "filename": file.filename})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
