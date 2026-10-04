"""
app.py
======
FastAPI Server for Offline ML Image-to-Text Reader.
Zero external cloud AI API calls. 100% local execution.
"""

import time
import io
import base64
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from ocr_engine import MLImageReader
from text_formatter import ProperTextFormatter

app = FastAPI(
    title="Offline ML Image-to-Text Reader",
    description="Reads text from images and reconstructs proper text using local Machine Learning models without cloud AI.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/samples", StaticFiles(directory="samples"), name="samples")

# Initialize local ML engine
reader_engine = MLImageReader()


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        content = f.read()
    response = HTMLResponse(content=content)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.post("/api/read-image")
async def api_read_image(
    file: UploadFile = File(...),
    spell_check: bool = Form(False),
    enable_preprocessing: bool = Form(True),
    psm: int = Form(3)
):
    """
    Receives an image file, runs local ML preprocessing & neural OCR,
    and returns formatted proper text along with stats.
    """
    try:
        start_time = time.time()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Run offline ML inference
        result = reader_engine.read_image(
            image_input=image,
            enable_preprocessing=enable_preprocessing,
            psm=psm,
            spell_check=spell_check
        )

        inference_time_ms = round((time.time() - start_time) * 1000, 1)

        # Convert preprocessed image to base64 preview
        prep_img = result.get("preprocessed_image")
        prep_base64 = None
        if prep_img:
            buf = io.BytesIO()
            prep_img.save(buf, format="PNG")
            prep_base64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        return JSONResponse({
            "status": "success",
            "proper_text": result["proper_text"],
            "raw_text": result["raw_text"],
            "confidence": result["confidence"],
            "word_count": result["word_count"],
            "character_count": result["character_count"],
            "skew_angle": result["skew_angle"],
            "variant": result.get("variant", "optimized"),
            "inference_time_ms": inference_time_ms,
            "preprocessed_image": prep_base64
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "engine": "Local Tesseract LSTM Neural Network",
        "cloud_ai_required": False
    }


if __name__ == "__main__":
    import socket
    import sys
    import uvicorn

    def is_port_in_use(p: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", p)) == 0

    target_port = 8001
    if is_port_in_use(target_port):
        # Find next available port
        for p in range(8002, 8020):
            if not is_port_in_use(p):
                target_port = p
                break

    print(f"\n========================================================")
    print(f"🚀 TextFlow DeFi Terminal is live!")
    print(f"👉 Open in browser: http://127.0.0.1:{target_port}")
    print(f"========================================================\n")

    uvicorn.run("app:app", host="127.0.0.1", port=target_port, reload=True)
