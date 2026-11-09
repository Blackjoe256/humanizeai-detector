#!/usr/bin/env python3
"""
FastAPI Backend for HumanizeAI.pro Detector Replica & Unlimited AI Humanizer.

Endpoints:
  - GET  /             -> Serves the interactive HumanizeAI Studio UI (index.html)
  - GET  /api/samples  -> Returns built-in calibration text presets
  - POST /api/detect   -> Runs stylometric & multi-signal AI detection
  - POST /api/humanize -> Runs closed-loop unlimited-word AI text humanization
"""

from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from detector import SAMPLES, HumanizeAIDetector
from humanizer import TextHumanizer

app = FastAPI(
    title="HumanizeAI Studio API",
    description="AI Detector Replica & Unlimited AI Text Humanizer Engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = HumanizeAIDetector()
humanizer = TextHumanizer(detector=detector)
BASE_DIR = Path(__file__).resolve().parent


class DetectRequest(BaseModel):
    text: str = Field(..., description="Text of any length to analyze for AI signals")


class HumanizeRequest(BaseModel):
    text: str = Field(..., description="Text of any length to humanize (no word limit)")
    tone: str = Field("standard", description="standard | conversational | academic | concise")
    strength: str = Field("balanced", description="light | balanced | enhanced")
    freeze_words: List[str] = Field(default_factory=list, description="Protected keywords/phrases")
    variation_seed: int = Field(0, description="Optional variation seed for re-humanizing")


@app.get("/")
def serve_ui():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/api/samples")
def get_samples():
    return {"samples": SAMPLES}


@app.post("/api/detect")
def detect_ai(req: DetectRequest):
    report = detector.analyze(req.text)
    return report.to_dict()


@app.post("/api/humanize")
def humanize_text(req: HumanizeRequest):
    result = humanizer.humanize(
        text=req.text,
        tone=req.tone,
        strength=req.strength,
        freeze_words=req.freeze_words,
        variation_seed=req.variation_seed,
    )
    return result.to_dict()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
