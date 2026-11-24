"""Spanish detector fine-tuned by revisor-textos (mmBERT-base, 307M params).

Trained with Claude, Opus 5.5 and Opus 5.5 API held out, it still caught 95 % of
them at a cut that leaves 98 % of 62 SciELO humans from collections it never saw
alone (revisor-textos, calibracion/entrena.py, run v3 of 2026-11-23). Spanish only.

The model is not on the Hub: it loads from SLOPTOTAL_REVISOR_ES (default
~/.local/share/revisor/detector-es-v3) and the engine is registered only when
that folder exists, so a missing model is an absent engine, not a 0 % score.

Softmax saturates: held-out humans and clear AI both reach 0.99999, and the
score is rounded to three decimals downstream. So the score is
sigmoid(logit margin / TEMPERATURE), which keeps the order and spreads the top.
"""

import math
import os
import threading

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.autoconfig import get_device
from app.engines.base import BaseEngine
from app.model_pool import LOAD_LOCK as _load_lock
from app.schemas import EngineResult, score_to_engine_verdict

MODEL_DIR = os.path.expanduser(
    os.getenv("SLOPTOTAL_REVISOR_ES", "~/.local/share/revisor/detector-es-v3")
)
AVAILABLE = os.path.isdir(MODEL_DIR)
TEMPERATURE = 4.0
_MAX_TOKENS = 384
_model = None
_tokenizer = None
_lock = threading.Lock()


def _load_model():
    global _model, _tokenizer
    if _model is None:
        with _load_lock:
            if _model is None:
                _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
                _model = (
                    AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
                    .eval()
                    .to(get_device())
                )
    return _model, _tokenizer


def margin_to_score(margin: float) -> float:
    return 1.0 / (1.0 + math.exp(-margin / TEMPERATURE))


def _margins(texts: list[str]) -> list[float]:
    model, tokenizer = _load_model()
    inputs = tokenizer(
        texts, return_tensors="pt", truncation=True, max_length=_MAX_TOKENS, padding=True
    ).to(next(model.parameters()).device)
    with _lock, torch.inference_mode():
        logits = model(**inputs).logits.float()
    return (logits[:, 1] - logits[:, 0]).tolist()


class ClassifierRevisorEsEngine(BaseEngine):
    @property
    def name(self) -> str:
        return "Revisor ES"

    @property
    def description(self) -> str:
        return "mmBERT-base fine-tuned on Spanish academic prose, Claude and Opus held out"

    @property
    def code(self) -> str:
        return "RES"

    @property
    def engine_type(self) -> str:
        return "embedding"

    def analyze_batch(self, texts: list[str]) -> list[float]:
        return [margin_to_score(m) for m in _margins(texts)]

    def analyze(self, text: str) -> EngineResult:
        try:
            margin = _margins([text])[0]
        except Exception as e:
            return EngineResult(
                engine_name=self.name,
                score=0.0,
                verdict=score_to_engine_verdict(0.0),
                details=f"Model loading failed: {e}",
                description=self.description,
            )
        score = margin_to_score(margin)
        return EngineResult(
            engine_name=self.name,
            score=round(score, 3),
            verdict=score_to_engine_verdict(score),
            details=f"Logit margin {margin:+.2f} (Spanish only; first 384 tokens)",
            description=self.description,
        )
