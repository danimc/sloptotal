"""Per-sentence Spanish detector trained by revisor-textos (mmBERT-base + 3-class head).

The paragraph goes in whole and each sentence is classified from the mean of its
tokens, so a sentence is read with its neighbours (revisor-textos,
calibracion/entrena-frases.py). Classes: human, AI, AI-polished. Spanish only.

The model is not on the Hub: it loads from SLOPTOTAL_REVISOR_FRASES (default
~/.local/share/revisor/frases-es) and the route answers 503 when that folder is
missing, so an absent model is never a page of zero scores.

The caller splits the sentences, not this module: the cut and its error rate
were measured on revisor's splitter, and the caller needs the character offsets
of each sentence to draw it.
"""

import json
import os
import threading

import torch
from transformers import AutoModel, AutoTokenizer

from app.autoconfig import get_device
from app.model_pool import LOAD_LOCK as _load_lock

MODEL_DIR = os.path.expanduser(
    os.getenv("SLOPTOTAL_REVISOR_FRASES", "~/.local/share/revisor/frases-es")
)
_MAX_TOKENS = 512
_model = None
_lock = threading.Lock()


def available() -> bool:
    return os.path.exists(os.path.join(MODEL_DIR, "cabeza.pt"))


def _load_model():
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:
                dev = get_device()
                tok = AutoTokenizer.from_pretrained(MODEL_DIR)
                enc = AutoModel.from_pretrained(MODEL_DIR).eval().to(dev)
                head = torch.nn.Linear(enc.config.hidden_size, 3)
                head.load_state_dict(
                    torch.load(os.path.join(MODEL_DIR, "cabeza.pt"), map_location="cpu")
                )
                info = json.load(open(os.path.join(MODEL_DIR, "corte.json")))
                _model = (tok, enc, head.eval().to(dev), info)
    return _model


def join(sentences: list[str]) -> tuple[str, list[tuple[int, int]]]:
    text, spans = "", []
    for s in sentences:
        if text:
            text += " "
        spans.append((len(text), len(text) + len(s)))
        text += s
    return text, spans


def token_sentence(offsets, spans) -> list[int]:
    """Which sentence each token belongs to, or -1 for specials and spaces."""
    out = []
    for a, b in offsets:
        if a == b:
            out.append(-1)
            continue
        out.append(next((i for i, (x, y) in enumerate(spans) if x <= a < y), -1))
    return out


def score(sentences: list[str]) -> list[list[float] | None]:
    """[P(ai), P(polished)] per sentence; None when truncation left it out."""
    tok, enc, head, _ = _load_model()
    text, spans = join(sentences)
    x = tok(text, truncation=True, max_length=_MAX_TOKENS, return_offsets_mapping=True,
            return_tensors="pt")
    ids = torch.tensor(token_sentence(x.pop("offset_mapping")[0].tolist(), spans))
    dev = next(enc.parameters()).device
    with _lock, torch.inference_mode():
        h = enc(**{k: v.to(dev) for k, v in x.items()}).last_hidden_state[0].float()
        out = []
        for i in range(len(sentences)):
            sel = (ids == i).to(dev)
            if not sel.any():
                out.append(None)
                continue
            p = torch.softmax(head(h[sel].mean(0)), -1).tolist()
            out.append([round(p[1], 4), round(p[2], 4)])
    return out


def model_info() -> dict:
    info = _load_model()[3]
    return {"model": "revisor-frases-es", "fold": info.get("fuera"), "max_tokens": _MAX_TOKENS}
