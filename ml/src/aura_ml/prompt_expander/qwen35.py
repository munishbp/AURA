"""Qwen3.5-9B prompt expander.

Takes (face_image, user_instruction, procedure) and returns a detailed,
anatomically grounded edit instruction the diffusion editor can act on.

Why this module exists: raw surgeon shorthand ("make the nose smaller") is
too vague for an instruction-tuned image editor to act on consistently, and
an unconstrained VLM rewrite is too *loose* — the original hackathon system
"overshot" modifications (tech report §7.2). So the expander is tightened at
three layers:

1. A strict system prompt: fixed output structure (region → change →
   preservation), a conservative-quantifier ladder, and an out-of-scope
   escape hatch instead of freestyle rewriting.
2. Deterministic post-processing: strip preamble/quotes/thinking, clamp
   length, replace amplifier language ("dramatically" → "subtly"), and
   guarantee an identity-preservation clause is present.
3. A rule-based template fallback so the pipeline still produces a usable
   prompt when the VLM is unavailable or its output fails validation.

Model: Qwen/Qwen3.5-9B (Apache 2.0) — unified early-fusion VLM, loaded 4-bit
(~7 GB) so it co-resides with the NF4 editor on a 32 GB RTX 5090.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from typing import Literal

import torch
from PIL import Image

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"

Procedure = Literal["rhinoplasty", "facelift", "blepharoplasty"]

# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You translate a physician's shorthand into ONE precise edit \
instruction for an image-editing model, using the attached patient photo as \
ground truth for what is anatomically present.

The image editor is literal: it changes exactly what you describe and should \
be told explicitly what to keep. Your output is consumed by that editor, not \
by a person.

Write the instruction in exactly this order, as one flowing paragraph:
1. REGION — name the edit region in plain visual terms anchored to anatomy, \
grounded in what you actually see in the photo (e.g. "the slight bump on the \
bridge of the nose (dorsal hump)").
2. CHANGE — the geometric change, in imperative voice, with a conservative \
magnitude word. Allowed magnitudes: "subtle", "slight", "mild", "moderate", \
"gentle", "refined". Never use "dramatic", "significantly", "completely", \
"much", "very", "drastically", "extremely".
3. PRESERVE — end with what must not change: the person's identity, skin \
texture and tone, eye color, hairstyle, facial expression, head pose, \
lighting, and background.

Hard rules:
- Describe a plausible POST-SURGICAL outcome of the named procedure only. \
Surgical results are conservative; when the physician's wording is aggressive, \
translate it down to a realistic magnitude.
- Only describe changes inside the procedure's region. If the instruction \
asks for anything outside it — a different procedure, changing identity, age, \
ethnicity, expression, style, or an entirely different person — reply with \
exactly: OUT_OF_SCOPE: <ten words or fewer explaining why>
- One paragraph, 40–90 words. No preamble, no quotes, no lists, no headings, \
no explanation. Output the instruction text only.

Example input: "shave the hump down"
Example output: Reduce the slight bony bump on the bridge of the nose (dorsal \
hump) so the nasal profile becomes a smooth, straight line from brow to tip, \
keeping tip projection and nostril width unchanged. Preserve the person's \
identity, skin texture and tone, eye color, hairstyle, facial expression, \
head pose, lighting, and background exactly as in the original photo."""


@dataclass(frozen=True)
class ProcedureSpec:
    """Anatomy vocabulary + scope for one supported procedure."""

    name: str
    context: str          # injected into the user message
    fallback_change: str  # used by the template fallback


PROCEDURES: dict[str, ProcedureSpec] = {
    "rhinoplasty": ProcedureSpec(
        name="rhinoplasty",
        context=(
            "Procedure: rhinoplasty (nose reshaping). Edit region: the nose only — "
            "nasal bridge and dorsal line, dorsal hump, tip projection and rotation, "
            "alar base width, nostril shape, columella."
        ),
        fallback_change=(
            "Refine the nose as instructed, keeping the change subtle and surgically "
            "plausible: smooth the dorsal line and refine the tip"
        ),
    ),
    "facelift": ProcedureSpec(
        name="facelift",
        context=(
            "Procedure: rhytidectomy (facelift). Edit region: lower two thirds of the "
            "face — jawline and mandibular border, jowls, nasolabial folds, marionette "
            "lines, midface/cheek volume, submental area under the chin."
        ),
        fallback_change=(
            "Gently tighten the jawline and soften the jowls and nasolabial folds, "
            "keeping the change subtle and surgically plausible"
        ),
    ),
    "blepharoplasty": ProcedureSpec(
        name="blepharoplasty",
        context=(
            "Procedure: blepharoplasty (eyelid surgery). Edit region: the eyelids and "
            "immediate periorbital area only — upper-lid skin redundancy, supratarsal "
            "crease, lower-lid bags, tear trough. Eye shape, eye color, and canthal "
            "tilt must stay unchanged."
        ),
        fallback_change=(
            "Reduce excess upper-eyelid skin and smooth under-eye bags, keeping the "
            "change subtle and surgically plausible"
        ),
    ),
}

EXAMPLE_INSTRUCTIONS: dict[str, str] = {
    "rhinoplasty": "Subtle dorsal hump reduction with refined nasal tip",
    "facelift": "Tighten the lower-face jawline and reduce nasolabial fold",
    "blepharoplasty": "Reduce upper-lid skin redundancy and refine the supratarsal crease",
}

_PRESERVE_CLAUSE = (
    "Preserve the person's identity, skin texture and tone, eye color, hairstyle, "
    "facial expression, head pose, lighting, and background exactly as in the "
    "original photo."
)

# Amplifiers the system prompt bans; sanitizer enforces with replacement.
# Adverb forms → "subtly", adjective forms → "subtle", "much <comparative>"
# → "subtly <comparative>".
_AMP_ADVERB = re.compile(
    r"\b(dramatically|drastically|significantly|completely|extremely|very)\b",
    re.IGNORECASE,
)
_AMP_ADJECTIVE = re.compile(
    r"\b(dramatic|drastic|significant|complete|extreme)\b", re.IGNORECASE
)
_AMP_MUCH = re.compile(
    r"\bmuch (smaller|larger|bigger|tighter|thinner|wider|narrower)\b", re.IGNORECASE
)


def _tone_down(text: str) -> str:
    text = _AMP_MUCH.sub(r"subtly \1", text)
    text = _AMP_ADVERB.sub("subtly", text)
    return _AMP_ADJECTIVE.sub("subtle", text)

_PREAMBLE = re.compile(
    r"^\s*(here (?:is|'s)[^:]*:|prompt:|instruction:|output:|edit instruction:)\s*",
    re.IGNORECASE,
)


class OutOfScopeError(ValueError):
    """The instruction asked for something outside the selected procedure."""


@dataclass
class ExpanderResult:
    prompt: str
    procedure: str
    out_of_scope: bool = False
    reason: str = ""
    used_fallback: bool = False
    raw_output: str = ""
    latency_s: float = 0.0


@dataclass
class Qwen35Config:
    model_id: str = DEFAULT_MODEL_ID
    quantize_4bit: bool = True
    device: str = "cuda"
    max_new_tokens: int = 220
    # Qwen3.5 non-thinking sampling recipe (model card), nudged colder for
    # output-shape stability.
    temperature: float = 0.6
    top_p: float = 0.8
    top_k: int = 20
    min_words: int = 15
    max_words: int = 110


class Qwen35PromptExpander:
    """Wraps Qwen3.5-9B in 4-bit. Holds the model in memory across calls.

    For training runs, instantiate this in a SEPARATE process so it doesn't
    compete for VRAM with Qwen-Image-Edit. For the Gradio demo, co-locating
    is fine on a 32 GB 5090 (editor NF4 ~17 GB + expander ~7 GB).
    """

    def __init__(self, config: Qwen35Config | None = None) -> None:
        self.config = config or Qwen35Config()
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the model (4-bit NF4) and processor."""
        if self._model is not None:
            return
        from transformers import AutoProcessor, BitsAndBytesConfig

        model_cls = _resolve_model_class()

        print(f"[expander] loading {self.config.model_id} "
              f"(4bit={self.config.quantize_4bit}) ...", file=sys.stderr)
        kwargs: dict = {"dtype": torch.bfloat16}
        if self.config.quantize_4bit and torch.cuda.is_available():
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = self.config.device
        elif torch.cuda.is_available():
            kwargs["device_map"] = self.config.device

        self._processor = AutoProcessor.from_pretrained(self.config.model_id)
        self._model = model_cls.from_pretrained(self.config.model_id, **kwargs).eval()
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 2**30
            print(f"[expander] ready ({used:.1f} GiB total allocated).", file=sys.stderr)

    def unload(self) -> None:
        import gc

        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def expand(
        self,
        face_image: Image.Image,
        user_instruction: str,
        procedure: Procedure | None = None,
        seed: int | None = None,
    ) -> str:
        """Return a detailed diffusion prompt. Raises OutOfScopeError if the
        instruction falls outside the selected procedure."""
        result = self.expand_detailed(face_image, user_instruction, procedure, seed=seed)
        if result.out_of_scope:
            raise OutOfScopeError(result.reason or "instruction outside procedure scope")
        return result.prompt

    def expand_detailed(
        self,
        face_image: Image.Image,
        user_instruction: str,
        procedure: Procedure | None = None,
        seed: int | None = None,
    ) -> ExpanderResult:
        """Full expansion with metadata. Never raises on model issues — falls
        back to the rule-based template instead (the demo must stay usable)."""
        proc = PROCEDURES.get(procedure or "", None)
        proc_name = proc.name if proc else (procedure or "unspecified")
        t0 = time.monotonic()

        try:
            raw = self._generate(face_image, user_instruction, proc, seed=seed)
        except Exception as e:  # model unavailable / OOM / API drift
            print(f"[expander] VLM generation failed ({e}); using template fallback",
                  file=sys.stderr)
            return ExpanderResult(
                prompt=expand_template(user_instruction, procedure),
                procedure=proc_name,
                used_fallback=True,
                reason=f"vlm-error: {e}",
                latency_s=time.monotonic() - t0,
            )

        scope = _parse_out_of_scope(raw)
        if scope is not None:
            return ExpanderResult(
                prompt="",
                procedure=proc_name,
                out_of_scope=True,
                reason=scope,
                raw_output=raw,
                latency_s=time.monotonic() - t0,
            )

        cleaned = _sanitize(raw, self.config)
        if cleaned is None:
            # Output failed validation (too short / empty after cleaning).
            return ExpanderResult(
                prompt=expand_template(user_instruction, procedure),
                procedure=proc_name,
                used_fallback=True,
                reason="vlm output failed validation",
                raw_output=raw,
                latency_s=time.monotonic() - t0,
            )

        return ExpanderResult(
            prompt=cleaned,
            procedure=proc_name,
            raw_output=raw,
            latency_s=time.monotonic() - t0,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _generate(
        self,
        face_image: Image.Image,
        user_instruction: str,
        proc: ProcedureSpec | None,
        seed: int | None = None,
    ) -> str:
        if self._model is None:
            self.load()

        context = proc.context if proc else "Procedure: unspecified facial procedure."
        user_msg = f"{context}\n\nPhysician instruction: {user_instruction.strip()}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": face_image.convert("RGB")},
                    {"type": "text", "text": user_msg},
                ],
            },
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self._model.device)

        if seed is not None:
            torch.manual_seed(seed)

        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = self._processor.decode(new_tokens, skip_special_tokens=True)
        return text.strip()


def _resolve_model_class():
    """Qwen3.5 is a unified multimodal model; the auto-class name has moved
    across transformers 5.x releases, so resolve defensively."""
    import transformers

    for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText",
                 "AutoModelForVision2Seq", "AutoModelForCausalLM"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise ImportError("no suitable AutoModel class found in transformers")


def _parse_out_of_scope(raw: str) -> str | None:
    """Returns the refusal reason if the model declared OUT_OF_SCOPE."""
    m = re.search(r"OUT_OF_SCOPE\s*:?\s*(.*)", raw)
    if m:
        return m.group(1).strip() or "instruction outside procedure scope"
    return None


def _sanitize(raw: str, cfg: Qwen35Config) -> str | None:
    """Deterministic guardrails over the VLM output. Returns None if the
    output is unusable (caller falls back to the template)."""
    text = raw

    # Strip <think> blocks if thinking mode leaked through.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip markdown emphasis and code fences.
    text = re.sub(r"[*_`#]+", "", text)
    # Strip a "Here is the prompt:" style preamble.
    text = _PREAMBLE.sub("", text.strip())
    # Strip wrapping quotes.
    text = text.strip().strip('"“”').strip()
    # Collapse to one paragraph.
    text = re.sub(r"\s+", " ", text).strip()

    # Enforce conservative language: replace banned amplifiers.
    text = _tone_down(text)

    words = text.split()
    if len(words) < cfg.min_words:
        return None
    if len(words) > cfg.max_words:
        # Truncate at the last sentence boundary within budget.
        clipped = " ".join(words[: cfg.max_words])
        cut = clipped.rfind(". ")
        text = clipped[: cut + 1] if cut > 40 else clipped.rstrip(",;: ") + "."

    # Guarantee the preservation clause survives truncation/rewrites.
    if not re.search(r"\b(preserve|maintain|keep|unchanged)\b", text, re.IGNORECASE):
        text = f"{text.rstrip('.')}. {_PRESERVE_CLAUSE}"

    return text


def expand_template(user_instruction: str, procedure: str | None = None) -> str:
    """Rule-based expansion — no VLM. Used as fallback and in --no-expander
    mode. Deterministic, conservative, always valid."""
    proc = PROCEDURES.get(procedure or "")
    instruction = re.sub(r"\s+", " ", user_instruction.strip()).rstrip(".")
    instruction = _tone_down(instruction)
    if proc:
        return (
            f"{proc.context.split('. Edit region')[0]}. "
            f"Apply this change conservatively, as a realistic post-surgical result: "
            f"{instruction}. The change must stay subtle and anatomically plausible. "
            f"{_PRESERVE_CLAUSE}"
        )
    return (
        f"Apply this change conservatively, as a realistic post-surgical result: "
        f"{instruction}. The change must stay subtle and anatomically plausible. "
        f"{_PRESERVE_CLAUSE}"
    )


# ---------------------------------------------------------------------------
# stdio service — run the expander in its own process to isolate VRAM
# ---------------------------------------------------------------------------


def serve_stdio() -> None:
    """Long-lived line protocol: one JSON object in, one JSON object out.

    Request : {"image_b64": <base64 jpeg/png>, "instruction": str,
               "procedure": str|null, "seed": int|null}
    Response: {"prompt": str, "out_of_scope": bool, "reason": str,
               "used_fallback": bool, "latency_s": float}
    A blank line or EOF shuts the service down.
    """
    import base64
    import io

    expander = Qwen35PromptExpander()
    expander.load()
    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            break
        try:
            req = json.loads(line)
            img = Image.open(io.BytesIO(base64.b64decode(req["image_b64"])))
            result = expander.expand_detailed(
                img,
                req["instruction"],
                req.get("procedure"),
                seed=req.get("seed"),
            )
            resp = {
                "prompt": result.prompt,
                "out_of_scope": result.out_of_scope,
                "reason": result.reason,
                "used_fallback": result.used_fallback,
                "latency_s": round(result.latency_s, 2),
            }
        except Exception as e:
            resp = {"prompt": "", "out_of_scope": False, "reason": f"error: {e}",
                    "used_fallback": False, "latency_s": 0.0}
        print(json.dumps(resp), flush=True)


if __name__ == "__main__":
    serve_stdio()
