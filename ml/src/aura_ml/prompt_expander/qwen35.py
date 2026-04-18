"""Qwen3.5-9B prompt expander.

Takes (face_image, user_instruction, procedure_hint) and returns a detailed
anatomically-grounded prompt the diffusion model can act on.

The system prompt biases toward conservative, surgically realistic language —
direct mitigation for the "overshoots modifications" complaint in the original
hackathon retrospective (tech report §7.2).

Workstream 3: implement load + expand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from PIL import Image

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"

Procedure = Literal["rhinoplasty", "facelift", "blepharoplasty"]


SYSTEM_PROMPT = """You are a surgical-visualization prompt assistant. Given a
face photo and a brief instruction from a physician, produce a single detailed
prompt for a diffusion image-editor.

Rules:
- Stay anatomically realistic. Describe changes in terms of anatomical
  landmarks (dorsal hump, alar base, nasolabial fold, supratarsal crease,
  jowl, etc.).
- Use conservative quantifiers. Prefer "subtle", "moderate", "refined".
  Avoid "dramatic", "much smaller", "completely".
- Preserve identity. Always include language asking the editor to maintain the
  subject's bone structure, skin texture, and core proportions.
- Output ONE paragraph, no preamble, no explanation. Just the prompt.
"""


PROCEDURE_HINTS: dict[str, str] = {
    "rhinoplasty": "Procedure context: rhinoplasty. Focus on nasal bridge, "
    "dorsal hump, tip projection, alar base, columella.",
    "facelift": "Procedure context: rhytidectomy. Focus on jawline definition, "
    "midface volume, nasolabial fold, marionette lines, jowl.",
    "blepharoplasty": "Procedure context: blepharoplasty. Focus on upper-lid "
    "skin redundancy, supratarsal crease, lower-lid bags, periorbital hollows.",
}


@dataclass
class Qwen35Config:
    model_id: str = DEFAULT_MODEL_ID
    quantize_4bit: bool = True
    device: str = "cuda"
    max_new_tokens: int = 256


class Qwen35PromptExpander:
    """Wraps Qwen3.5-9B in 4-bit. Holds the model in memory across calls.

    For training runs, instantiate this in a SEPARATE process so it doesn't
    compete for VRAM with Qwen-Image-Edit. For the Gradio demo, co-locating
    is fine on a 32 GB 5090.
    """

    def __init__(self, config: Qwen35Config | None = None) -> None:
        self.config = config or Qwen35Config()
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Lazy-load the model and processor."""
        # TODO(workstream 3):
        #   from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
        #   bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        #                            bnb_4bit_quant_type="nf4")
        #   self._processor = AutoProcessor.from_pretrained(self.config.model_id)
        #   self._model = AutoModelForCausalLM.from_pretrained(
        #       self.config.model_id, quantization_config=bnb, device_map=self.config.device
        #   )
        raise NotImplementedError("workstream 3: implement load")

    def expand(
        self,
        face_image: Image.Image,
        user_instruction: str,
        procedure: Procedure | None = None,
    ) -> str:
        """Return a detailed diffusion prompt."""
        if self._model is None:
            self.load()

        hint = PROCEDURE_HINTS.get(procedure, "") if procedure else ""
        user_msg = (
            f"{hint}\n\nPhysician instruction: {user_instruction}".strip()
        )

        # TODO(workstream 3):
        #   messages = [
        #       {"role": "system", "content": SYSTEM_PROMPT},
        #       {"role": "user", "content": [
        #           {"type": "image", "image": face_image},
        #           {"type": "text", "text": user_msg},
        #       ]},
        #   ]
        #   inputs = self._processor.apply_chat_template(messages, ...)
        #   out = self._model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
        #   text = self._processor.decode(out[0], skip_special_tokens=True)
        #   return text.strip()
        raise NotImplementedError("workstream 3: implement expand")


def serve_stdio() -> None:
    """Run the expander as a long-lived stdio process.

    Protocol: read JSON line from stdin {"image_b64": ..., "instruction": ...,
    "procedure": ...}, write JSON line to stdout {"prompt": ...}.

    Used when the diffuser and expander need to live in separate processes
    (e.g., during training to avoid VRAM contention).
    """
    # TODO(workstream 3): line-protocol service loop
    raise NotImplementedError("workstream 3: implement stdio service")


if __name__ == "__main__":
    serve_stdio()
