"""Unified Qwen3-VL client for both closed-set species verification and RAG scene description."""

from __future__ import annotations

import contextlib
import gc
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
from PIL import Image

from app.src.config import QwenVLMConfig
from app.src.schemas import SceneDescriptionResult, VLMClassificationResult
from app.src.utils.geometry import norm_species, prepare_full_frame_pil
from app.src.utils.logging_utils import get_logger

logger = get_logger("QwenWildlifeVLM")

# Closed-set prompt used with GRPO LoRA for verifying YOLO bounding boxes
CLOSED_SET_CLASSIFICATION_PROMPT = (
    "You are an expert Arabian wildlife camera-trap analyst. "
    "You will review the result of Yolo as supervision to check if its result is True or False. "
    "Inspect ONLY the single target inside the red rectangle. "
    "Classify it strictly into one of three classes: "
    "'arabian_oryx' (bright white body, long straight slender horns, dark facial markings), "
    "'arabian_gazelle' (light sandy-tan body, white belly, short curved/lyre horns), "
    "or 'none' (ANY other animal species, rock, bush, shadow, vehicle, or person). "
    "Output ONLY a valid JSON object inside <answer>...</answer> tags: "
    '{"species": "arabian_oryx" | "arabian_gazelle" | "none"}.'
)

# Open-ended structured prompt used for rich RAG wildlife scene descriptions
SCENE_DESCRIPTION_SYSTEM_PROMPT = (
    "You are an expert Arabian wildlife ecologist and camera-trap scene analyst. "
    "Analyze the entire wildlife image carefully to produce a rich, accurate description "
    "for a Retrieval-Augmented Generation (RAG) wildlife monitoring database. "
    "Describe: "
    "(1) exact number of Arabian Oryx (المها العربي) and Arabian Gazelles (الغزال العربي) visible, "
    "(2) their physical state, posture, group behavior, and activity (e.g., grazing, walking, resting, alert, running, drinking, nursing), "
    "(3) the surrounding habitat and terrain (e.g., sand dunes, gravel plain, desert shrubland, rocky hills, oasis/waterhole), "
    "(4) notable background elements (trees, fences, hills, sky, vehicles, shadows), "
    "(5) lighting conditions and estimated time of day (dawn, morning, midday, afternoon, dusk, night, unknown), "
    "and (6) visual clarity/visibility quality. "
    "Output ONLY a valid JSON object inside <description>...</description> tags with this exact schema:\n"
    "{\n"
    '  "description_ar": "وصف طبيعي وغني باللغة العربية للمشهد يشمل عدد المها والغزلان وحالتهم وسلوكهم والبيئة والخلفية والوقت المتوقع من اليوم",\n'
    '  "description_en": "Concise, rich English scene summary covering species counts, behavior, habitat, background, and estimated time of day",\n'
    '  "estimated_time_of_day": "dawn" | "morning" | "midday" | "afternoon" | "dusk" | "night" | "unknown",\n'
    '  "environment_type": "sand_dunes" | "gravel_plain" | "desert_shrubland" | "rocky_hills" | "waterhole" | "savanna_scrub" | "unknown",\n'
    '  "animal_states": ["grazing", "walking", "resting", "alert", "running", "drinking", "standing"],\n'
    '  "background_elements": ["short snake_case tags describing background objects"],\n'
    '  "visibility_quality": "clear" | "partial_occlusion" | "distant" | "blurry" | "low_light",\n'
    '  "observed_counts": {"arabian_oryx": 0, "arabian_gazelle": 0, "other": 0}\n'
    "}"
)

VALID_TIMES_OF_DAY = {"dawn", "morning", "midday", "afternoon", "dusk", "night", "unknown"}
VALID_VISIBILITY = {"clear", "partial_occlusion", "distant", "blurry", "low_light"}


def parse_classification_response(raw_text: str) -> VLMClassificationResult:
    """Parse <answer>{"species": "..."}</answer> safely from Qwen output."""
    match = re.search(r"<answer>\s*(\{.*?\})\s*</answer>", raw_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if "species" in data:
                return VLMClassificationResult(
                    species=norm_species(data["species"]),
                    raw_text=raw_text,
                    parsed_ok=True,
                )
        except json.JSONDecodeError:
            pass
    return VLMClassificationResult(species="unknown", raw_text=raw_text, parsed_ok=False)


def _extract_json_block(raw_text: str, tag: str = "description") -> Optional[Dict[str, Any]]:
    """Extract JSON from <tag>...</tag>, fenced code blocks, or the outermost JSON object."""
    patterns = [
        rf"<{tag}>\s*(\{{.*?\}})\s*</{tag}>",
        r"```(?:json)?\s*(\{.*?\})\s*```",
        r"(\{.*\})",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_text, re.DOTALL)
        if not match:
            continue
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def parse_scene_description_response(raw_text: str) -> SceneDescriptionResult:
    """Parse structured JSON scene description from Qwen output with graceful fallback."""
    data = _extract_json_block(raw_text, tag="description")
    if data is None:
        cleaned = raw_text.strip()
        return SceneDescriptionResult(
            description_ar=cleaned,
            description_en="",
            estimated_time_of_day="unknown",
            environment_type="unknown",
            animal_states=[],
            background_elements=[],
            visibility_quality="clear",
            observed_counts={},
            raw_text=raw_text,
            parsed_ok=False,
        )

    tod = norm_species(data.get("estimated_time_of_day", "unknown"))
    if tod not in VALID_TIMES_OF_DAY:
        tod = "unknown"

    vis = norm_species(data.get("visibility_quality", "clear"))
    if vis not in VALID_VISIBILITY:
        vis = "clear"

    raw_counts = data.get("observed_counts") or {}
    observed_counts: Dict[str, int] = {}
    if isinstance(raw_counts, dict):
        for k, v in raw_counts.items():
            try:
                observed_counts[norm_species(k)] = max(0, int(v))
            except (TypeError, ValueError):
                continue

    states = [norm_species(s) for s in (data.get("animal_states") or []) if str(s).strip()]
    bg_elems = [norm_species(b) for b in (data.get("background_elements") or []) if str(b).strip()]

    return SceneDescriptionResult(
        description_ar=str(data.get("description_ar", "")).strip(),
        description_en=str(data.get("description_en", "")).strip(),
        estimated_time_of_day=tod,
        environment_type=norm_species(data.get("environment_type", "unknown")) or "unknown",
        animal_states=states,
        background_elements=bg_elems,
        visibility_quality=vis,
        observed_counts=observed_counts,
        raw_text=raw_text,
        parsed_ok=True,
    )


def _format_scene_user_prompt(
    context: Optional[Dict[str, Any]] = None,
    num_frames: int = 1,
) -> str:
    """Build the user prompt for 1-to-3 chronological frame scene description."""
    frame_phrase = (
        f"These {num_frames} chronological frames show the same wildlife tracking event across time. "
        if num_frames > 1
        else "This wildlife camera-trap frame shows a wildlife tracking event. "
    )
    if not context:
        return (
            f"{frame_phrase}"
            "Analyze the animal counts, movement/behavior across frames (grazing, walking, running, resting, alert), "
            "habitat/terrain, background, lighting, and estimated time of day. "
            "Output ONLY the required <description>{...}</description> JSON."
        )

    video_source = context.get("video_source", "unknown")
    best_frame = context.get("best_frame", "N/A")
    temporal_indices = context.get("temporal_frame_indices") or []
    start_hms = context.get("start_hms", "00:00:00")
    end_hms = context.get("end_hms", "00:00:00")
    best_hms = context.get("best_hms", start_hms)
    counts = context.get("species_counts") or {}
    oryx_n = counts.get("arabian_oryx", 0)
    gazelle_n = counts.get("arabian_gazelle", 0)
    avg_conf = context.get("avg_conf")
    conf_str = f"{avg_conf:.2f}" if isinstance(avg_conf, (int, float)) else "N/A"
    primary_sp = context.get("species", "wildlife")

    return (
        f"{frame_phrase}Primary tracked species: {primary_sp}.\n"
        "Use all chronological frames to accurately determine movement/behavior (grazing, walking, running, resting, alert), "
        "exact animal counts, terrain/habitat, background elements, and lighting/time of day.\n"
        f"Detector context hint (verify visually): video='{video_source}', frames={temporal_indices or [best_frame]}, "
        f"appearance_window={start_hms} to {end_hms} (peak frame {best_frame} at {best_hms}), "
        f"detected_counts={{arabian_oryx: {oryx_n}, arabian_gazelle: {gazelle_n}}}, detector_avg_conf={conf_str}.\n"
        "In description_ar, write a rich Arabic wildlife report covering counts of Arabian Oryx (المها العربي) "
        "and Arabian Gazelle (الغزال العربي), their movement/behavior across the frames, terrain/environment, "
        "background, and estimated time of day. Output ONLY <description>{...}</description> JSON."
    )


def describe_track_for_rag(
    model: Any,
    processor: Any,
    frame_paths: Sequence[str | Path],
    species: str,
    context: Optional[Dict[str, Any]] = None,
    max_dim: int = 512,
    max_new_tokens: int = 384,
    use_adapter: bool = False,
) -> SceneDescriptionResult:
    """Use Base Qwen3-VL (adapter disabled by default) on up to 3 chronological frames to generate rich RAG captions."""
    import torch

    imgs: List[Image.Image] = []
    for p in list(frame_paths)[:3]:
        im = Image.open(p).convert("RGB")
        im.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        imgs.append(im)

    if not imgs:
        raise ValueError("describe_track_for_rag requires at least one valid frame path.")

    merged_ctx = dict(context or {})
    merged_ctx.setdefault("species", species)
    usr_prompt = _format_scene_user_prompt(merged_ctx, num_frames=len(imgs))

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SCENE_DESCRIPTION_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                *[{"type": "image"} for _ in imgs],
                {"type": "text", "text": usr_prompt},
            ],
        },
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=imgs, padding=True, return_tensors="pt").to(model.device)

    adapter_ctx = (
        model.disable_adapter()
        if (not use_adapter and hasattr(model, "disable_adapter"))
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), adapter_ctx:
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, gen_ids)]
    raw_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    return parse_scene_description_response(raw_text)


class QwenWildlifeVLM:
    """Managed Qwen3-VL inference service supporting classification and RAG scene description."""

    def __init__(self, config: QwenVLMConfig) -> None:
        self.config = config
        self.processor: Any = None
        self.model: Any = None
        self._has_peft_adapter: bool = False

    def load(self) -> "QwenWildlifeVLM":
        """Load Qwen3-VL base model and optional PEFT LoRA adapter into GPU memory."""
        if self.model is not None and self.processor is not None:
            return self

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("Loading Qwen3-VL processor & base model: %s", self.config.model_id)
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            min_pixels=self.config.min_pixels,
            max_pixels=self.config.max_pixels,
        )
        # Left padding is required for batched autoregressive generation
        self.processor.tokenizer.padding_side = "left"

        dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else (torch.float16 if torch.cuda.is_available() else torch.float32)
        )
        base_model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_id,
            device_map=self.config.device_map,
            dtype=dtype,
        )

        if self.config.lora_dir:
            from peft import PeftModel

            logger.info("Attaching PEFT LoRA adapter from: %s", self.config.lora_dir)
            self.model = PeftModel.from_pretrained(base_model, self.config.lora_dir)
            self._has_peft_adapter = True
        else:
            self.model = base_model
            self._has_peft_adapter = False

        self.model.eval()
        return self

    def unload(self) -> None:
        """Unload model weights and clear CUDA cache."""
        self.model = None
        self.processor = None
        self._has_peft_adapter = False
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def __enter__(self) -> "QwenWildlifeVLM":
        return self.load()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.unload()

    def _adapter_context(self, use_adapter: bool):
        """Return a context manager that disables LoRA weights when base VLM fluency is preferred."""
        if self._has_peft_adapter and not use_adapter and hasattr(self.model, "disable_adapter"):
            return self.model.disable_adapter()
        return contextlib.nullcontext()

    def _generate_batch(
        self,
        image_groups: Sequence[Sequence[Image.Image]],
        system_prompts: Sequence[str],
        user_prompts: Sequence[str],
        max_new_tokens: int,
        use_adapter: bool = True,
    ) -> List[str]:
        """Run deterministic batched multi-image generation on Qwen3-VL (1 to N images per prompt)."""
        import torch

        if self.model is None or self.processor is None:
            self.load()

        prompts: List[str] = []
        flat_images: List[Image.Image] = []
        for imgs, sys_p, usr_p in zip(image_groups, system_prompts, user_prompts):
            img_list = list(imgs)
            flat_images.extend(img_list)
            msgs = [
                {"role": "system", "content": [{"type": "text", "text": sys_p}]},
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image"} for _ in img_list],
                        {"type": "text", "text": usr_p},
                    ],
                },
            ]
            prompts.append(
                self.processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            )

        inputs = self.processor(
            text=prompts,
            images=flat_images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode(), self._adapter_context(use_adapter=use_adapter):
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, gen_ids)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)

    def classify_crops_batch(
        self,
        zoomed_images: Sequence[Image.Image],
    ) -> List[VLMClassificationResult]:
        """Classify a batch of 3x zoomed visual prompts into arabian_oryx, arabian_gazelle, or none."""
        if not zoomed_images:
            return []

        n = len(zoomed_images)
        raw_texts = self._generate_batch(
            image_groups=[[img] for img in zoomed_images],
            system_prompts=[CLOSED_SET_CLASSIFICATION_PROMPT] * n,
            user_prompts=["Classify ONLY the object inside the red rectangle."] * n,
            max_new_tokens=self.config.classification_max_new_tokens,
            use_adapter=True,
        )
        return [parse_classification_response(txt) for txt in raw_texts]

    def describe_scenes_batch(
        self,
        frame_groups: Sequence[Image.Image | Sequence[Image.Image]],
        scene_contexts: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> List[SceneDescriptionResult]:
        """Generate rich structured RAG scene descriptions from 1-to-3 chronological frames per track.

        By default, disables the GRPO classification LoRA adapter (0 extra VRAM) so Base Qwen3-VL
        produces fluent, descriptive multi-frame narratives capturing animal movement & behavior.
        """
        if not frame_groups:
            return []

        normalized_groups: List[List[Image.Image]] = []
        for item in frame_groups:
            if isinstance(item, Image.Image):
                normalized_groups.append([item])
            else:
                normalized_groups.append(list(item)[:3])

        n = len(normalized_groups)
        contexts = list(scene_contexts) if scene_contexts is not None else [None] * n
        user_prompts = [
            _format_scene_user_prompt(ctx, num_frames=len(grp))
            for ctx, grp in zip(contexts, normalized_groups)
        ]

        raw_texts = self._generate_batch(
            image_groups=normalized_groups,
            system_prompts=[SCENE_DESCRIPTION_SYSTEM_PROMPT] * n,
            user_prompts=user_prompts,
            max_new_tokens=self.config.description_max_new_tokens,
            use_adapter=self.config.use_adapter_for_description,
        )
        return [parse_scene_description_response(txt) for txt in raw_texts]

    def describe_track_for_rag(
        self,
        frame_paths: Sequence[str | Path],
        species: str = "wildlife",
        context: Optional[Dict[str, Any]] = None,
        max_dim: int = 512,
    ) -> SceneDescriptionResult:
        """Describe up to 3 chronological frames of a tracked animal using Base Qwen3-VL (adapter disabled)."""
        if self.model is None or self.processor is None:
            self.load()
        return describe_track_for_rag(
            model=self.model,
            processor=self.processor,
            frame_paths=frame_paths,
            species=species,
            context=context,
            max_dim=max_dim,
            max_new_tokens=self.config.description_max_new_tokens,
            use_adapter=self.config.use_adapter_for_description,
        )

    def describe_single_image(
        self,
        image_path: str | Path,
        context: Optional[Dict[str, Any]] = None,
        max_dim: int = 512,
    ) -> SceneDescriptionResult:
        """Convenience wrapper for an Agent tool to describe a single wildlife image file."""
        raw_bgr = cv2.imread(str(image_path))
        if raw_bgr is None:
            raise FileNotFoundError(f"Could not read image at: {image_path}")
        pil_img = prepare_full_frame_pil(raw_bgr, max_dim=max_dim)
        return self.describe_scenes_batch([[pil_img]], [context])[0]
