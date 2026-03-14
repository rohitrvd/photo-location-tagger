import io
import json
import re
from pathlib import Path

from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions
from PIL import Image, ImageOps
import pillow_heif
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

pillow_heif.register_heif_opener()

GEMINI_MODEL = "gemini-2.5-flash"
MAX_IMAGE_DIM = 8000

# Primary prompt — comprehensive location identification
ANALYSIS_PROMPT = """You are a travel photo location identifier. Examine this image carefully and identify the location using ALL available clues:

1. READ ALL TEXT visible in the image — signs, banners, plaques, shop names, street signs, monument inscriptions. These are the strongest clues.
2. RECOGNIZE landmarks — famous buildings, monuments, natural wonders, skylines, bridges, towers.
3. USE CONTEXT — architectural style, vegetation, vehicles, clothing, language of visible text, geographic features.
4. MAKE YOUR BEST GUESS — even if not 100% certain, provide your best identification with an appropriate confidence level. Only use confidence "low" if you have absolutely no idea.

Return ONLY a valid JSON object (no markdown, no explanation):
{
  "location": {
    "city": "city name or null",
    "region": "state/province/region or null",
    "country": "country name or null",
    "landmark": "specific landmark or place name or null",
    "popular_name": "the most recognizable name a tourist would use or null",
    "location_type": "landmark|nature|urban|rural|indoor|unknown",
    "confidence": "high|medium|low"
  },
  "placement": {
    "recommendation": "top-left|top-right|bottom-left|bottom-right|top-center|bottom-center",
    "reasoning": "brief explanation",
    "subject_position": "where the main subject is",
    "quiet_regions": ["regions with minimal visual content"]
  },
  "tags": ["5 to 8 descriptive tags"]
}

Confidence guide:
- "high": you are confident — clear landmark, readable sign, or unmistakable location
- "medium": reasonable guess based on visual clues — use this instead of low when you have any supporting evidence
- "low": genuinely no identifiable information whatsoever

Choose placement to avoid covering the main subject."""

# Second-pass prompt used when first pass returns low confidence
RETRY_PROMPT = """Look very carefully at this photo again. Focus specifically on:
- Any text, words, or numbers visible anywhere in the image (signs, labels, inscriptions, watermarks)
- Any distinctive architectural features, natural formations, or landmarks
- Any flags, logos, or symbols that indicate a country or region
- The style of buildings, roads, vehicles that might indicate a country

Even a partial location (just a country, or just a region) is better than nothing. If you can identify anything at all, use confidence "medium".

Return ONLY a valid JSON object (no markdown):
{
  "location": {
    "city": "city name or null",
    "region": "state/province/region or null",
    "country": "country name or null",
    "landmark": "specific landmark or null",
    "popular_name": "recognizable tourist name or null",
    "location_type": "landmark|nature|urban|rural|indoor|unknown",
    "confidence": "high|medium|low"
  },
  "placement": {
    "recommendation": "top-left|top-right|bottom-left|bottom-right|top-center|bottom-center",
    "reasoning": "brief explanation",
    "subject_position": "where the main subject is",
    "quiet_regions": []
  },
  "tags": ["5 to 8 descriptive tags"]
}"""

_RETRYABLE = (
    google_exceptions.ResourceExhausted,
    google_exceptions.ServiceUnavailable,
    google_exceptions.DeadlineExceeded,
    google_exceptions.InternalServerError,
)


class ImageAnalyzer:
    def __init__(self, api_key: str):
        self._client = genai.Client(api_key=api_key)

    async def analyze(self, image_path: Path) -> dict:
        """Analyze an image, with a second pass if first returns low confidence."""
        try:
            image_bytes = self._load_image_bytes(image_path)

            # First pass
            response = await self._call_api(image_bytes, ANALYSIS_PROMPT)
            data = self._parse_response(response.text)
            validated = self._validate_response(data)
            usage = response.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0

            # Second pass if first returned low confidence with no useful location info
            loc = validated["location"]
            has_useful_info = loc.get("popular_name") or loc.get("landmark") or loc.get("city")
            if loc.get("confidence") == "low" and not has_useful_info:
                response2 = await self._call_api(image_bytes, RETRY_PROMPT)
                data2 = self._parse_response(response2.text)
                validated2 = self._validate_response(data2)
                usage2 = response2.usage_metadata
                input_tokens += getattr(usage2, "prompt_token_count", 0) or 0
                output_tokens += getattr(usage2, "candidates_token_count", 0) or 0

                # Use second pass result only if it's more informative
                loc2 = validated2["location"]
                if loc2.get("popular_name") or loc2.get("landmark") or loc2.get("city"):
                    validated = validated2

            return {
                "success": True,
                "location": validated["location"],
                "placement": validated["placement"],
                "tags": validated["tags"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        except google_exceptions.InvalidArgument as e:
            return {"success": False, "error": f"Invalid request: {e}", "input_tokens": 0, "output_tokens": 0}
        except Exception as e:
            return {"success": False, "error": str(e), "input_tokens": 0, "output_tokens": 0}

    def _load_image_bytes(self, path: Path) -> bytes:
        suffix = path.suffix.lower()
        if suffix == ".heic":
            try:
                heif_file = pillow_heif.open_heif(str(path))
                img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
            except Exception:
                # File has .heic extension but is actually JPEG or another format
                img = Image.open(path)
                img.load()
        else:
            img = Image.open(path)
            img.load()

        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        if img.mode == "RGBA":
            img = img.convert("RGB")

        img = self._resize_if_needed(img)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        return buf.read()

    def _resize_if_needed(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w > MAX_IMAGE_DIM or h > MAX_IMAGE_DIM:
            ratio = min(MAX_IMAGE_DIM / w, MAX_IMAGE_DIM / h)
            return img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        return img

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=30),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    async def _call_api(self, image_bytes: bytes, prompt: str):
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        return await self._client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=1024,
                temperature=0,
            ),
        )

    def _parse_response(self, text: str) -> dict:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {}

    def _validate_response(self, data: dict) -> dict:
        valid_confidences = {"high", "medium", "low"}
        valid_placements = {
            "top-left", "top-right", "bottom-left", "bottom-right",
            "top-center", "bottom-center"
        }

        location = data.get("location", {})
        if not isinstance(location, dict):
            location = {}
        location.setdefault("city", None)
        location.setdefault("region", None)
        location.setdefault("country", None)
        location.setdefault("landmark", None)
        location.setdefault("popular_name", None)
        location.setdefault("location_type", "unknown")
        if location.get("confidence") not in valid_confidences:
            location["confidence"] = "low"

        placement = data.get("placement", {})
        if not isinstance(placement, dict):
            placement = {}
        if placement.get("recommendation") not in valid_placements:
            placement["recommendation"] = "bottom-right"
        placement.setdefault("reasoning", "")
        placement.setdefault("subject_position", "")
        placement.setdefault("quiet_regions", [])

        tags = data.get("tags", [])
        if not isinstance(tags, list):
            tags = []

        return {"location": location, "placement": placement, "tags": tags}
