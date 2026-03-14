import io
import json
import re
from pathlib import Path

from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions
from PIL import Image
import pillow_heif
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Register HEIF opener at module level (idempotent)
pillow_heif.register_heif_opener()

GEMINI_MODEL = "gemini-2.5-flash"
MAX_IMAGE_DIM = 8000

ANALYSIS_PROMPT = """Analyze this image and identify the location and best text placement.
Return ONLY a valid JSON object (no markdown, no explanation) with this exact structure:
{
  "location": {
    "city": "city name or null",
    "region": "state/province/region or null",
    "country": "country name or null",
    "landmark": "specific landmark name or null",
    "popular_name": "well-known name tourists would recognize or null",
    "location_type": "landmark|nature|urban|rural|indoor|unknown",
    "confidence": "high|medium|low"
  },
  "placement": {
    "recommendation": "top-left|top-right|bottom-left|bottom-right|top-center|bottom-center",
    "reasoning": "brief explanation",
    "subject_position": "where the main subject is",
    "quiet_regions": ["list of regions with minimal visual content"]
  },
  "tags": ["5 to 8 descriptive tags"]
}

If location cannot be determined, use null for location fields and set confidence to "low".
Choose placement recommendation to avoid covering the main subject."""

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
        """Analyze an image and return location/placement data with token counts."""
        try:
            image_bytes = self._load_image_bytes(image_path)
            response = await self._call_api(image_bytes)
            response_text = response.text
            data = self._parse_response(response_text)
            validated = self._validate_response(data)
            usage = response.usage_metadata
            return {
                "success": True,
                "location": validated["location"],
                "placement": validated["placement"],
                "tags": validated["tags"],
                "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
                "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
            }
        except google_exceptions.InvalidArgument as e:
            return {"success": False, "error": f"Invalid request: {e}", "input_tokens": 0, "output_tokens": 0}
        except Exception as e:
            return {"success": False, "error": str(e), "input_tokens": 0, "output_tokens": 0}

    def _load_image_bytes(self, path: Path) -> bytes:
        """Load image as JPEG bytes, handling HEIC and resizing if needed."""
        suffix = path.suffix.lower()
        if suffix == ".heic":
            heif_file = pillow_heif.open_heif(str(path))
            img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
        else:
            img = Image.open(path)
            img.load()

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
    async def _call_api(self, image_bytes: bytes):
        """Call Gemini Vision API with retry logic."""
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        return await self._client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=[image_part, ANALYSIS_PROMPT],
            config=types.GenerateContentConfig(
                max_output_tokens=1024,
                temperature=0.1,
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
