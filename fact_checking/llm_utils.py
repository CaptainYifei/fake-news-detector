import json
import re
from typing import Any, Dict, List


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract a JSON object from an LLM response."""
    if not text:
        return {}

    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def extract_json_array(text: str) -> List[Any]:
    """Extract a JSON array from an LLM response."""
    if not text:
        return []

    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def normalize_verdict(verdict: str) -> str:
    raw = (verdict or "").strip().upper().replace("-", "_").replace(" ", "_")
    mapping = {
        "TRUE": "TRUE",
        "SUPPORTED": "TRUE",
        "CORRECT": "TRUE",
        "正确": "TRUE",
        "FALSE": "FALSE",
        "REFUTED": "FALSE",
        "INCORRECT": "FALSE",
        "错误": "FALSE",
        "PARTIALLY_TRUE": "PARTIALLY_TRUE",
        "PARTIAL": "PARTIALLY_TRUE",
        "部分正确": "PARTIALLY_TRUE",
        "NOT_ENOUGH_INFO": "UNVERIFIABLE",
        "NEI": "UNVERIFIABLE",
        "UNKNOWN": "UNVERIFIABLE",
        "UNVERIFIABLE": "UNVERIFIABLE",
        "无法验证": "UNVERIFIABLE",
    }
    return mapping.get(raw, "UNVERIFIABLE")


def looks_like_json(text: str) -> bool:
    stripped = (text or "").strip()
    return stripped.startswith("{") or stripped.startswith("[") or "```json" in stripped[:20].lower()


def compact_text(text: str, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    return cleaned[:limit]
