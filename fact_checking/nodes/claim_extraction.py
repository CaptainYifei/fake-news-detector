from typing import Callable, List

from ..llm_utils import extract_json_object
from ..schemas import ClaimExtractionResult


class ClaimExtractionNode:
    """
    Node 1: semantic central claim extraction.

    This is a lightweight LLM-only version of chapter 3: central semantic
    understanding, contextual disambiguation, and check-worthiness selection are
    all handled in a single structured prompt.
    """

    def __init__(
        self,
        client,
        model: str,
        max_tokens: int,
        detect_language: Callable[[str], str],
        output_language: str = "auto",
    ):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.detect_language = detect_language
        self.output_language = output_language

    def run(self, news_text: str) -> ClaimExtractionResult:
        target_language = (
            self.detect_language(news_text)
            if self.output_language == "auto"
            else self.output_language
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt(target_language)},
                {"role": "user", "content": news_text},
            ],
            temperature=0.0,
            max_tokens=min(max(self.max_tokens, 800), 2000),
        )
        content = response.choices[0].message.content or ""
        data = extract_json_object(content)

        claim = str(data.get("claim") or "").strip()
        if not claim:
            claim = news_text.strip()

        return ClaimExtractionResult(
            central_summary=str(data.get("central_summary") or claim).strip(),
            claim=claim,
            check_worthiness=self._as_score(data.get("check_worthiness")),
            entities=self._as_entities(data.get("entities")),
            reason=str(data.get("reason") or "").strip(),
        )

    def _system_prompt(self, target_language: str) -> str:
        language_hint = {
            "zh": "请用中文输出。",
            "en": "Write in English.",
            "ja": "日本語で出力してください。",
            "ko": "한국어로 출력하세요.",
        }.get(target_language, "Use the same language as the input.")

        return f"""
You are the claim extraction node of a fact-checking system.
Follow the thesis idea of semantic central claim extraction, but implement it
with LLM reasoning only.

Task:
1. Understand the central meaning of the whole news text.
2. Select one central claim that is both important and independently verifiable.
3. Resolve ambiguous references using the original context.
4. Do not invent details that are not in the input.

Return only valid JSON with this schema:
{{
  "central_summary": "one-sentence central meaning",
  "claim": "the best central verifiable claim",
  "check_worthiness": 0.0,
  "entities": ["key people/organizations/places/times"],
  "reason": "brief reason for selecting this claim"
}}

If the text is not news or has no verifiable factual claim, still extract the
most concrete factual statement from the input; do not return refusal text as
the claim.

{language_hint}
""".strip()

    def _as_score(self, value) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, score))

    def _as_entities(self, value) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]
