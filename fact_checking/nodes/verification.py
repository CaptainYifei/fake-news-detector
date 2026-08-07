from typing import Dict, List

from ..llm_utils import extract_json_object, looks_like_json, normalize_verdict
from ..schemas import AtomicCheck, ClaimExtractionResult, EvidenceRetrievalResult, VerificationResult


class VerificationNode:
    """
    Node 3: multi-step decomposed claim verification.

    This lightweight version asks the LLM to perform atomic decomposition, 5W
    structuring, question answering over evidence, and final aggregation in a
    single structured verification call.
    """

    def __init__(
        self,
        client,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
        output_language: str = "auto",
    ):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.output_language = output_language

    def run(
        self,
        claim_result: ClaimExtractionResult,
        evidence_result: EvidenceRetrievalResult,
    ) -> VerificationResult:
        if not evidence_result.evidence_chunks:
            return VerificationResult(
                final_verdict="UNVERIFIABLE",
                reasoning="无法找到相关证据进行核查。",
            )

        evidence_text = self._format_evidence(evidence_result.evidence_chunks)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": (
                        f"中心声明:\n{claim_result.claim}\n\n"
                        f"新闻中心思想:\n{claim_result.central_summary}\n\n"
                        f"证据:\n{evidence_text}"
                    ),
                },
            ],
            temperature=self.temperature,
            max_tokens=max(self.max_tokens, 1800),
        )

        data = extract_json_object(response.choices[0].message.content or "")
        atomic_checks = self._parse_atomic_checks(data.get("atomic_checks", []))
        final_verdict = normalize_verdict(str(data.get("final_verdict") or ""))
        reasoning = self._resolve_reasoning(
            data=data,
            atomic_checks=atomic_checks,
            raw_content=response.choices[0].message.content or "",
        )

        return VerificationResult(
            atomic_checks=atomic_checks,
            final_verdict=final_verdict,
            reasoning=reasoning,
        )

    def _system_prompt(self) -> str:
        return """
You are the verification node of a fact-checking system.
Use the thesis idea of multi-step decomposed claim verification, implemented
only with LLM reasoning.

Do the following:
1. Decompose the central claim into at most 3 non-overlapping atomic claims.
2. For each atomic claim, extract 5W elements: who, what, when, where, why.
3. Generate verification questions for the 5W elements.
4. Answer each question using only the provided evidence.
5. Decide each atomic claim as SUPPORTED, REFUTED, or NOT_ENOUGH_INFO.
6. Aggregate all atomic checks into TRUE, FALSE, PARTIALLY_TRUE, or UNVERIFIABLE.

Rules:
- Do not use outside knowledge.
- Cite evidence by numeric ids from the evidence list.
- If evidence is missing or contradictory, say so.
- Do not repeat the same atomic claim.
- Keep each reasoning field within 80 Chinese characters.
- Keep the overall reasoning within 160 Chinese characters.
- Return only valid JSON.

Schema:
{
  "atomic_checks": [
    {
      "atomic_claim": "...",
      "five_w": {
        "who": "...",
        "what": "...",
        "when": "...",
        "where": "...",
        "why": "..."
      },
      "questions": ["..."],
      "answers": [
        {"question": "...", "answer": "...", "evidence_id": "1"}
      ],
      "verdict": "SUPPORTED / REFUTED / NOT_ENOUGH_INFO",
      "reasoning": "...",
      "evidence_ids": [1]
    }
  ],
  "final_verdict": "TRUE / FALSE / PARTIALLY_TRUE / UNVERIFIABLE",
  "reasoning": "overall reasoning in Chinese unless the input is clearly not Chinese"
}
""".strip()

    def _format_evidence(self, evidence_chunks: List[Dict]) -> str:
        lines = []
        for idx, chunk in enumerate(evidence_chunks, start=1):
            lines.append(
                "\n".join(
                    [
                        f"[{idx}] {chunk.get('text', '')}",
                        f"Source: {chunk.get('source', '')}",
                        f"Relevance: {chunk.get('similarity', 0.0):.2f}",
                    ]
                )
            )
        return "\n\n".join(lines)

    def _resolve_reasoning(
        self,
        data: Dict,
        atomic_checks: List[AtomicCheck],
        raw_content: str,
    ) -> str:
        reasoning = str(data.get("reasoning") or "").strip()
        if reasoning:
            return reasoning

        if atomic_checks:
            supported = sum(1 for check in atomic_checks if check.verdict.upper() == "SUPPORTED")
            refuted = sum(1 for check in atomic_checks if check.verdict.upper() == "REFUTED")
            unknown = len(atomic_checks) - supported - refuted
            parts = []
            if supported:
                parts.append(f"{supported} 个原子声明被证据支持")
            if refuted:
                parts.append(f"{refuted} 个原子声明被证据反驳")
            if unknown:
                parts.append(f"{unknown} 个原子声明证据不足")
            return "；".join(parts) + "。"

        if raw_content.strip() and not looks_like_json(raw_content):
            return raw_content.strip()

        return "验证模型返回了结构化结果，但未能解析出可展示的推理摘要。"

    def _parse_atomic_checks(self, value) -> List[AtomicCheck]:
        if not isinstance(value, list):
            return []

        checks: List[AtomicCheck] = []
        seen_claims = set()
        for item in value:
            if not isinstance(item, dict):
                continue
            atomic_claim = str(item.get("atomic_claim") or "").strip()
            claim_key = self._normalize_claim_key(atomic_claim)
            if not atomic_claim or claim_key in seen_claims:
                continue
            seen_claims.add(claim_key)
            checks.append(
                AtomicCheck(
                    atomic_claim=atomic_claim,
                    five_w=self._as_string_dict(item.get("five_w", {})),
                    questions=self._as_string_list(item.get("questions", [])),
                    answers=self._as_answer_list(item.get("answers", [])),
                    verdict=str(item.get("verdict") or "NOT_ENOUGH_INFO").strip(),
                    reasoning=str(item.get("reasoning") or "").strip(),
                    evidence_ids=self._as_int_list(item.get("evidence_ids", [])),
                )
            )
        return checks

    def _normalize_claim_key(self, text: str) -> str:
        return "".join(str(text or "").split()).lower()

    def _as_string_dict(self, value) -> Dict[str, str]:
        if not isinstance(value, dict):
            return {}
        return {str(key): str(val) for key, val in value.items()}

    def _as_string_list(self, value) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _as_answer_list(self, value) -> List[Dict[str, str]]:
        if not isinstance(value, list):
            return []
        answers = []
        for item in value:
            if isinstance(item, dict):
                answers.append({str(key): str(val) for key, val in item.items()})
        return answers

    def _as_int_list(self, value) -> List[int]:
        if not isinstance(value, list):
            return []
        ids = []
        for item in value:
            try:
                ids.append(int(item))
            except (TypeError, ValueError):
                continue
        return ids
