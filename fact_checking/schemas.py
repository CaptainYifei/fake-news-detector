from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class ClaimExtractionResult:
    central_summary: str
    claim: str
    check_worthiness: float = 0.0
    entities: List[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryRewrite:
    query: str
    purpose: str = ""
    language: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceRetrievalResult:
    queries: List[QueryRewrite] = field(default_factory=list)
    documents: List[Dict[str, Any]] = field(default_factory=list)
    evidence_chunks: List[Dict[str, Any]] = field(default_factory=list)
    filter_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["queries"] = [query.to_dict() for query in self.queries]
        return data


@dataclass
class AtomicCheck:
    atomic_claim: str
    five_w: Dict[str, str] = field(default_factory=dict)
    questions: List[str] = field(default_factory=list)
    answers: List[Dict[str, str]] = field(default_factory=list)
    verdict: str = "NOT_ENOUGH_INFO"
    reasoning: str = ""
    evidence_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    atomic_checks: List[AtomicCheck] = field(default_factory=list)
    final_verdict: str = "UNVERIFIABLE"
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["atomic_checks"] = [check.to_dict() for check in self.atomic_checks]
        return data


@dataclass
class FactCheckResult:
    original_text: str
    claim_extraction: ClaimExtractionResult
    evidence_retrieval: EvidenceRetrievalResult
    verification: VerificationResult

    @property
    def claim(self) -> str:
        return self.claim_extraction.claim

    @property
    def verdict(self) -> str:
        return self.verification.final_verdict

    @property
    def reasoning(self) -> str:
        return self.verification.reasoning

    @property
    def evidence_chunks(self) -> List[Dict[str, Any]]:
        return self.evidence_retrieval.evidence_chunks

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "claim": self.claim,
            "verdict": self.verdict,
            "reasoning": self.reasoning,
            "claim_extraction": self.claim_extraction.to_dict(),
            "evidence_retrieval": self.evidence_retrieval.to_dict(),
            "verification": self.verification.to_dict(),
            "evidence": self.evidence_chunks,
        }
