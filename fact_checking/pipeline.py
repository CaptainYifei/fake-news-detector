from typing import Callable, Dict, List

from .nodes import ClaimExtractionNode, EvidenceRetrievalNode, VerificationNode
from .schemas import FactCheckResult


class FactCheckingPipeline:
    """
    Three-node fact-checking pipeline inspired by chapters 3, 4, and 5.
    """

    def __init__(
        self,
        client,
        model: str,
        max_tokens: int,
        temperature: float,
        detect_language: Callable[[str], str],
        search_executor: Callable[[str, int], List[Dict]],
        evidence_ranker: Callable[[List[Dict], str, int], List[Dict]],
        output_language: str = "auto",
        evidence_min_similarity: float = 0.45,
        evidence_entity_gate: bool = True,
        evidence_ranker_multiplier: int = 3,
    ):
        self.claim_node = ClaimExtractionNode(
            client=client,
            model=model,
            max_tokens=max_tokens,
            detect_language=detect_language,
            output_language=output_language,
        )
        self.evidence_node = EvidenceRetrievalNode(
            client=client,
            model=model,
            max_tokens=max_tokens,
            search_executor=search_executor,
            evidence_ranker=evidence_ranker,
            detect_language=detect_language,
            output_language=output_language,
            evidence_min_similarity=evidence_min_similarity,
            evidence_entity_gate=evidence_entity_gate,
            ranker_multiplier=evidence_ranker_multiplier,
        )
        self.verification_node = VerificationNode(
            client=client,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            output_language=output_language,
        )

    def run(
        self,
        news_text: str,
        search_results_per_query: int = 5,
        evidence_top_k: int = 10,
    ) -> FactCheckResult:
        claim_result = self.claim_node.run(news_text)
        evidence_result = self.evidence_node.run(
            claim_result=claim_result,
            num_results=search_results_per_query,
            top_k=evidence_top_k,
        )
        verification_result = self.verification_node.run(
            claim_result=claim_result,
            evidence_result=evidence_result,
        )

        return FactCheckResult(
            original_text=news_text,
            claim_extraction=claim_result,
            evidence_retrieval=evidence_result,
            verification=verification_result,
        )
