import re
from typing import Callable, Dict, List, Tuple

from ..llm_utils import compact_text, extract_json_array
from ..schemas import ClaimExtractionResult, EvidenceRetrievalResult, QueryRewrite


class EvidenceRetrievalNode:
    """
    Node 2: query-rewriting hierarchical evidence retrieval.

    The full thesis version uses query rewriting, document retrieval, weighted
    ranking, and sentence selection. This lightweight implementation keeps the
    hierarchy but delegates query rewriting to the LLM and reuses the existing
    search + embedding ranker.
    """

    def __init__(
        self,
        client,
        model: str,
        max_tokens: int,
        search_executor: Callable[[str, int], List[Dict]],
        evidence_ranker: Callable[[List[Dict], str, int], List[Dict]],
        detect_language: Callable[[str], str],
        output_language: str = "auto",
        evidence_min_similarity: float = 0.45,
        evidence_entity_gate: bool = True,
        ranker_multiplier: int = 3,
    ):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.search_executor = search_executor
        self.evidence_ranker = evidence_ranker
        self.detect_language = detect_language
        self.output_language = output_language
        self.evidence_min_similarity = evidence_min_similarity
        self.evidence_entity_gate = evidence_entity_gate
        self.ranker_multiplier = max(1, ranker_multiplier)

    def run(
        self,
        claim_result: ClaimExtractionResult,
        num_results: int = 5,
        top_k: int = 10,
    ) -> EvidenceRetrievalResult:
        queries = self.rewrite_queries(claim_result)
        documents = self._search_documents(queries, num_results)
        ranked_top_k = max(top_k * self.ranker_multiplier, top_k)
        ranked_chunks = self.evidence_ranker(
            documents,
            claim_result.claim,
            ranked_top_k,
        )
        evidence_chunks, filter_summary = self._filter_evidence_chunks(
            ranked_chunks,
            claim_result=claim_result,
            top_k=top_k,
        )
        return EvidenceRetrievalResult(
            queries=queries,
            documents=documents,
            evidence_chunks=evidence_chunks,
            filter_summary=filter_summary,
        )

    def rewrite_queries(self, claim_result: ClaimExtractionResult) -> List[QueryRewrite]:
        target_language = (
            self.detect_language(claim_result.claim)
            if self.output_language == "auto"
            else self.output_language
        )
        prompt = self._query_prompt(target_language)
        user_content = {
            "claim": claim_result.claim,
            "central_summary": claim_result.central_summary,
            "entities": claim_result.entities,
        }

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": str(user_content)},
            ],
            temperature=0.0,
            max_tokens=min(max(self.max_tokens, 800), 1600),
        )
        data = extract_json_array(response.choices[0].message.content or "")
        queries: List[QueryRewrite] = []

        for item in data:
            if not isinstance(item, dict):
                continue
            query = str(item.get("query") or "").strip()
            if not query:
                continue
            queries.append(
                QueryRewrite(
                    query=query,
                    purpose=str(item.get("purpose") or "").strip(),
                    language=str(item.get("language") or "auto").strip() or "auto",
                )
            )

        if not queries:
            queries = [
                QueryRewrite(
                    query=claim_result.claim,
                    purpose="original central claim",
                    language=target_language,
                )
            ]

        return self._dedupe_queries(queries)[:4]

    def _query_prompt(self, target_language: str) -> str:
        return f"""
You are the evidence retrieval node of a fact-checking system.
Rewrite the claim into high-quality web search queries.

Follow these rules:
1. Focus on core entities, time, place, event, and numbers.
2. Generate complementary queries, not paraphrases only.
3. Include one direct query, one entity-focused query, and one verification or
   contradiction-oriented query when possible.
4. Keep queries short enough for a search engine.
5. Return only valid JSON array.

Schema:
[
  {{"query": "...", "purpose": "direct/entity/time/contradiction", "language": "{target_language}"}}
]
""".strip()

    def _search_documents(
        self, queries: List[QueryRewrite], num_results: int
    ) -> List[Dict]:
        documents: List[Dict] = []
        seen_urls = set()

        for query in queries:
            for doc in self.search_executor(query.query, num_results):
                url = doc.get("url")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                normalized = {
                    "title": compact_text(doc.get("title", ""), 300),
                    "url": url,
                    "snippet": compact_text(doc.get("snippet", ""), 1200),
                    "search_query": query.query,
                    "query_purpose": query.purpose,
                    "search_language": query.language,
                    "detected_language": query.language,
                }
                documents.append(normalized)

        return documents

    def _dedupe_queries(self, queries: List[QueryRewrite]) -> List[QueryRewrite]:
        seen = set()
        unique = []
        for query in queries:
            key = query.query.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(query)
        return unique

    def _filter_evidence_chunks(
        self,
        chunks: List[Dict],
        claim_result: ClaimExtractionResult,
        top_k: int,
    ) -> Tuple[List[Dict], Dict]:
        entity_terms = self._normalize_entities(claim_result.entities)
        filtered = []
        dropped_by_similarity = 0
        dropped_by_entity = 0

        for chunk in chunks:
            similarity = self._as_float(chunk.get("similarity"))
            if similarity < self.evidence_min_similarity:
                dropped_by_similarity += 1
                continue

            if self.evidence_entity_gate and entity_terms:
                text = self._normalize_text(
                    f"{chunk.get('text', '')} {chunk.get('source', '')}"
                )
                if not any(entity in text for entity in entity_terms):
                    dropped_by_entity += 1
                    continue

            filtered.append(chunk)

        filtered.sort(key=lambda item: self._as_float(item.get("similarity")), reverse=True)
        selected = filtered[:top_k]

        return selected, {
            "input_count": len(chunks),
            "selected_count": len(selected),
            "top_k": top_k,
            "min_similarity": self.evidence_min_similarity,
            "entity_gate": self.evidence_entity_gate,
            "entities": claim_result.entities,
            "dropped_by_similarity": dropped_by_similarity,
            "dropped_by_entity": dropped_by_entity,
        }

    def _normalize_entities(self, entities: List[str]) -> List[str]:
        normalized = []
        for entity in entities:
            text = self._normalize_text(entity)
            if len(text) >= 2:
                normalized.append(text)
        return list(dict.fromkeys(normalized))

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", "", str(text or "")).lower()

    def _as_float(self, value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
