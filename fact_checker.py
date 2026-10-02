import os
import json
from datetime import datetime
from typing import Dict, Iterator, List, Tuple, Any, Optional
import streamlit as st
from openai import OpenAI
import requests
from ddgs import DDGS
import numpy as np
import re
from fact_checking import FactCheckingPipeline, FactCheckResult

# You.com MCP 默认端点：免费档免 API Key；设置 YDC_API_KEY 后切换到已认证端点
YOUCOM_FREE_ENDPOINT = "https://api.you.com/mcp?profile=free"
YOUCOM_AUTH_ENDPOINT = "https://api.you.com/mcp"


def _parse_mcp_response_events(response) -> List[Dict[str, Any]]:
    """
    Parse all JSON-RPC events from an MCP streamable-HTTP response.

    The You.com MCP server replies with SSE (application/json, text/event-stream),
    where a tools/call result may be preceded by progress notifications; plain
    JSON bodies are handled as a single-event fallback. SSE payloads are UTF-8,
    but requests falls back to ISO-8859-1 for text/event-stream responses
    without an explicit charset, so the body is decoded explicitly here.
    """
    content_type = response.headers.get("Content-Type", "")
    body = response.content.decode("utf-8", errors="replace")
    events: List[Dict[str, Any]] = []

    if "text/event-stream" in content_type:
        for block in body.split("\n\n"):
            data = "".join(
                line[len("data:"):].strip()
                for line in block.splitlines()
                if line.startswith("data:")
            )
            if not data:
                continue
            try:
                events.append(json.loads(data))
            except json.JSONDecodeError:
                continue
    elif "application/json" in content_type and body.strip():
        events.append(json.loads(body))

    return events


def _youcom_mcp_search(
    query: str,
    count: int,
    endpoint: str,
    api_key: Optional[str] = None,
    timeout: int = 30,
    proxy: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Call the you-search tool on a You.com MCP server (streamable HTTP)
    and return the raw web results.
    """
    session = requests.Session()
    if proxy:
        session.proxies.update({"http": proxy, "https": proxy})
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def post(payload: Dict[str, Any]) -> requests.Response:
        response = session.post(endpoint, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response

    # 1) initialize 握手（会话 ID 存在则回传）
    init_response = post(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "fake-news-detector", "version": "1.0"},
            },
        }
    )
    session_id = init_response.headers.get("Mcp-Session-Id")
    if session_id:
        headers["Mcp-Session-Id"] = session_id

    # 2) notifications/initialized
    post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    # 3) tools/call you-search
    call_response = post(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "you-search",
                "arguments": {"query": query, "count": max(1, count)},
            },
        }
    )
    events = _parse_mcp_response_events(call_response)
    result_event = next(
        (event for event in events if event.get("id") == 2 and "result" in event),
        None,
    )
    if result_event is None:
        raise RuntimeError("You.com MCP returned no search result event")

    content = result_event.get("result", {}).get("content", [])
    text = next(
        (item.get("text") for item in content if item.get("type") == "text"),
        None,
    )
    if not text:
        raise RuntimeError("You.com MCP returned no text content")

    payload = json.loads(text)
    return payload.get("results", {}).get("web", []) or []


class FactChecker:
    def __init__(
        self,
        api_base: str,
        model: str,
        temperature: float,
        max_tokens: int,
        embedding_base_url: str = None,
        embedding_model: str = "bge-m3:latest",
        embedding_api_key: str = "lm-studio",
        search_engine: str = "searxng",
        searxng_url: str = None,
        output_language: str = "auto",
        search_config: dict = None,
    ):
        """
        Initialize the fact checker with configuration parameters.

        Args:
            api_base: The base URL for the LLM API
            model: The model to use for fact checking
            temperature: Temperature parameter for LLM
            max_tokens: Maximum tokens for LLM response
            embedding_base_url: The base URL for embedding API
            embedding_model: The embedding model name
            embedding_api_key: API key for embedding service
            search_engine: Search engine to use ('duckduckgo', 'searxng' or 'youcom')
            searxng_url: Base URL for SearXNG instance
        """
        self.api_base = api_base
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.openai_api_key = "EMPTY"  # Placeholder for local setup

        # Initialize the OpenAI client with local settings
        self.client = OpenAI(
            api_key=self.openai_api_key,
            base_url=self.api_base,
        )

        # Initialize embedding client for online API
        # 如果没有提供地址，使用环境变量或默认 Ollama OpenAI 兼容端点
        if embedding_base_url is None:
            import os
            embedding_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

        self.embedding_base_url = embedding_base_url
        self.embedding_model_name = embedding_model
        self.embedding_api_key = embedding_api_key

        # Create embedding client
        self.embedding_client = OpenAI(
            api_key=self.embedding_api_key,
            base_url=self.embedding_base_url,
        )

        # Set embedding_model to None as we're not using local model
        self.embedding_model = None

        # Search engine configuration
        self.search_engine = search_engine
        # 如果没有提供SearXNG地址，使用环境变量或默认localhost
        if searxng_url is None:
            import os
            searxng_url = os.getenv("SEARXNG_BASE_URL", "http://localhost:8090")
        self.searxng_url = searxng_url

        # Language configuration
        self.output_language = output_language

        # Search configuration
        self.search_config = search_config or {}

    def check_news(
        self,
        news_text: str,
        search_results_per_query: int = 5,
        evidence_top_k: int = 10,
    ) -> FactCheckResult:
        """
        Run the redesigned three-node fact-checking pipeline.
        """
        pipeline = self._build_pipeline()
        return pipeline.run(
            news_text,
            search_results_per_query=search_results_per_query,
            evidence_top_k=evidence_top_k,
        )

    def check_news_stream(
        self,
        news_text: str,
        search_results_per_query: int = 5,
        evidence_top_k: int = 10,
    ) -> Iterator[Tuple[str, Any]]:
        """
        Run the three-node pipeline and yield each node result as soon as it is ready.
        """
        pipeline = self._build_pipeline()

        claim_result = pipeline.claim_node.run(news_text)
        yield "claim", claim_result

        evidence_result = pipeline.evidence_node.run(
            claim_result=claim_result,
            num_results=search_results_per_query,
            top_k=evidence_top_k,
        )
        yield "evidence", evidence_result

        verification_result = pipeline.verification_node.run(
            claim_result=claim_result,
            evidence_result=evidence_result,
        )
        yield "verification", verification_result

        yield "complete", FactCheckResult(
            original_text=news_text,
            claim_extraction=claim_result,
            evidence_retrieval=evidence_result,
            verification=verification_result,
        )

    def _build_pipeline(self) -> FactCheckingPipeline:
        return FactCheckingPipeline(
            client=self.client,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            detect_language=self._detect_language,
            search_executor=self._search_by_config,
            evidence_ranker=self._rank_evidence_for_pipeline,
            output_language=self.output_language,
            evidence_min_similarity=self._get_float_config(
                "evidence_min_similarity", 0.45
            ),
            evidence_entity_gate=self._get_bool_config("evidence_entity_gate", True),
            evidence_ranker_multiplier=self._get_int_config(
                "evidence_ranker_multiplier", 3
            ),
        )

    def _search_by_config(self, query: str, num_results: int) -> List[Dict[str, str]]:
        if self.search_engine == "searxng":
            return self._search_with_searxng(query, num_results)
        if self.search_engine == "youcom":
            return self._search_with_youcom(query, num_results)
        return self._search_with_duckduckgo(query, num_results)

    def _rank_evidence_for_pipeline(
        self, evidence_docs: List[Dict[str, str]], claim: str, top_k: int
    ) -> List[Dict[str, Any]]:
        return self.get_evidence_chunks(evidence_docs, claim, top_k=top_k)

    def _get_float_config(self, key: str, default: float) -> float:
        try:
            return float(self.search_config.get(key, default))
        except (TypeError, ValueError):
            return default

    def _get_int_config(self, key: str, default: int) -> int:
        try:
            return int(self.search_config.get(key, default))
        except (TypeError, ValueError):
            return default

    def _get_bool_config(self, key: str, default: bool) -> bool:
        value = self.search_config.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _detect_language(self, text: str) -> str:
        """
        Simple language detection based on character patterns
        """
        # Check for Chinese characters
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        # Check for Japanese characters
        japanese_chars = len(re.findall(r"[\u3040-\u309f\u30a0-\u30ff]", text))
        # Check for Korean characters
        korean_chars = len(re.findall(r"[\uac00-\ud7af]", text))

        total_chars = len(text)
        if total_chars == 0:
            return "en"

        # If more than 30% are CJK characters, detect specific language
        cjk_ratio = (chinese_chars + japanese_chars + korean_chars) / total_chars
        if cjk_ratio > 0.3:
            if chinese_chars > japanese_chars and chinese_chars > korean_chars:
                return "zh"
            elif japanese_chars > korean_chars:
                return "ja"
            elif korean_chars > 0:
                return "ko"

        return "en"

    def _get_language_prompts(self, target_lang: str) -> dict:
        """
        Get localized prompts for the specified language
        """

        prompts = {
            "zh": {
                "extract_claim": """
                你是一个精确的声明提取助手。分析提供的新闻并总结其核心思想。
                将核心思想格式化为一个值得验证的陈述，即一个可以独立验证的声明。
                输出格式：
                claim: <声明>
                """,
                "evaluate_claim": """
                你是事实核查助手。根据证据判断声明的真实性。请用中文回复。

                格式要求：
                VERDICT: TRUE/FALSE/PARTIALLY TRUE
                REASONING: 你的中文推理过程

                重要：请确保推理过程使用中文撰写。
                """,
                "user_extract": "从以下文本中提取关键的事实声明：",
                "user_evaluate": "声明：{claim}\n\n证据：\n{evidence}\n\n请判断声明是否正确。",
            },
            "en": {
                "extract_claim": """
                You are a precise claim extraction assistant. Analyze the provided news and summarize the central idea of it.
                Format the central idea as a worthy-check statement, which is a claim that can be verified independently.
                output format:
                claim: <claim>
                """,
                "evaluate_claim": """
                You are a fact-checking assistant. Judge if the claim is true based on evidence.

                Format required:
                VERDICT: TRUE/FALSE/PARTIALLY TRUE
                REASONING: Your reasoning process
                """,
                "user_extract": "Extract the key factual claims from this text:",
                "user_evaluate": "CLAIM: {claim}\n\nEVIDENCE:\n{evidence}",
            },
            "ja": {
                "extract_claim": """
                あなたは正確なクレーム抽出アシスタントです。提供されたニュースを分析し、その中心的なアイデアを要約してください。
                中心的なアイデアを独立して検証可能なクレームとして形式化してください。
                出力形式：
                claim: <クレーム>
                """,
                "evaluate_claim": """
                あなたは正確なファクトチェックアシスタントです。提供されたクレームを分析し、提供された証拠に基づいてその正確性を判定してください。

                以下の手順に従ってください：
                1. 各証拠を慎重に検討する
                2. 証拠がクレームをどのように支持または反駁するかを評価する
                3. 明確な判定を提供する：TRUE（真）、FALSE（偽）、またはPARTIALLY TRUE（部分的に真）
                4. 具体的な証拠を引用して推論を説明する

                以下の形式で回答してください：

                VERDICT: [TRUE/FALSE/PARTIALLY TRUE]

                REASONING: [具体的な証拠を引用した詳細な説明]

                中立的で客観的であり、証拠が示すことのみに焦点を当ててください。提供された証拠を超えて推測しないでください。
                """,
                "user_extract": "このテキストから重要な事実のクレームを抽出してください：",
                "user_evaluate": "クレーム：{claim}\n\n証拠：\n{evidence}",
            },
        }

        return prompts.get(target_lang, prompts["en"])

    def _translate_claim(self, claim: str, target_languages: list) -> dict:
        """
        Translate claim to multiple languages for comprehensive search

        Args:
            claim: The claim to translate
            target_languages: List of target language codes ['en', 'zh', 'ja']

        Returns:
            Dictionary with language code as key and translated text as value
        """
        translations = {self._detect_language(claim): claim}  # Original language

        for target_lang in target_languages:
            if target_lang in translations:
                continue

            try:
                # Use LLM to translate the claim
                translation_prompt = {
                    "en": f"Please translate the following text to English, keep the meaning precise: {claim}",
                    "zh": f"请将以下文本翻译成中文，保持意思准确: {claim}",
                    "ja": f"以下のテキストを日本語に翻訳してください、意味を正確に保ってください: {claim}",
                    "ko": f"다음 텍스트를 한국어로 번역해주세요, 의미를 정확하게 유지하세요: {claim}"
                }

                if target_lang not in translation_prompt:
                    continue

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a professional translator. Translate accurately and concisely."},
                        {"role": "user", "content": translation_prompt[target_lang]}
                    ],
                    temperature=0.0,
                    max_tokens=200
                )

                translated_text = response.choices[0].message.content.strip()
                # Clean up any translation artifacts
                if translated_text and not translated_text.startswith("Translation:"):
                    translations[target_lang] = translated_text

            except Exception as e:
                st.warning(f"Translation to {target_lang} failed: {str(e)}")
                continue

        return translations

    def _optimize_language_diversity(self, ranked_chunks: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Optimize evidence selection to ensure language diversity while maintaining relevance.

        Args:
            ranked_chunks: Chunks ranked by similarity score
            top_k: Target number of chunks to return

        Returns:
            Optimized list of chunks with language diversity
        """
        if len(ranked_chunks) <= top_k:
            return ranked_chunks

        # Group chunks by search language (if available)
        language_groups = {}
        no_lang_chunks = []

        for chunk in ranked_chunks:
            # Check if we have search language metadata first
            search_lang = chunk.get('detected_language') or chunk.get('search_language')

            if not search_lang:
                # Fallback to content-based language detection
                text_content = chunk.get('text', '')
                if any('\u4e00' <= char <= '\u9fff' for char in text_content):  # Chinese
                    search_lang = 'zh'
                elif any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text_content):  # Japanese
                    search_lang = 'ja'
                elif text_content and all(ord(char) < 256 for char in text_content if char.isalpha()):  # Likely English
                    search_lang = 'en'

            if search_lang:
                if search_lang not in language_groups:
                    language_groups[search_lang] = []
                language_groups[search_lang].append(chunk)
            else:
                no_lang_chunks.append(chunk)

        # Select diverse evidence - aim for balanced representation
        selected_chunks = []
        remaining_slots = top_k

        # First, select top chunks from each language group
        languages = list(language_groups.keys())
        if languages:
            chunks_per_language = max(1, remaining_slots // len(languages))

            for lang in languages:
                lang_chunks = language_groups[lang][:chunks_per_language]
                selected_chunks.extend(lang_chunks)
                remaining_slots -= len(lang_chunks)

        # Fill remaining slots with highest scoring chunks
        all_remaining = []
        for lang, chunks in language_groups.items():
            chunks_per_language = max(1, top_k // len(languages)) if languages else 0
            all_remaining.extend(chunks[chunks_per_language:])
        all_remaining.extend(no_lang_chunks)

        # Sort remaining by similarity and take what we need
        all_remaining.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        selected_chunks.extend(all_remaining[:remaining_slots])

        # Final sort by similarity to maintain quality
        selected_chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        return selected_chunks[:top_k]

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text using online API.

        Args:
            text: Text to get embedding for

        Returns:
            numpy array of embedding
        """
        try:
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model_name, input=[text]
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            st.error(f"Error getting single embedding: {str(e)}")
            return np.array([])

    def _get_embeddings(self, texts: list) -> np.ndarray:
        """
        Get embeddings for multiple texts using online API.

        Args:
            texts: List of texts to get embeddings for

        Returns:
            numpy array of embeddings
        """
        try:
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model_name, input=texts
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
        except Exception as e:
            st.error(f"Error getting embeddings: {str(e)}")
            st.info(f"Embedding API URL: {self.embedding_base_url}")
            st.info(f"Embedding Model: {self.embedding_model_name}")
            return np.array([])

    def extract_claim(self, text: str) -> str:
        """
        Extract core claims from the input text using LLM.

        Args:
            text: The input text to extract claims from

        Returns:
            extracted claim
        """
        # Get appropriate prompts based on user language setting
        if self.output_language == "auto":
            # Auto-detect based on input text
            detected_lang = self._detect_language(text)
            prompts = self._get_language_prompts(detected_lang)
        else:
            # Use user-configured language directly
            prompts = self._get_language_prompts(self.output_language)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompts["extract_claim"]},
                    {"role": "user", "content": f"{prompts['user_extract']} {text}"},
                ],
                temperature=0.0,  # Use low temperature for consistent claim extraction
                max_tokens=500,
            )

            claims_text = response.choices[0].message.content

            # Parse the numbered list into separate claims
            claims = re.findall(r"\d+\.\s+(.*?)(?=\n\d+\.|\Z)", claims_text, re.DOTALL)

            # Clean up the claims
            claims = [claim.strip() for claim in claims if claim.strip()]

            # If no numbered claims were found, split by newlines
            if not claims and claims_text.strip():
                claims = [
                    line.strip()
                    for line in claims_text.strip().split("\n")
                    if line.strip()
                ]

            # Return the first claim if available, otherwise return the original text
            if claims:
                return claims[0]
            else:
                # Fallback: return the original text or a cleaned version
                return claims_text.strip() if claims_text.strip() else text

        except Exception as e:
            st.error(f"Error extracting claims: {str(e)}")
            return text  # Return original text as fallback

    def search_evidence(self, claim: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Search for evidence using multi-language approach to avoid language bias.

        Args:
            claim: The claim to search for evidence
            num_results: Number of search results to return per language

        Returns:
            List of evidence documents with title, url, and snippet from multiple languages
        """
        # Multi-language search to avoid language bias
        progress_placeholder = st.empty()
        progress_placeholder.info("🌍 执行多语言搜索以获取全面的证据...")

        # Define target languages for search
        search_languages = ['en', 'zh', 'ja']  # English, Chinese, Japanese

        # Translate claim to multiple languages
        translations = self._translate_claim(claim, search_languages)
        progress_placeholder.success(f"✅ 已翻译到 {len(translations)} 种语言进行搜索")

        # Auto-clear after 2 seconds
        import time
        time.sleep(2)
        progress_placeholder.empty()

        all_evidence = []

        # Create a container for search progress that will be cleared
        search_progress = st.empty()

        for lang_code, translated_claim in translations.items():
            try:
                # Show current search progress
                with search_progress.container():
                    st.info(f"🔍 搜索语言: {lang_code} - {translated_claim[:50]}...")

                # Search with translated claim
                if self.search_engine == "searxng":
                    evidence_docs = self._search_with_searxng(translated_claim, num_results)
                elif self.search_engine == "youcom":
                    evidence_docs = self._search_with_youcom(translated_claim, num_results)
                else:
                    evidence_docs = self._search_with_duckduckgo(translated_claim, num_results)

                # Add language metadata to evidence
                for doc in evidence_docs:
                    doc['search_language'] = lang_code
                    doc['search_query'] = translated_claim
                    # Add language identifier to help with diversity optimization
                    doc['detected_language'] = lang_code

                all_evidence.extend(evidence_docs)

                # Update progress
                with search_progress.container():
                    st.success(f"✅ {lang_code}: 找到 {len(evidence_docs)} 条证据")

            except Exception as e:
                with search_progress.container():
                    st.warning(f"⚠️ {lang_code} 搜索失败: {str(e)}")
                continue

        # Remove duplicates based on URL
        seen_urls = set()
        unique_evidence = []
        for doc in all_evidence:
            if doc['url'] not in seen_urls:
                seen_urls.add(doc['url'])
                unique_evidence.append(doc)

        # Clear search progress and show final result
        search_progress.empty()

        # Brief final summary that will be cleared by the calling function
        final_status = st.empty()
        final_status.success(f"🎯 多语言搜索完成，共获得 {len(unique_evidence)} 条独特证据")

        # Auto-clear final status after 2 seconds
        time.sleep(2)
        final_status.empty()

        return unique_evidence

    def _search_with_searxng(
        self, query: str, num_results: int = 5
    ) -> List[Dict[str, str]]:
        """
        Search using SearXNG API.
        """
        try:
            search_url = f"{self.searxng_url}/search"
            params = {"q": query, "format": "json", "categories": "general"}

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            }

            # Get timeout from config
            timeout_setting = self.search_config.get('timeout', 30)
            response = requests.get(
                search_url, params=params, headers=headers, timeout=timeout_setting
            )
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])

            evidence_docs = []
            for result in results[:num_results]:
                evidence_docs.append(
                    {
                        "title": result.get("title", "No title"),
                        "url": result.get("url", "No URL"),
                        "snippet": result.get("content", "No snippet"),
                    }
                )

            return evidence_docs

        except Exception as e:
            st.error(f"SearXNG search failed: {str(e)}")
            return []

    def _search_with_duckduckgo(
        self, query: str, num_results: int = 5
    ) -> List[Dict[str, str]]:
        """
        Search using DuckDuckGo (fallback method).
        """
        # Use proxy from search configuration if available
        proxy_setting = None
        if hasattr(self, 'search_config') and 'proxy' in self.search_config:
            proxy_setting = self.search_config['proxy']

        timeout_setting = self.search_config.get('timeout', 60)
        safesearch = self.search_config.get("safesearch", "off")
        strategies = self._get_duckduckgo_strategies()
        errors = []

        for strategy in strategies:
            backend = strategy["backend"]
            region = strategy["region"]
            try:
                evidence_docs = self._search_with_duckduckgo_once(
                    query=query,
                    num_results=num_results,
                    backend=backend,
                    region=region,
                    proxy_setting=proxy_setting,
                    timeout_setting=timeout_setting,
                    safesearch=safesearch,
                )
                if evidence_docs:
                    return evidence_docs

                errors.append(f"{backend}/{region}: no results")
            except Exception as e:
                errors.append(f"{backend}/{region}: {str(e)}")

        st.warning(f"DuckDuckGo search failed after fallbacks: {'; '.join(errors)}")
        return []

    def _get_duckduckgo_strategies(self) -> List[Dict[str, str]]:
        configured = self.search_config.get("fallback_strategies")
        if isinstance(configured, list):
            strategies = []
            for item in configured:
                if not isinstance(item, dict):
                    continue
                backend = str(item.get("backend") or "").strip()
                region = str(item.get("region") or "").strip()
                if backend and region:
                    strategies.append({"backend": backend, "region": region})
            if strategies:
                return self._dedupe_duckduckgo_strategies(strategies)

        backend = self.search_config.get("backend", "duckduckgo")
        region = self.search_config.get("region", "wt-wt")
        return self._dedupe_duckduckgo_strategies(
            [
                {"backend": backend, "region": region},
                {"backend": "duckduckgo", "region": "cn-zh"},
                {"backend": "auto", "region": "wt-wt"},
                {"backend": "brave", "region": "wt-wt"},
            ]
        )

    def _dedupe_duckduckgo_strategies(
        self, strategies: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        seen = set()
        unique = []
        for strategy in strategies:
            key = (strategy["backend"], strategy["region"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(strategy)
        return unique

    def _search_with_duckduckgo_once(
        self,
        query: str,
        num_results: int,
        backend: str,
        region: str,
        proxy_setting: str,
        timeout_setting: int,
        safesearch: str,
    ) -> List[Dict[str, str]]:
        with DDGS(proxy=proxy_setting, timeout=timeout_setting) as ddgs:
            results = list(
                ddgs.text(
                    query,
                    backend=backend,
                    region=region,
                    safesearch=safesearch,
                    max_results=num_results,
                )
            )

        evidence_docs = []
        for result in results:
            evidence_docs.append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                }
            )

        return evidence_docs

    def _search_with_youcom(
        self, query: str, num_results: int = 5
    ) -> List[Dict[str, str]]:
        """
        Search using the You.com MCP server.

        默认使用免 API Key 的免费档端点（profile=free）；
        配置或环境变量提供 YDC_API_KEY 时切换到已认证端点。
        """
        try:
            # Use proxy from search configuration if available
            proxy_setting = None
            if "proxy" in self.search_config:
                proxy_setting = self.search_config["proxy"]

            api_key = self.search_config.get("api_key") or os.getenv("YDC_API_KEY")
            endpoint = (
                self.search_config.get("base_url")
                or os.getenv("YOUCOM_BASE_URL")
                or (YOUCOM_AUTH_ENDPOINT if api_key else YOUCOM_FREE_ENDPOINT)
            )
            timeout_setting = self.search_config.get("timeout", 30)

            web_results = _youcom_mcp_search(
                query=query,
                count=num_results,
                endpoint=endpoint,
                api_key=api_key,
                timeout=timeout_setting,
                proxy=proxy_setting,
            )

            evidence_docs = []
            for result in web_results:
                url = result.get("url", "")
                if not url:
                    continue
                snippet = result.get("description", "")
                if not snippet and result.get("snippets"):
                    snippet = result["snippets"][0]
                evidence_docs.append(
                    {
                        "title": result.get("title", ""),
                        "url": url,
                        "snippet": snippet,
                    }
                )

            return evidence_docs

        except Exception as e:
            st.warning(f"You.com search failed: {str(e)}")
            return []

    def get_evidence_chunks(
        self,
        evidence_docs: List[Dict[str, str]],
        claim: str,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Extract and rank evidence chunks related to the claim using BGE-M3.

        Args:
            evidence_docs: List of evidence documents
            claim: The claim to match with evidence
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            top_k: Number of top chunks to return

        Returns:
            List of ranked evidence chunks with similarity scores
        """
        if not self.embedding_client:
            return [
                {
                    "text": "Evidence ranking unavailable - Embedding API not available.",
                    "source": "System",
                    "similarity": 0.0,
                }
            ]

        if not evidence_docs:
            return []

        try:
            # Create text chunks from evidence documents
            all_chunks = []

            for doc in evidence_docs:
                # Add title as a separate chunk
                chunk_data = {
                    "text": doc["title"],
                    "source": doc["url"],
                }
                # Preserve language metadata if available
                if 'detected_language' in doc:
                    chunk_data['detected_language'] = doc['detected_language']
                if 'search_language' in doc:
                    chunk_data['search_language'] = doc['search_language']
                all_chunks.append(chunk_data)

                # Process the snippet into overlapping chunks
                snippet = doc["snippet"]
                if len(snippet) <= chunk_size:
                    # If snippet is shorter than chunk_size, use it as is
                    chunk_data = {
                        "text": snippet,
                        "source": doc["url"],
                    }
                    # Preserve language metadata if available
                    if 'detected_language' in doc:
                        chunk_data['detected_language'] = doc['detected_language']
                    if 'search_language' in doc:
                        chunk_data['search_language'] = doc['search_language']
                    all_chunks.append(chunk_data)
                else:
                    # Create overlapping chunks
                    for i in range(0, len(snippet), chunk_size - chunk_overlap):
                        chunk_text = snippet[i : i + chunk_size]
                        if (
                            len(chunk_text) >= chunk_size // 2
                        ):  # Only keep chunks of reasonable size
                            chunk_data = {
                                "text": chunk_text,
                                "source": doc["url"],
                            }
                            # Preserve language metadata if available
                            if 'detected_language' in doc:
                                chunk_data['detected_language'] = doc['detected_language']
                            if 'search_language' in doc:
                                chunk_data['search_language'] = doc['search_language']
                            all_chunks.append(chunk_data)

            if not all_chunks:
                return []

            # Compute embeddings for claim using online API
            claim_embedding = self._get_embedding(claim)

            # Compute embeddings for chunks
            chunk_texts = [chunk["text"] for chunk in all_chunks]
            chunk_embeddings = self._get_embeddings(chunk_texts)

            if claim_embedding.size == 0 or chunk_embeddings.size == 0:
                return []

            # Calculate similarities
            similarities = []
            for i, chunk_embedding in enumerate(chunk_embeddings):
                similarity = np.dot(claim_embedding, chunk_embedding) / (
                    np.linalg.norm(claim_embedding) * np.linalg.norm(chunk_embedding)
                )
                similarities.append(float(similarity))

            # Add similarities to chunks
            for i, similarity in enumerate(similarities):
                all_chunks[i]["similarity"] = similarity

            # Sort chunks by similarity (descending)
            ranked_chunks = sorted(
                all_chunks, key=lambda x: x["similarity"], reverse=True
            )

            # Optimize evidence selection for language diversity
            optimized_chunks = self._optimize_language_diversity(ranked_chunks, top_k)

            # Return optimized chunks
            return optimized_chunks

        except Exception as e:
            st.error(f"Error ranking evidence: {str(e)}")
            return [
                {
                    "text": f"Error ranking evidence: {str(e)}",
                    "source": "System",
                    "similarity": 0.0,
                }
            ]

    def evaluate_claim(
        self, claim: str, evidence_chunks: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Evaluate the truthfulness of a claim based on evidence using LLM.

        Args:
            claim: The claim to evaluate
            evidence_chunks: The evidence chunks to use for evaluation

        Returns:
            Dictionary with verdict and reasoning
        """
        # Get appropriate prompts based on user language setting
        if self.output_language == "auto":
            # Auto-detect based on claim text
            detected_lang = self._detect_language(claim)
            prompts = self._get_language_prompts(detected_lang)
        else:
            # Use user-configured language directly
            prompts = self._get_language_prompts(self.output_language)

        # Check if evidence chunks are available
        if not evidence_chunks:
            st.warning("No evidence found for evaluation. Returning unverifiable verdict.")
            return {
                "verdict": "UNVERIFIABLE",
                "reasoning": "无法找到相关证据进行核查。"
            }

        # Prepare evidence text for the prompt
        evidence_text = "\n\n".join(
            [
                f"EVIDENCE {i+1} (Relevance: {chunk.get('similarity', 0.0):.2f}):\n{chunk['text']}\nSource: {chunk['source']}"
                for i, chunk in enumerate(evidence_chunks)
            ]
        )

        try:
            messages = [
                {"role": "system", "content": prompts["evaluate_claim"]},
                {
                    "role": "user",
                    "content": prompts["user_evaluate"].format(
                        claim=claim, evidence=evidence_text
                    ),
                },
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            result_text = response.choices[0].message.content

            # Clean up Unicode characters that might cause encoding issues
            if result_text:
                # Replace problematic Unicode characters
                result_text = result_text.replace('\u2011', '-')  # Non-breaking hyphen to normal hyphen
                result_text = result_text.replace('\u2013', '-')  # En dash to normal hyphen
                result_text = result_text.replace('\u2014', '-')  # Em dash to normal hyphen
                result_text = result_text.replace('\u2010', '-')  # Hyphen to normal hyphen
                # Remove other potentially problematic characters
                result_text = ''.join(char for char in result_text if ord(char) < 65536)

            # Handle empty response
            if not result_text or result_text.strip() == "":
                st.error("⚠️ 模型返回空响应！")
                st.info("🔧 建议解决方案：")
                st.info("1. 切换到更强的模型（如 gemma-3-270m-it 或 GPT 模型）")
                st.info("2. 检查模型是否正确加载在 LM Studio 中")
                st.info("3. 尝试降低输入文本长度")
                return {
                    "verdict": "UNVERIFIABLE",
                    "reasoning": f"当前模型 '{self.model}' 返回空响应。建议切换到更强的模型（如 gemma-3-270m-it）或检查 LM Studio 模型加载状态。"
                }

            # Extract verdict and reasoning with more flexible patterns
            verdict_match = re.search(
                r"(?:VERDICT|判断|结论)[:：]\s*(TRUE|FALSE|PARTIALLY TRUE|正确|错误|部分正确|无法验证)",
                result_text,
                re.IGNORECASE,
            )

            if verdict_match:
                verdict_raw = verdict_match.group(1).upper()
                # Map Chinese terms to English
                if verdict_raw in ["正确", "TRUE"]:
                    verdict = "TRUE"
                elif verdict_raw in ["错误", "FALSE"]:
                    verdict = "FALSE"
                elif verdict_raw in ["部分正确", "PARTIALLY TRUE"]:
                    verdict = "PARTIALLY TRUE"
                else:
                    verdict = "UNVERIFIABLE"
            else:
                # Try to infer from content if no explicit verdict found
                if "is true" in result_text.lower() or "supported" in result_text.lower():
                    verdict = "TRUE"
                elif "is false" in result_text.lower() or "contradicted" in result_text.lower():
                    verdict = "FALSE"
                else:
                    verdict = "UNVERIFIABLE"

            reasoning_match = re.search(
                r"(?:REASONING|推理过程|推理|分析)[:：]\s*(.*)",
                result_text,
                re.DOTALL | re.IGNORECASE,
            )
            reasoning = (
                reasoning_match.group(1).strip()
                if reasoning_match
                else result_text.strip()
            )

            return {"verdict": verdict, "reasoning": reasoning}

        except Exception as e:
            st.error(f"Error evaluating claim: {str(e)}")
            return {
                "verdict": "ERROR",
                "reasoning": f"An error occurred during evaluation: {str(e)}",
            }

    def check_fact(self, text: str) -> Dict[str, Any]:
        """
        Main function to check the factuality of a statement.

        Args:
            text: The statement to fact-check

        Returns:
            Dictionary with all results of the fact-checking process
        """
        # 1. Extract core claim
        claim = self.extract_claim(text)

        result = {"original_text": text, "claim": claim, "results": []}
        # 2. Search for evidence
        evidence_docs = self.search_evidence(claim)

        # 3. Get relevant evidence chunks
        evidence_chunks = self.get_evidence_chunks(evidence_docs, claim)

        # 4. Evaluate claim based on evidence
        evaluation = self.evaluate_claim(claim, evidence_chunks)

        # Add results for this claim
        result = {
            "claim": claim,
            "evidence_docs": evidence_docs,
            "evidence_chunks": evidence_chunks,
            "verdict": evaluation["verdict"],
            "reasoning": evaluation["reasoning"],
        }

        return result


# Function to be imported in the main Streamlit app
def check_fact(
    claim: str, api_base: str, model: str, temperature: float, max_tokens: int
) -> Dict[str, Any]:
    """
    Public interface for fact checking to be used by the Streamlit app.

    Args:
        claim: The statement to fact-check
        api_base: The base URL for the LLM API
        model: The model to use for fact checking
        temperature: Temperature parameter for LLM
        max_tokens: Maximum tokens for LLM response

    Returns:
        Dictionary with all results of the fact-checking process
    """
    checker = FactChecker(api_base, model, temperature, max_tokens)
    return checker.check_fact(claim)
