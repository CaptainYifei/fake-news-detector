"""Exercise the configured search caller over the MCP HTTP transport."""
import asyncio
import json
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import anyio
import httpx
import numpy as np
from fastapi.testclient import TestClient

import api
import parallel_search
from fact_checker import FactChecker
from fact_checking.schemas import ClaimExtractionResult
from model_manager import ModelManager


RESULT = {
    "search_id": "search_fixture",
    "session_id": "fixture_session",
    "results": [{
        "title": "NASA Apollo 11",
        "url": "https://www.nasa.gov/apollo11/",
        "excerpts": ["NASA launched Apollo 11 on July 16, 1969.", "It landed on July 20."],
    }],
}


class ParallelSearchTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.tool_result = {"content": [], "structuredContent": RESULT, "isError": False}
        self.mode = "normal"
        self.closed = False

    async def respond(self, request):
        self.assertEqual(str(request.url), parallel_search.MCP_URL)
        self.assertEqual(request.headers["User-Agent"], parallel_search.USER_AGENT)
        self.assertNotIn("Authorization", request.headers)
        if request.method != "POST":
            return httpx.Response(405)
        data = json.loads(request.content)
        self.requests.append(data)
        if data["method"] == "notifications/initialized":
            return httpx.Response(202)
        if data["method"] == "initialize":
            result = {"protocolVersion": data["params"]["protocolVersion"],
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fixture", "version": "1"}}
        elif data["method"] == "tools/list":
            result = {"tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]}
        else:
            self.assertEqual(data["method"], "tools/call")
            self.assertEqual(data["params"]["name"], "web_search")
            if self.mode == "timeout":
                raise httpx.ReadTimeout("fixture timeout", request=request)
            if self.mode in ("slow", "cancel"):
                try:
                    await anyio.sleep_forever()
                finally:
                    self.closed = True
            result = self.tool_result
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": data["id"], "result": result})

    @contextmanager
    def transport(self):
        real_client = httpx.AsyncClient
        def create(**kwargs):
            self.assertEqual(kwargs["headers"]["User-Agent"], parallel_search.USER_AGENT)
            self.assertGreater(kwargs["timeout"], 0)
            return real_client(**kwargs, transport=httpx.MockTransport(self.respond))
        with patch("parallel_search.httpx.AsyncClient", side_effect=create):
            yield

    def checker(self):
        # Load the maintained configuration path, including saved user overrides.
        with patch("model_manager.get_user_config_manager", return_value=None):
            manager = ModelManager()
            config = manager.get_search_provider_config("parallel")
        return FactChecker("http://localhost:11434/v1", "qwen2.5:3b", 0, 1000,
                           search_engine="parallel", search_config=config)

    def evidence(self, checker):
        node = checker._build_pipeline().evidence_node
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='[{"query":"NASA Apollo 11 launch date","language":"en"}]'))])
        with patch.object(checker.client.chat.completions, "create", return_value=completion), \
             patch.object(checker, "_get_embedding", return_value=np.array([1., 0.])), \
             patch.object(checker, "_get_embeddings", side_effect=lambda texts: np.array([[1., 0.]] * len(texts))):
            return node.run(ClaimExtractionResult("Apollo launch", "NASA launched Apollo 11", entities=["NASA"]),
                            num_results=1, top_k=2)

    def test_configured_caller_and_downstream_evidence(self):
        checker = self.checker()
        with self.transport():
            result = self.evidence(checker)
            again = checker._search_by_config("NASA Apollo 11 landing date", 1)
        self.assertEqual(result.documents[0]["url"], RESULT["results"][0]["url"])
        self.assertIn("July 16", result.documents[0]["snippet"])
        self.assertEqual(result.filter_summary["selected_count"], 2)
        self.assertEqual(result.evidence_chunks[0]["source"], RESULT["results"][0]["url"])
        self.assertEqual(len(again), 1)
        calls = [r for r in self.requests if r["method"] == "tools/call"]
        self.assertEqual(calls[0]["params"]["arguments"]["search_queries"], ["NASA Apollo 11 launch date"])
        self.assertIn("verify this claim", calls[0]["params"]["arguments"]["objective"])
        self.assertEqual(calls[0]["params"]["arguments"]["session_id"], checker.parallel_session_id)
        self.assertEqual(calls[1]["params"]["arguments"]["session_id"], checker.parallel_session_id)

    def test_filter_invalid_results_and_limit(self):
        self.tool_result["structuredContent"] = dict(RESULT, results=[
            {"url": "file:///tmp/local", "excerpts": ["not web evidence"]},
            {"url": "https://example.org/empty", "excerpts": []},
            {"url": "https://example.org/valid", "title": None, "excerpts": ["Evidence", None]},
            RESULT["results"][0],
        ])
        with self.transport():
            docs = self.checker()._search_by_config("claim", 1)
        self.assertEqual(docs, [{"title": "https://example.org/valid", "url": "https://example.org/valid", "snippet": "Evidence"}])

    def test_errors_do_not_switch_provider_or_create_evidence(self):
        for tool_result in [
            {"content": [{"type": "text", "text": "quota exceeded"}], "isError": True},
            {"content": [], "structuredContent": {"unexpected": []}},
            {"content": [], "structuredContent": dict(RESULT, results=[])},
        ]:
            with self.subTest(tool_result=tool_result):
                self.tool_result = tool_result
                checker = self.checker()
                with self.transport(), patch.object(checker, "_search_with_duckduckgo") as fallback:
                    result = self.evidence(checker)
                self.assertEqual(result.documents, [])
                self.assertEqual(result.evidence_chunks, [])
                fallback.assert_not_called()

    def test_timeout_and_total_deadline(self):
        for mode in ("timeout", "slow"):
            with self.subTest(mode=mode):
                self.mode = mode
                checker = self.checker()
                checker.search_config["timeout"] = 0.2
                with self.transport(), patch("fact_checker.st.error") as error:
                    self.assertEqual(checker._search_by_config("claim", 1), [])
                error.assert_called_once()
        self.assertTrue(self.closed)

    def test_cancellation_propagates_and_closes_transport(self):
        self.mode = "cancel"
        async def run():
            with anyio.move_on_after(0.2) as scope:
                await parallel_search._search("claim", 1, 30, "stable_session")
            self.assertTrue(scope.cancel_called)
        with self.transport():
            anyio.run(run)
        self.assertTrue(self.closed)

    def test_empty_request_and_invalid_timeout(self):
        with self.transport():
            self.assertEqual(parallel_search.search_parallel("", 5, session_id="stable"), [])
            self.assertEqual(parallel_search.search_parallel("claim", 0, session_id="stable"), [])
            for timeout in (0, -1, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    parallel_search.search_parallel("claim", 1, timeout=timeout, session_id="stable")
        self.assertEqual(self.requests, [])

    def test_existing_defaults_saved_choices_and_optional_import(self):
        user = SimpleNamespace(get_user_config=lambda: {"default_config": {"defaults": {"search_provider": "searxng"}},
                                                       "search_config": {"search_providers": {"parallel": {"timeout": 12}}}})
        with patch("model_manager.get_user_config_manager", return_value=user):
            manager = ModelManager()
            self.assertEqual(manager.get_current_config()["defaults"]["search_provider"], "searxng")
            self.assertEqual(manager.get_search_provider_config("parallel")["timeout"], 12)
        with open("model_config.json") as config:
            self.assertEqual(json.load(config)["defaults"]["search_provider"], "duckduckgo")
        checker = self.checker()
        for provider, method in [("duckduckgo", "_search_with_duckduckgo"), ("searxng", "_search_with_searxng")]:
            checker.search_engine = provider
            with patch.object(checker, method, return_value=[]) as caller:
                checker._search_by_config("claim", 3)
                caller.assert_called_once_with("claim", 3)
        checker.search_engine = "parallel"
        with patch.dict("sys.modules", {"parallel_search": None}), patch("fact_checker.st.error") as error:
            self.assertEqual(checker._search_by_config("claim", 3), [])
        self.assertIn("requirements-parallel.txt", error.call_args.args[0])
        self.assertEqual(api.FactCheckRequest(text="claim").search_engine, "searxng")

    def test_api_parallel_search_runs_outside_asgi_loop(self):
        real_checker = FactChecker
        def create(*args, **kwargs):
            checker = real_checker(*args, **kwargs)
            def check_news(text):
                with self.assertRaises(RuntimeError):
                    asyncio.get_running_loop()
                evidence = self.evidence(checker)
                return SimpleNamespace(claim=text, verdict="UNVERIFIABLE", reasoning="Fixture",
                                       evidence_chunks=evidence.evidence_chunks, to_dict=evidence.to_dict)
            checker.check_news = check_news
            return checker
        with self.transport(), patch("api.FactChecker", side_effect=create), TestClient(api.app) as client:
            response = client.post("/check", json={"text": "NASA launched Apollo 11", "search_engine": "parallel"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["evidence"])


if __name__ == "__main__":
    unittest.main()
