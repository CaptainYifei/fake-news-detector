"""Optional anonymous Parallel Search MCP adapter for fact-check evidence."""

import math
from datetime import timedelta

import anyio
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "fake-news-detector/2.0 (+https://github.com/CaptainYifei/fake-news-detector)"


def search_parallel(query, num_results, *, timeout=30, session_id):
    """Search from a synchronous worker; return the existing evidence format.

    Only excerpts are used. The normal embedding ranker and evidence gates still
    decide which sources reach verification. No fallback changes the provider.
    """
    if num_results <= 0 or not query.strip():
        return []
    timeout = float(timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Parallel timeout must be a positive finite number")
    return anyio.run(_search, query, num_results, timeout, session_id)


async def _search(query, num_results, timeout, session_id):
    # Bound initialization, tool execution and transport cleanup together.
    with anyio.fail_after(timeout):
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT}, timeout=timeout,
        ) as client:
            async with streamable_http_client(MCP_URL, http_client=client) as streams:
                async with ClientSession(
                    streams[0], streams[1],
                    read_timeout_seconds=timedelta(seconds=timeout),
                ) as session:
                    await session.initialize()
                    response = await session.call_tool("web_search", {
                        "objective": f"Find web evidence to verify this claim: {query}",
                        "search_queries": [query],
                        "session_id": session_id,
                    })
                    if response.isError:
                        raise RuntimeError("Parallel search tool returned an error")
                    data = response.structuredContent
                    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
                        raise ValueError("Parallel search response has no structured results")
                    documents = []
                    for item in data["results"]:
                        if not isinstance(item, dict):
                            continue
                        url = item.get("url")
                        excerpts = item.get("excerpts")
                        if (not isinstance(url, str) or not url.startswith(("http://", "https://"))
                                or not isinstance(excerpts, list)):
                            continue
                        snippet = "\n".join(text for text in excerpts if isinstance(text, str))
                        if not snippet.strip():
                            continue
                        title = item.get("title")
                        documents.append({
                            "title": title if isinstance(title, str) and title else url,
                            "url": url,
                            "snippet": snippet,
                        })
                        if len(documents) >= num_results:
                            break
                    return documents
