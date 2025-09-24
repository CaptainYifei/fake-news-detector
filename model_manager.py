import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple
import streamlit as st
from openai import OpenAI
import requests
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import SentenceTransformer


class ModelManager:
    def __init__(self, config_path: str = "model_config.json"):
        """
        Initialize the model manager with configuration.

        Args:
            config_path: Path to the model configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.llm_clients = {}
        self.embedding_models = {}

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            st.error(f"Configuration file {self.config_path} not found")
            return {}
        except json.JSONDecodeError as e:
            st.error(f"Error parsing configuration file: {e}")
            return {}

    def get_llm_client(self, provider: str = None) -> OpenAI:
        """
        Get LLM client for specified provider.

        Args:
            provider: Provider name (uses default if not specified)

        Returns:
            OpenAI client instance
        """
        if provider is None:
            provider = self.config.get("defaults", {}).get("llm_provider", "local_api")

        if provider in self.llm_clients:
            return self.llm_clients[provider]

        provider_config = self.config.get("providers", {}).get(provider, {})
        if not provider_config:
            raise ValueError(f"Provider {provider} not found in configuration")

        base_url = os.getenv(
            f"{provider.upper()}_BASE_URL", provider_config.get("base_url")
        )
        api_key = os.getenv(
            f"{provider.upper()}_API_KEY", provider_config.get("api_key", "EMPTY")
        )

        if provider_config["type"] == "ollama":
            # Ollama uses a different API structure
            client = OllamaClient(base_url, api_key)
        else:
            # OpenAI compatible APIs
            client = OpenAI(api_key=api_key, base_url=base_url)

        self.llm_clients[provider] = client
        return client

    def get_embedding_model(self, provider: str = None):
        """
        Get embedding model for specified provider.

        Args:
            provider: Provider name (uses default if not specified)

        Returns:
            Embedding model instance
        """
        if provider is None:
            provider = self.config.get("defaults", {}).get(
                "embedding_provider", "bge_m3_local"
            )

        if provider in self.embedding_models:
            return self.embedding_models[provider]

        provider_config = self.config.get("embedding_providers", {}).get(provider, {})
        if not provider_config:
            raise ValueError(
                f"Embedding provider {provider} not found in configuration"
            )

        try:
            if provider_config["type"] == "local":
                model = BGEM3FlagModel(
                    provider_config["model_path"],
                    use_fp16=provider_config.get("use_fp16", True),
                    device=provider_config.get("device", "cuda"),
                )
            elif provider_config["type"] == "sentence_transformers":
                model = SentenceTransformer(
                    provider_config["model_name"],
                    device=provider_config.get("device", "cpu"),
                )
            elif provider_config["type"] == "api":
                model = APIEmbeddingClient(
                    provider_config["base_url"],
                    provider_config.get("api_key", "EMPTY"),
                    provider_config.get("model", "bge-m3"),
                )
            elif provider_config["type"] == "openai_compatible":
                model = OpenAIEmbeddingClient(
                    provider_config["base_url"],
                    provider_config.get("api_key", "EMPTY"),
                    provider_config.get("model", "text-embedding-3-small"),
                )
            else:
                raise ValueError(
                    f"Unsupported embedding provider type: {provider_config['type']}"
                )

            self.embedding_models[provider] = model
            return model

        except Exception as e:
            st.error(f"Error loading embedding model {provider}: {e}")
            return None

    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers."""
        return list(self.config.get("providers", {}).keys())

    def get_available_models(self, provider: str) -> List[str]:
        """Get list of available models for a provider."""
        return list(
            self.config.get("providers", {}).get(provider, {}).get("models", {}).keys()
        )

    def get_models_from_api(
        self, provider: str, base_url: str, api_key: str = "EMPTY", timeout: int = 5
    ) -> List[str]:
        """
        从API端点动态获取模型列表
        支持OpenAI兼容的/models接口
        """
        try:
            # 确保URL格式正确
            if not base_url.endswith("/"):
                base_url += "/"
            if base_url.endswith("/v1/"):
                models_url = base_url + "models"
            elif base_url.endswith("/v1"):
                models_url = base_url + "/models"
            else:
                models_url = base_url + "models"

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            response = requests.get(models_url, headers=headers, timeout=timeout)

            if response.status_code == 200:
                data = response.json()
                # OpenAI格式: {"data": [{"id": "model_name"}, ...]}
                if "data" in data:
                    return [model["id"] for model in data["data"]]
                # 简单格式: ["model1", "model2", ...]
                elif isinstance(data, list):
                    return data
                # Ollama格式: {"models": [{"name": "model_name"}, ...]}
                elif "models" in data:
                    return [model["name"] for model in data["models"]]
                else:
                    return []
            else:
                st.warning(
                    f"API请求失败 (状态码: {response.status_code}): {models_url}"
                )
                return []

        except requests.exceptions.Timeout:
            st.warning(f"API请求超时: {base_url}")
            return []
        except requests.exceptions.ConnectionError:
            st.warning(f"无法连接到API: {base_url}")
            return []
        except Exception as e:
            st.warning(f"获取模型列表失败: {str(e)}")
            return []

    def get_dynamic_models(
        self, provider: str, custom_base_url: Optional[str] = None
    ) -> List[str]:
        """
        获取指定提供商的可用模型列表
        优先从API动态获取，失败则使用配置文件中的静态列表
        """
        provider_config = self.config.get("providers", {}).get(provider, {})
        if not provider_config:
            return []

        # 使用自定义URL或配置中的URL
        base_url = custom_base_url or provider_config.get("base_url", "")
        api_key = provider_config.get("api_key", "EMPTY")

        # 处理环境变量
        if api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.getenv(env_var, "EMPTY")

        # 尝试从API获取模型列表
        api_models = self.get_models_from_api(provider, base_url, api_key)

        if api_models:
            return api_models

        # API获取失败，使用配置文件中的静态模型列表
        static_models = self.get_available_models(provider)
        if static_models:
            st.info(f"使用配置文件中的静态模型列表 (共{len(static_models)}个模型)")
            return static_models

        return []

    def create_model_selection_ui(
        self,
    ) -> Tuple[str, str, str, str, str, str, Dict[str, Any]]:
        """
        创建统一的模型选择界面
        返回: (provider_key, base_url, chat_model, embedding_model, search_provider, selected_language, provider_config)
        """
        providers = self.get_available_providers()
        provider_names = {}

        # 创建提供商名称映射
        for key in providers:
            provider_config = self.config.get("providers", {}).get(key, {})
            provider_names[key] = provider_config.get("name", key)

        # 第一级：选择提供商
        if not providers:
            st.error("未找到可用的模型提供商")
            return "", "", "", "", "", "", {}

        # 使用默认提供商
        default_provider = self.config.get("defaults", {}).get(
            "llm_provider", providers[0]
        )
        if default_provider not in providers:
            default_provider = providers[0]

        selected_provider_key = st.selectbox(
            "选择模型提供商",
            options=providers,
            format_func=lambda x: provider_names.get(x, x),
            index=(
                providers.index(default_provider)
                if default_provider in providers
                else 0
            ),
            help="选择要使用的模型提供商",
        )

        provider_config = self.config.get("providers", {}).get(
            selected_provider_key, {}
        )

        # 第二级：API端点配置
        default_base_url = provider_config.get("base_url", "http://localhost:8000/v1")

        # 处理环境变量
        if default_base_url.startswith("${") and default_base_url.endswith("}"):
            env_var = default_base_url[2:-1]
            default_base_url = os.getenv(env_var, "http://localhost:8000/v1")

        base_url = st.text_input(
            "API基础URL",
            value=default_base_url,
            help="模型API的基础URL地址",
            key=f"base_url_{selected_provider_key}",
        )


        # 第三级：动态获取并选择具体模型
        if st.button("🔄 刷新", help="重新从API获取最新的模型列表"):
            st.rerun()

        available_models = self.get_dynamic_models(selected_provider_key, base_url)

        if not available_models:
            st.warning("未找到可用的模型，请检查API连接")
            return selected_provider_key, base_url, "", "", "", "auto", provider_config

        # 分类模型：聊天模型和嵌入模型
        chat_models = []
        embedding_models = []

        for model in available_models:
            model_info = provider_config.get("models", {}).get(model, {})
            model_type = model_info.get("type", "chat")

            if model_type == "embedding":
                embedding_models.append(model)
            else:
                chat_models.append(model)

        # 如果没有分类信息，根据模型名称推断
        if not embedding_models and not chat_models:
            for model in available_models:
                if ("embed" in model.lower() or
                    "embedding" in model.lower() or
                    "nomic" in model.lower()):  # 特别处理 nomic-embed-text
                    embedding_models.append(model)
                else:
                    chat_models.append(model)

        # 选择聊天模型和嵌入模型
        col_chat, col_embed = st.columns(2)

        with col_chat:
            st.subheader("🤖 聊天模型")
            if not chat_models:
                st.warning("未找到聊天模型")
                selected_chat_model = ""
            else:
                default_chat = self.config.get("defaults", {}).get("llm_model", "")
                chat_index = 0
                if default_chat in chat_models:
                    chat_index = chat_models.index(default_chat)

                selected_chat_model = st.selectbox(
                    "选择聊天模型",
                    options=chat_models,
                    index=chat_index,
                    help=f"共{len(chat_models)}个聊天模型可用",
                    key=f"chat_{selected_provider_key}",
                )

        with col_embed:
            st.subheader("🧠 嵌入模型")
            if not embedding_models:
                st.warning("未找到嵌入模型")
                selected_embedding_model = ""
            else:
                default_embed = self.config.get("defaults", {}).get(
                    "embedding_model", ""
                )
                embed_index = 0
                if default_embed in embedding_models:
                    embed_index = embedding_models.index(default_embed)

                selected_embedding_model = st.selectbox(
                    "选择嵌入模型",
                    options=embedding_models,
                    index=embed_index,
                    help=f"共{len(embedding_models)}个嵌入模型可用",
                    key=f"embed_{selected_provider_key}",
                )

        # 显示选择结果
        if selected_chat_model or selected_embedding_model:
            status_parts = []
            if selected_chat_model:
                status_parts.append(f"🤖 {selected_chat_model}")
            if selected_embedding_model:
                status_parts.append(f"🧠 {selected_embedding_model}")
            st.success(
                f"✅ {provider_names.get(selected_provider_key, selected_provider_key)}: {' | '.join(status_parts)}"
            )

        st.divider()
        st.subheader("🔍 搜索引擎配置")

        # 搜索引擎选择
        search_providers = list(self.config.get("search_providers", {}).keys())
        if not search_providers:
            st.error("未配置搜索引擎")
            selected_search_provider = ""
        else:
            search_names = {}
            for key in search_providers:
                search_config = self.config.get("search_providers", {}).get(key, {})
                search_names[key] = search_config.get("name", key)

            default_search = self.config.get("defaults", {}).get(
                "search_provider", search_providers[0]
            )
            if default_search not in search_providers:
                default_search = search_providers[0]

            selected_search_provider = st.selectbox(
                "选择搜索引擎",
                options=search_providers,
                format_func=lambda x: search_names.get(x, x),
                index=(
                    search_providers.index(default_search)
                    if default_search in search_providers
                    else 0
                ),
                help="选择用于获取证据的搜索引擎",
            )

            # 显示搜索引擎信息
            if selected_search_provider:
                search_config = self.config.get("search_providers", {}).get(
                    selected_search_provider, {}
                )
                if search_config.get("type") == "searxng":
                    searxng_url = st.text_input(
                        "SearXNG API URL",
                        value=search_config.get("base_url", "http://localhost:8090"),
                        help="SearXNG实例的API地址",
                        key="searxng_url",
                    )
                    st.info("💡 提示：确保SearXNG实例正在运行且启用了JSON API")
                elif search_config.get("type") == "duckduckgo":
                    st.warning("⚠️ DuckDuckGo可能需要代理设置才能正常使用")

                st.success(
                    f"🔍 搜索引擎: {search_names.get(selected_search_provider, selected_search_provider)}"
                )

        st.divider()
        st.subheader("🌐 语言配置")

        # 语言选择
        languages = self.config.get("languages", {})
        if languages:
            language_names = {key: lang["name"] for key, lang in languages.items()}
            default_lang = self.config.get("defaults", {}).get(
                "output_language", "auto"
            )

            if default_lang not in languages:
                default_lang = "auto"

            selected_language = st.selectbox(
                "输出语言",
                options=list(languages.keys()),
                format_func=lambda x: language_names.get(x, x),
                index=(
                    list(languages.keys()).index(default_lang)
                    if default_lang in languages
                    else 0
                ),
                help="选择AI回复的语言（自动检测将根据输入文本语言决定输出语言）",
            )

            if selected_language == "auto":
                st.info("💡 自动检测模式：AI将根据你输入的文本语言来决定回复语言")
            else:
                lang_info = languages[selected_language]
                st.success(f"🌐 输出语言: {lang_info['name']}")
        else:
            selected_language = "auto"

        return (
            selected_provider_key,
            base_url,
            selected_chat_model,
            selected_embedding_model,
            selected_search_provider,
            selected_language,
            provider_config,
        )

    def test_connection(self, base_url: str, api_key: str = "EMPTY") -> bool:
        """测试与API的连接"""
        try:
            models = self.get_models_from_api("test", base_url, api_key, timeout=3)
            return len(models) > 0
        except:
            return False

    def get_available_embedding_providers(self) -> List[str]:
        """Get list of available embedding providers."""
        return self.get_available_providers()  # 现在统一使用providers

    def get_search_providers(self) -> List[str]:
        """Get list of available search providers."""
        return list(self.config.get("search_providers", {}).keys())

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return self.config.get("defaults", {})

    def update_config(self, updates: Dict[str, Any]):
        """Update configuration and save to file."""
        self.config.update(updates)
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"Error saving configuration: {e}")


class OllamaClient:
    """Client for Ollama API."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/v1")
        self.api_key = api_key

    def chat_completions_create(
        self,
        model: str,
        messages: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 2000,
        **kwargs,
    ):
        """Create chat completion using Ollama API."""
        url = f"{self.base_url}/api/chat"

        # Convert OpenAI message format to Ollama format
        system_message = ""
        user_message = ""
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            elif msg["role"] == "user":
                user_message = msg["content"]

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        try:
            response = requests.post(
                url, json=payload, headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()

            result = response.json()
            # Convert Ollama response to OpenAI-like format
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message",
                                    (),
                                    {
                                        "content": result.get("message", {}).get(
                                            "content", ""
                                        )
                                    },
                                )()
                            },
                        )()
                    ]
                },
            )()
        except Exception as e:
            raise Exception(f"Ollama API error: {e}")


class APIEmbeddingClient:
    """Client for API-based embedding models."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def encode(self, texts: Union[str, List[str]]):
        """Encode texts using API-based embedding model."""
        if isinstance(texts, str):
            texts = [texts]

        payload = {"model": self.model, "input": texts}

        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()

            result = response.json()
            embeddings = [item["embedding"] for item in result["data"]]

            return {
                "dense_vecs": np.array(
                    embeddings[0] if len(embeddings) == 1 else embeddings
                )
            }
        except Exception as e:
            raise Exception(f"API embedding error: {e}")


class OpenAIEmbeddingClient:
    """Client for OpenAI-compatible embedding APIs."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.dimensions = None

    def encode(self, texts: Union[str, List[str]]):
        """Encode texts using OpenAI-compatible embedding API."""
        try:
            if isinstance(texts, str):
                texts = [texts]

            response = self.client.embeddings.create(model=self.model, input=texts)

            embeddings = [item.embedding for item in response.data]

            return {
                "dense_vecs": np.array(
                    embeddings[0] if len(embeddings) == 1 else embeddings
                )
            }
        except Exception as e:
            raise Exception(f"OpenAI embedding error: {e}")


# Global model manager instance
model_manager = ModelManager()
