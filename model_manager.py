import json
import os
import re
from typing import Dict, List, Any, Optional, Union, Tuple
import streamlit as st
from openai import OpenAI
import requests
import numpy as np
from user_config import get_user_config_manager


class ModelManager:
    def __init__(self, config_path: str = "model_config.json"):
        """
        Initialize the model manager with configuration.

        Args:
            config_path: Path to the model configuration file
        """
        self.config_path = config_path
        self.base_config = self._load_config()
        self.config = self._apply_user_config(self.base_config)
        self.llm_clients = {}
        self.embedding_models = {}

    def _substitute_env_vars(self, obj):
        """Recursively substitute environment variables in configuration object."""
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str) and "${" in obj and "}" in obj:
            # Match ${ENV_VAR:-default} or ${ENV_VAR} pattern
            pattern = r"\$\{([^}]+)\}"

            def replace_match(match):
                env_expr = match.group(1)
                # Check if there's a default value
                if ":-" in env_expr:
                    env_var, default_val = env_expr.split(":-", 1)
                    return os.getenv(env_var, default_val)
                else:
                    return os.getenv(
                        env_expr, match.group(0)
                    )  # Return original if not found

            return re.sub(pattern, replace_match, obj)
        else:
            return obj

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file with environment variable substitution."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            # Apply environment variable substitution
            return self._substitute_env_vars(config)
        except FileNotFoundError:
            st.error(f"Configuration file {self.config_path} not found")
            return {}
        except json.JSONDecodeError as e:
            st.error(f"Error parsing configuration file: {e}")
            return {}

    def _apply_user_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用用户配置到基础配置

        Args:
            base_config: 基础配置

        Returns:
            应用用户配置后的配置
        """
        user_config_manager = get_user_config_manager()
        if not user_config_manager:
            return base_config

        user_config = user_config_manager.get_user_config()
        if not user_config:
            return base_config

        # 深度合并配置
        merged_config = base_config.copy()

        # 应用用户模型配置
        if "model_config" in user_config:
            self._merge_config(merged_config, user_config["model_config"])

        # 应用用户搜索配置
        if "search_config" in user_config:
            self._merge_config(merged_config, user_config["search_config"])

        # 应用用户默认配置
        if "default_config" in user_config:
            self._merge_config(merged_config, user_config["default_config"])

        return merged_config

    def _merge_config(self, base_config: Dict[str, Any], user_config: Dict[str, Any]):
        """
        递归合并用户配置到基础配置

        Args:
            base_config: 基础配置（会被修改）
            user_config: 用户配置
        """
        for key, value in user_config.items():
            if (
                key in base_config
                and isinstance(base_config[key], dict)
                and isinstance(value, dict)
            ):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value

    def save_user_model_config(self, model_config: Dict[str, Any]):
        """
        保存用户模型配置

        Args:
            model_config: 模型配置
        """
        user_config_manager = get_user_config_manager()
        if user_config_manager:
            user_config_manager.save_model_config(model_config)
            # 重新加载配置
            self.config = self._apply_user_config(self.base_config)

    def save_user_search_config(self, search_config: Dict[str, Any]):
        """
        保存用户搜索配置

        Args:
            search_config: 搜索配置
        """
        user_config_manager = get_user_config_manager()
        if user_config_manager:
            user_config_manager.save_search_config(search_config)
            # 重新加载配置
            self.config = self._apply_user_config(self.base_config)

    def save_user_defaults(self, defaults: Dict[str, Any]):
        """
        保存用户默认配置

        Args:
            defaults: 默认配置
        """
        user_config_manager = get_user_config_manager()
        if user_config_manager:
            user_config_manager.save_default_config(defaults)
            # 重新加载配置
            self.config = self._apply_user_config(self.base_config)

    def reset_user_config(self):
        """重置用户配置"""
        user_config_manager = get_user_config_manager()
        if user_config_manager:
            user_config_manager.reset_config()
            # 重新加载配置
            self.config = self._apply_user_config(self.base_config)

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
            if provider_config["type"] == "api":
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

    # 注意：create_model_selection_ui 方法已移除
    # 现在使用 app.py 中的启动时配置向导替代

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
        return self.base_config

    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration with user overrides applied."""
        # Refresh the configuration to ensure latest user settings are applied
        self.config = self._apply_user_config(self.base_config)
        return self.config

    def get_search_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """Get configuration for a specific search provider with user overrides applied."""
        current_config = self.get_current_config()
        return current_config.get("search_providers", {}).get(provider_name, {})

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
