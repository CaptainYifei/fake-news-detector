#!/usr/bin/env python3
"""
测试脚本用于验证不同模型平台的配置和调用
"""

import json
import os
import sys
from model_manager import model_manager


def test_llm_provider(provider: str, model: str):
    """测试LLM提供商"""
    print(f"\n测试 LLM Provider: {provider}, Model: {model}")
    print("=" * 50)

    try:
        client = model_manager.get_llm_client(provider)
        print(f"✓ LLM 客户端初始化成功")

        # 测试简单的对话
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Hello! Please respond with 'Test successful'",
                },
            ],
            temperature=0.0,
            max_tokens=50,
        )

        result = response.choices[0].message.content
        print(f"✓ LLM 调用成功: {result}")

    except Exception as e:
        print(f"✗ LLM 测试失败: {e}")


def test_embedding_provider(provider: str):
    """测试嵌入模型提供商"""
    print(f"\n测试 Embedding Provider: {provider}")
    print("=" * 50)

    try:
        model = model_manager.get_embedding_model(provider)
        print(f"✓ 嵌入模型初始化成功")

        # 测试嵌入生成
        test_text = "This is a test sentence for embedding."
        embedding = model.encode(test_text)

        if isinstance(embedding, dict) and "dense_vecs" in embedding:
            embedding = embedding["dense_vecs"]

        print(
            f"✓ 嵌入生成成功，维度: {len(embedding) if hasattr(embedding, '__len__') else 'N/A'}"
        )

    except Exception as e:
        print(f"✗ 嵌入模型测试失败: {e}")


def print_available_providers():
    """打印可用的提供商"""
    print("\n可用的 LLM 提供商:")
    for provider in model_manager.get_available_providers():
        models = model_manager.get_available_models(provider)
        print(f"  - {provider}: {models}")

    print("\n可用的嵌入模型提供商:")
    for provider in model_manager.get_available_embedding_providers():
        print(f"  - {provider}")


def main():
    """主函数"""
    print("模型管理器测试脚本")
    print("=" * 50)

    # 打印配置信息
    print_available_providers()

    # 打印默认配置
    defaults = model_manager.get_default_config()
    print(f"\n默认配置: {defaults}")

    # 测试各个提供商
    test_cases = [
        # LLM 测试
        ("local_api", "qwen2.5-14b-instruct"),
        ("lmstudio", "gemma-3-270m-it"),
        ("ollama", "llama3.1"),
        ("openai", "gpt-3.5-turbo"),
        # 嵌入模型测试
        ("bge_m3_local", None),
        ("lmstudio_embeddings", None),
        ("openai_embeddings", None),
        ("sentence_transformers", None),
    ]

    for provider, model in test_cases:
        if model:
            test_llm_provider(provider, model)
        else:
            test_embedding_provider(provider)

    print("\n测试完成!")


if __name__ == "__main__":
    main()
