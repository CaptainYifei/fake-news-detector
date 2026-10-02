import streamlit as st
import os
from datetime import datetime
import time
import base64
from fact_checker import FactChecker
import auth
import db_utils
from pdf_export import generate_fact_check_pdf
from model_manager import model_manager
from fact_checking.llm_utils import looks_like_json

from reportlab.pdfgen import canvas
from io import BytesIO


def generate_test_pdf():
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    c.drawString(100, 750, "这是一个测试PDF")
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# 初始化数据库
db_utils.init_db()

# 页面配置
st.set_page_config(
    page_title="AI虚假新闻检测器",
    page_icon="🔍",
    layout="wide",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)


def check_user_config_status():
    """检查用户配置状态，判断是否需要显示配置向导"""
    from user_config import get_user_config_manager
    
    config_manager = get_user_config_manager()
    if not config_manager:
        return False  # 未登录，不需要检查配置
    
    user_config = config_manager.get_user_config()
    
    # 检查是否有基本配置
    has_model_config = bool(user_config.get("model_config", {}))
    has_working_config = "config_completed" in user_config
    
    return has_model_config and has_working_config

def show_initial_config_wizard():
    """显示初始配置向导"""
    st.title("🚀 欢迎使用AI虚假新闻检测器")
    st.markdown("""
    在开始使用前，请先进行一次性配置。
    配置完成后，您就可以直接使用系统了。
    """)
    
    st.divider()
    
    # 自动检测配置
    st.subheader("🔍 步骤1: 检测本地环境")
    
    auto_config = detect_available_services()
    if auto_config:
        st.success(f"✅ 检测到可用服务: {auto_config['name']}")
        st.info(f"📍 服务地址: {auto_config['url']}")
        st.info(f"🤖 可用模型: {len(auto_config['available_models'])}个")
        
        # 显示模型选择
        st.subheader("🤖 选择模型")
        
        # 分类模型
        chat_models, embedding_models = categorize_models(auto_config['available_models'])
        
        col1, col2 = st.columns(2)
        with col1:
            if chat_models:
                selected_chat_model = st.selectbox(
                    "💬 聊天模型",
                    options=chat_models,
                    help=f"共{len(chat_models)}个聊天模型可用"
                )
            else:
                st.warning("未找到聊天模型")
                selected_chat_model = None
        
        with col2:
            if embedding_models:
                selected_embedding_model = st.selectbox(
                    "🧠 嵌入模型",
                    options=embedding_models,
                    help=f"共{len(embedding_models)}个嵌入模型可用"
                )
            else:
                # 如果没有嵌入模型，从聊天模型中选择一个
                if chat_models:
                    selected_embedding_model = st.selectbox(
                        "🧠 嵌入模型",
                        options=chat_models,
                        help="未找到专用嵌入模型，使用聊天模型代替"
                    )
                else:
                    st.warning("未找到可用模型")
                    selected_embedding_model = None
        
        if selected_chat_model and selected_embedding_model:
            # 添加搜索引擎选择
            st.subheader("🔍 选择搜索引擎")
            search_options = {
                "🦆 DuckDuckGo (默认推荐)": "duckduckgo",
                "🔍 SearXNG (本地备用)": "searxng",
                "🌐 You.com (免密钥)": "youcom"
            }
            
            selected_search = st.radio(
                "搜索引擎",
                options=list(search_options.keys()),
                help="默认使用 DuckDuckGo；SearXNG 可作为本地备用搜索；You.com 为免密钥远程搜索",
                horizontal=True
            )
            
            search_provider = search_options[selected_search]
            searxng_url = None
            
            # 如果选择了 SearXNG，让用户配置地址
            if search_provider == "searxng":
                searxng_url = st.text_input(
                    "🌐 SearXNG 服务地址",
                    value="http://localhost:8090",
                    help="请输入您的 SearXNG 实例地址",
                    placeholder="http://localhost:8090"
                )
                
                if searxng_url:
                    # 测试 SearXNG 连接
                    searxng_available = test_searxng_connection(searxng_url)
                    if searxng_available:
                        st.success("✅ SearXNG 服务可用")
                    else:
                        st.warning("⚠️ SearXNG 服务不可用，请检查地址或服务状态")
            
            if st.button("✨ 使用此配置", type="primary", use_container_width=True):
                auto_config['chat_model'] = selected_chat_model
                auto_config['embedding_model'] = selected_embedding_model
                auto_config['search_provider'] = search_provider
                if searxng_url:
                    auto_config['searxng_url'] = searxng_url
                save_auto_config(auto_config)
                st.success("✅ 配置完成！正在进入主界面...")
                time.sleep(1)
                st.rerun()
                return
    else:
        st.warning("⚠️ 未检测到本地AI服务，请手动配置")
    
    st.divider()
    
    # 手动配置
    st.subheader("⚙️ 步骤2: 手动配置")
    
    # 简化的配置选项
    config_option = st.radio(
        "选择AI服务类型",
        options=[
            "🚀 Ollama (本地推荐)",
            "💻 LM Studio (本地图形界面)", 
            "☁️ OpenAI (云端服务)",
            "🔧 自定义配置"
        ],
        help="选择您要使用的AI服务类型"
    )
    
    manual_config = None
    
    if "🚀 Ollama" in config_option:
        st.subheader("🚀 Ollama 配置")
        models = get_models_for_provider("ollama", "http://localhost:11434")
        if models:
            chat_models, embedding_models = categorize_models(models)
            
            col1, col2 = st.columns(2)
            with col1:
                chat_model = st.selectbox("💬 聊天模型", options=chat_models if chat_models else models)
            with col2:
                embedding_model = st.selectbox("🧠 嵌入模型", options=embedding_models if embedding_models else models)
            
            if chat_model and embedding_model:
                # 添加搜索引擎选择
                st.subheader("🔍 选择搜索引擎")
                search_options = {
                    "🦆 DuckDuckGo (默认推荐)": "duckduckgo",
                    "🔍 SearXNG (本地备用)": "searxng",
                    "🌐 You.com (免密钥)": "youcom"
                }
                
                selected_search = st.radio(
                    "搜索引擎",
                    options=list(search_options.keys()),
                    help="默认使用 DuckDuckGo；SearXNG 可作为本地备用搜索；You.com 为免密钥远程搜索",
                    horizontal=True,
                    key="ollama_search"
                )
                
                search_provider = search_options[selected_search]
                searxng_url = None
                
                # 如果选择了 SearXNG，让用户配置地址
                if search_provider == "searxng":
                    searxng_url = st.text_input(
                        "🌐 SearXNG 服务地址",
                        value="http://localhost:8090",
                        help="请输入您的 SearXNG 实例地址",
                        placeholder="http://localhost:8090",
                        key="ollama_searxng_url"
                    )
                
                manual_config = {
                    "name": "Ollama",
                    "provider": "ollama",
                    "url": "http://localhost:11434/v1",
                    "chat_model": chat_model,
                    "embedding_model": embedding_model,
                    "search_provider": search_provider
                }
                
                if searxng_url:
                    manual_config["searxng_url"] = searxng_url
        else:
            st.warning("⚠️ 无法连接到 Ollama 服务，请确保 Ollama 已启动")
    
    elif "💻 LM Studio" in config_option:
        st.subheader("💻 LM Studio 配置")
        models = get_models_for_provider("lmstudio", "http://localhost:1234")
        if models:
            chat_models, embedding_models = categorize_models(models)
            
            col1, col2 = st.columns(2)
            with col1:
                chat_model = st.selectbox("💬 聊天模型", options=chat_models if chat_models else models)
            with col2:
                embedding_model = st.selectbox("🧠 嵌入模型", options=embedding_models if embedding_models else models)
            
            if chat_model and embedding_model:
                # 添加搜索引擎选择
                st.subheader("🔍 选择搜索引擎")
                search_options = {
                    "🦆 DuckDuckGo (默认推荐)": "duckduckgo",
                    "🔍 SearXNG (本地备用)": "searxng",
                    "🌐 You.com (免密钥)": "youcom"
                }
                
                selected_search = st.radio(
                    "搜索引擎",
                    options=list(search_options.keys()),
                    help="默认使用 DuckDuckGo；SearXNG 可作为本地备用搜索；You.com 为免密钥远程搜索",
                    horizontal=True,
                    key="lmstudio_search"
                )
                
                search_provider = search_options[selected_search]
                searxng_url = None
                
                # 如果选择了 SearXNG，让用户配置地址
                if search_provider == "searxng":
                    searxng_url = st.text_input(
                        "🌐 SearXNG 服务地址",
                        value="http://localhost:8090",
                        help="请输入您的 SearXNG 实例地址",
                        placeholder="http://localhost:8090",
                        key="lmstudio_searxng_url"
                    )
                
                manual_config = {
                    "name": "LM Studio", 
                    "provider": "lmstudio",
                    "url": "http://localhost:1234/v1",
                    "chat_model": chat_model,
                    "embedding_model": embedding_model,
                    "search_provider": search_provider
                }
                
                if searxng_url:
                    manual_config["searxng_url"] = searxng_url
        else:
            st.warning("⚠️ 无法连接到 LM Studio 服务，请确保 LM Studio 已启动")
    
    elif "☁️ OpenAI" in config_option:
        st.subheader("☁️ OpenAI 配置")
        api_key = st.text_input("🔑 OpenAI API Key", type="password", help="请输入您的OpenAI API密钥")
        if api_key:
            # 预定义 OpenAI 模型（因为需要 API Key 才能获取）
            openai_models = {
                "💬 聊天模型": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
                "🧠 嵌入模型": ["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"]
            }
            
            col1, col2 = st.columns(2)
            with col1:
                chat_model = st.selectbox("💬 聊天模型", options=openai_models["💬 聊天模型"])
            with col2:
                embedding_model = st.selectbox("🧠 嵌入模型", options=openai_models["🧠 嵌入模型"])
            
            # 添加搜索引擎选择
            st.subheader("🔍 选择搜索引擎")
            search_options = {
                "🦆 DuckDuckGo (默认推荐)": "duckduckgo",
                "🔍 SearXNG (本地备用)": "searxng",
                "🌐 You.com (免密钥)": "youcom"
            }
            
            selected_search = st.radio(
                "搜索引擎",
                options=list(search_options.keys()),
                help="默认使用 DuckDuckGo；SearXNG 可作为本地备用搜索；You.com 为免密钥远程搜索",
                horizontal=True,
                key="openai_search"
            )
            
            search_provider = search_options[selected_search]
            searxng_url = None
            
            # 如果选择了 SearXNG，让用户配置地址
            if search_provider == "searxng":
                searxng_url = st.text_input(
                    "🌐 SearXNG 服务地址",
                    value="http://localhost:8090",
                    help="请输入您的 SearXNG 实例地址",
                    placeholder="http://localhost:8090",
                    key="openai_searxng_url"
                )
            
            manual_config = {
                "name": "OpenAI",
                "provider": "openai", 
                "url": "https://api.openai.com/v1",
                "api_key": api_key,
                "chat_model": chat_model,
                "embedding_model": embedding_model,
                "search_provider": search_provider
            }
            
            if searxng_url:
                manual_config["searxng_url"] = searxng_url
    
    elif "🔧 自定义" in config_option:
        with st.expander("🚀 自定义配置", expanded=True):
            url = st.text_input("🌐 API地址", placeholder="http://localhost:8000/v1")
            
            if url:
                # 尝试获取模型列表
                models = get_models_for_provider("custom", url.rstrip('/v1'))
                
                if models:
                    st.success(f"✅ 检测到 {len(models)} 个可用模型")
                    chat_models, embedding_models = categorize_models(models)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        chat_model = st.selectbox("💬 聊天模型", options=chat_models if chat_models else models)
                    with col2:
                        embedding_model = st.selectbox("🧠 嵌入模型", options=embedding_models if embedding_models else models)
                    
                    if chat_model and embedding_model:
                        # 添加搜索引擎选择
                        st.subheader("🔍 选择搜索引擎")
                        search_options = {
                            "🦆 DuckDuckGo (默认推荐)": "duckduckgo",
                            "🔍 SearXNG (本地备用)": "searxng",
                            "🌐 You.com (免密钥)": "youcom"
                        }
                        
                        selected_search = st.radio(
                            "搜索引擎",
                            options=list(search_options.keys()),
                            help="默认使用 DuckDuckGo；SearXNG 可作为本地备用搜索；You.com 为免密钥远程搜索",
                            horizontal=True,
                            key="custom_search_1"
                        )
                        
                        search_provider = search_options[selected_search]
                        searxng_url = None
                        
                        # 如果选择了 SearXNG，让用户配置地址
                        if search_provider == "searxng":
                            searxng_url = st.text_input(
                                "🌐 SearXNG 服务地址",
                                value="http://localhost:8090",
                                help="请输入您的 SearXNG 实例地址",
                                placeholder="http://localhost:8090",
                                key="custom_searxng_url_1"
                            )
                        
                        manual_config = {
                            "name": "自定义配置",
                            "provider": "custom",
                            "url": url,
                            "chat_model": chat_model,
                            "embedding_model": embedding_model,
                            "search_provider": search_provider
                        }
                        
                        if searxng_url:
                            manual_config["searxng_url"] = searxng_url
                else:
                    st.warning("⚠️ 无法从此地址获取模型列表，请检查地址是否正确")
                    # 手动输入模型名
                    st.info("📝 请手动输入模型名称")
                    col1, col2 = st.columns(2)
                    with col1:
                        chat_model = st.text_input("💬 聊天模型", placeholder="例如: llama2")
                    with col2:
                        embedding_model = st.text_input("🧠 嵌入模型", placeholder="例如: nomic-embed-text")
                    
                    if chat_model and embedding_model:
                        # 添加搜索引擎选择
                        st.subheader("🔍 选择搜索引擎")
                        search_options = {
                            "🦆 DuckDuckGo (默认推荐)": "duckduckgo",
                            "🔍 SearXNG (本地备用)": "searxng",
                            "🌐 You.com (免密钥)": "youcom"
                        }
                        
                        selected_search = st.radio(
                            "搜索引擎",
                            options=list(search_options.keys()),
                            help="默认使用 DuckDuckGo；SearXNG 可作为本地备用搜索；You.com 为免密钥远程搜索",
                            horizontal=True,
                            key="custom_search_2"
                        )
                        
                        search_provider = search_options[selected_search]
                        searxng_url = None
                        
                        # 如果选择了 SearXNG，让用户配置地址
                        if search_provider == "searxng":
                            searxng_url = st.text_input(
                                "🌐 SearXNG 服务地址",
                                value="http://localhost:8090",
                                help="请输入您的 SearXNG 实例地址",
                                placeholder="http://localhost:8090",
                                key="custom_searxng_url_2"
                            )
                        
                        manual_config = {
                            "name": "自定义配置",
                            "provider": "custom",
                            "url": url,
                            "chat_model": chat_model,
                            "embedding_model": embedding_model,
                            "search_provider": search_provider
                        }
                        
                        if searxng_url:
                            manual_config["searxng_url"] = searxng_url
    
    # 测试配置
    if manual_config:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔗 测试连接", use_container_width=True):
                with st.spinner("正在测试连接..."):
                    if test_config_connection(manual_config):
                        st.success("✅ 连接成功！")
                    else:
                        st.error("❌ 连接失败，请检查配置")
        
        with col2:
            if st.button("✨ 保存配置", type="primary", use_container_width=True):
                save_manual_config(manual_config)
                st.success("✅ 配置完成！正在进入主界面...")
                time.sleep(1)
                st.rerun()

def detect_available_services():
    """检测可用的本地服务并获取模型列表"""
    import requests
    
    services = [
        ("http://localhost:11434", "Ollama", "ollama"),
        ("http://localhost:1234", "LM Studio", "lmstudio"),
        ("http://localhost:8000", "本地API", "local_api")
    ]
    
    for url, name, provider in services:
        try:
            # 先测试基本连接
            response = requests.get(f"{url}/v1/models", timeout=3)
            if response.status_code == 200:
                # 获取模型列表
                models_data = response.json()
                available_models = []
                
                if "data" in models_data:
                    # OpenAI格式: {"data": [{"id": "model_name"}, ...]}
                    available_models = [model["id"] for model in models_data["data"]]
                elif "models" in models_data:
                    # Ollama格式: {"models": [{"name": "model_name"}, ...]}
                    available_models = [model["name"] for model in models_data["models"]]
                elif isinstance(models_data, list):
                    # 简单格式: ["model1", "model2", ...]
                    available_models = models_data
                
                if available_models:
                    return {
                        "name": name,
                        "provider": provider,
                        "url": f"{url}/v1",
                        "available_models": available_models
                    }
        except:
            continue
    return None

def categorize_models(models):
    """将模型分类为聊天模型和嵌入模型"""
    chat_models = []
    embedding_models = []
    
    for model in models:
        model_lower = model.lower()
        # 判断是否为嵌入模型
        if any(keyword in model_lower for keyword in ['embed', 'embedding', 'nomic', 'bge', 'gte']):
            embedding_models.append(model)
        else:
            chat_models.append(model)
    
    return chat_models, embedding_models

def get_models_for_provider(provider_type, url):
    """为指定提供商获取模型列表"""
    import requests
    
    try:
        response = requests.get(f"{url}/models", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            
            if "data" in models_data:
                return [model["id"] for model in models_data["data"]]
            elif "models" in models_data:
                return [model["name"] for model in models_data["models"]]
            elif isinstance(models_data, list):
                return models_data
        return []
    except:
        return []

def test_searxng_connection(searxng_url="http://localhost:8090"):
    """测试 SearXNG 连接"""
    try:
        import requests
        # 确保 URL格式正确
        if not searxng_url.startswith('http'):
            searxng_url = f"http://{searxng_url}"
        
        # 测试搜索接口
        response = requests.get(f"{searxng_url}/search", 
                               params={"q": "test", "format": "json"}, 
                               timeout=3)
        return response.status_code == 200
    except:
        return False

def test_config_connection(config):
    """测试配置连接"""
    try:
        import requests
        response = requests.get(f"{config['url']}/models", timeout=3)
        return response.status_code == 200
    except:
        return False

def save_auto_config(config):
    """保存自动检测的配置"""
    from user_config import get_user_config_manager
    
    config_manager = get_user_config_manager()
    if config_manager:
        user_config = {
            "model_config": {
                "providers": {
                    config["provider"]: {
                        "base_url": config["url"]
                    }
                },
                "defaults": {
                    "llm_provider": config["provider"],
                    "llm_model": config["chat_model"],
                    "embedding_model": config["embedding_model"],
                    "search_provider": config.get("search_provider", "duckduckgo"),
                    "output_language": "zh"
                }
            },
            "config_completed": True,
            "config_source": "auto"
        }
        
        # 如果有自定义 SearXNG 地址，保存到搜索配置中
        if config.get("searxng_url"):
            user_config["search_config"] = {
                "search_providers": {
                    "searxng": {
                        "base_url": config["searxng_url"]
                    }
                }
            }
        
        config_manager.save_user_config(user_config)

def save_manual_config(config):
    """保存手动配置"""
    from user_config import get_user_config_manager
    
    config_manager = get_user_config_manager()
    if config_manager:
        user_config = {
            "model_config": {
                "providers": {
                    config["provider"]: {
                        "base_url": config["url"]
                    }
                },
                "defaults": {
                    "llm_provider": config["provider"],
                    "llm_model": config["chat_model"], 
                    "embedding_model": config["embedding_model"],
                    "search_provider": config.get("search_provider", "duckduckgo"),
                    "output_language": "zh"
                }
            },
            "config_completed": True,
            "config_source": "manual"
        }
        
        if "api_key" in config:
            user_config["model_config"]["providers"][config["provider"]]["api_key"] = config["api_key"]
        
        # 如果有自定义 SearXNG 地址，保存到搜索配置中
        if config.get("searxng_url"):
            user_config["search_config"] = {
                "search_providers": {
                    "searxng": {
                        "base_url": config["searxng_url"]
                    }
                }
            }
        
        config_manager.save_user_config(user_config)

def get_saved_config_info():
    """获取已保存的配置信息用于显示"""
    from user_config import get_user_config_manager
    
    config_manager = get_user_config_manager()
    if not config_manager:
        return None
    
    user_config = config_manager.get_user_config()
    model_config = user_config.get("model_config", {})
    defaults = model_config.get("defaults", {})
    
    return {
        "model_name": defaults.get("llm_model", "未配置"),
        "search_name": get_search_display_name(defaults.get("search_provider", "duckduckgo"))
    }

def get_search_display_name(search_provider):
    """获取搜索引擎显示名称"""
    search_names = {
        "duckduckgo": "DuckDuckGo",
        "searxng": "SearXNG",
        "youcom": "You.com"
    }
    return search_names.get(search_provider, search_provider)

def get_config_parameters():
    """从已保存的配置获取参数"""
    from user_config import get_user_config_manager
    
    config_manager = get_user_config_manager()
    if not config_manager:
        return None
    
    user_config = config_manager.get_user_config()
    model_config = user_config.get("model_config", {})
    
    if not model_config:
        return None
    
    providers = model_config.get("providers", {})
    defaults = model_config.get("defaults", {})
    
    provider_key = defaults.get("llm_provider")
    if not provider_key or provider_key not in providers:
        return None
    
    provider_config = providers[provider_key]
    
    return {
        "provider_key": provider_key,
        "api_base": provider_config.get("base_url"),
        "chat_model": defaults.get("llm_model"),
        "embedding_model": defaults.get("embedding_model"),
        "search_provider": defaults.get("search_provider", "duckduckgo"),
        "selected_language": defaults.get("output_language", "zh"),
        "provider_config": provider_config
    }

def reset_user_config():
    """重置用户配置"""
    from user_config import get_user_config_manager
    
    config_manager = get_user_config_manager()
    if config_manager:
        config_manager.reset_config()


def format_claim_markdown(claim_info):
    claim_md = "### 节点 1：新闻中心声明抽取\n\n"
    claim_md += f"**中心思想：** {claim_info.central_summary}\n\n"
    claim_md += f"**中心声明：** {claim_info.claim}\n\n"
    if claim_info.entities:
        claim_md += f"**核心实体：** {'、'.join(claim_info.entities)}\n\n"
    if claim_info.reason:
        claim_md += f"**选择理由：** {claim_info.reason}\n\n"
    return claim_md


def format_evidence_markdown(evidence_info):
    evaluation_evidence = evidence_info.evidence_chunks
    filter_summary = evidence_info.filter_summary

    evidence_md = "### 节点 2：查询改写与证据检索\n\n"
    if evidence_info.queries:
        evidence_md += "**查询改写：**\n\n"
        for idx, query in enumerate(evidence_info.queries, start=1):
            evidence_md += f"{idx}. `{query.query}`"
            if query.purpose:
                evidence_md += f" - {query.purpose}"
            evidence_md += "\n"
        evidence_md += "\n"

    evidence_md += "**证据来源：**\n\n"
    if filter_summary:
        evidence_md += (
            f"相关度阈值: `{filter_summary.get('min_similarity')}`；"
            f"Top K: `{filter_summary.get('top_k')}`；"
            f"候选: `{filter_summary.get('input_count')}`；"
            f"保留: `{filter_summary.get('selected_count')}`\n\n"
        )

    if not evaluation_evidence:
        evidence_md += "检索到网页，但没有达到相关度阈值和实体命中要求的证据。\n\n"

    for j, chunk in enumerate(evaluation_evidence):
        try:
            similarity = float(chunk.get("similarity", 0))
        except (TypeError, ValueError):
            similarity = 0.0
        evidence_md += f"**[{j+1}]:**\n"
        evidence_md += f"{chunk['text']}\n"
        evidence_md += f"相关度: {similarity:.3f}\n"
        evidence_md += f"来源: {chunk['source']}\n\n"

    return evidence_md


def get_verdict_display(verdict):
    normalized_verdict = verdict.upper().replace("_", " ")
    if normalized_verdict == "TRUE":
        return "✅", "正确"
    if normalized_verdict == "FALSE":
        return "❌", "错误"
    if normalized_verdict == "PARTIALLY TRUE":
        return "⚠️", "部分正确"
    return "❓", "无法验证"


def format_verdict_markdown(verification):
    emoji, verdict_cn = get_verdict_display(verification.final_verdict)
    verdict_md = "### 节点 3：多步分解声明验证\n\n"
    if verification.atomic_checks:
        verdict_md += "**原子声明核验：**\n\n"
        for idx, check in enumerate(verification.atomic_checks, start=1):
            verdict_md += f"**{idx}. {check.atomic_claim}**\n\n"
            if check.five_w:
                five_w_text = "；".join(
                    f"{key}: {value}"
                    for key, value in check.five_w.items()
                    if value
                )
                if five_w_text:
                    verdict_md += f"5W: {five_w_text}\n\n"
            verdict_md += f"判定: `{check.verdict}`\n\n"
            if check.reasoning:
                verdict_md += f"{check.reasoning}\n\n"

    reasoning_text = verification.reasoning
    if looks_like_json(reasoning_text):
        reasoning_text = "验证模型返回了结构化 JSON，但未能生成可直接展示的推理摘要。请以上方原子声明核验结果为准。"

    verdict_md += f"### {emoji} 结论: {verdict_cn}\n\n"
    verdict_md += f"### 推理过程\n\n{reasoning_text}\n\n"
    return verdict_md


def show_simplified_fact_check_page():
    """显示简化的事实核查页面 - 无复杂配置界面"""
    st.markdown(
        """
    本应用程序使用本地AI模型验证陈述的准确性。
    请在下方输入需要核查的新闻，系统将检索网络证据进行新闻核查。
    """
    )

    # 简化的侧边栏 - 只显示状态和基本信息
    with st.sidebar:
        st.header("📊 系统状态")
        
        # 获取已保存的配置
        config_info = get_saved_config_info()
        if config_info:
            st.success(f"✅ AI模型: {config_info['model_name']}")
            st.success(f"✅ 搜索引擎: {config_info['search_name']}")
        
        st.divider()
        
        # 快速设置 - 只显示必要的
        with st.expander("⚙️ 快速设置"):
            temperature = st.slider(
                "创造性",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                help="较低的值使响应更确定，较高的值使响应更具创造性",
            )
            language = st.selectbox(
                "输出语言",
                options=["自动检测", "中文", "English"],
                help="选择AI回复的语言"
            )
        
        st.divider()
        
        # 配置管理链接
        if st.button("🔧 重新配置", help="重新设置 AI 模型和服务"):
            reset_user_config()
            st.rerun()
        
        st.divider()
        st.markdown("### 关于")
        st.markdown("虚假新闻检测器:")
        st.markdown("1. 抽取新闻中心声明")
        st.markdown("2. 改写查询并检索证据")
        st.markdown("3. 分解声明并逐步验证")
        st.markdown("使用Streamlit、BGE-M3和LLM开发 ❤️")

    # 使用已保存的配置获取参数
    config_params = get_config_parameters()
    if not config_params:
        st.error("配置获取失败，请重新配置")
        if st.button("重新配置"):
            reset_user_config()
            st.rerun()
        return

    # 以下的逻辑保持不变，只是使用保存的配置参数
    # 如果不存在，初始化会话状态以存储聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 主输入区域
    user_input = st.chat_input("请在下方输入需要核查的新闻...")

    if user_input:
        # 将用户消息添加到聊天历史
        st.session_state.messages.append({"role": "user", "content": user_input})

        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(user_input)

        # 创建助手消息容器用于流式输出
        assistant_message = st.chat_message("assistant")

        # 创建空的placeholder组件用于逐步更新
        claim_placeholder = assistant_message.empty()
        evidence_placeholder = assistant_message.empty()
        verdict_placeholder = assistant_message.empty()

        # 检查模型配置是否有效 - 使用保存的配置
        api_base = config_params["api_base"]
        chat_model = config_params["chat_model"]
        embedding_model = config_params["embedding_model"]
        search_provider = config_params["search_provider"]
        language_map = {
            "自动检测": "auto",
            "中文": "zh",
            "English": "en",
        }
        selected_language = language_map.get(language, config_params["selected_language"])
        provider_config = config_params["provider_config"]
        
        if not api_base or not chat_model:
            st.error("配置信息不完整，请重新配置模型提供商")
            st.stop()

        if not embedding_model:
            st.error("配置信息不完整，请重新配置嵌入模型")
            st.stop()

        # 获取配置
        embedding_api_key = provider_config.get("api_key", "lm-studio")
        current_defaults = model_manager.get_current_config().get("defaults", {})
        search_config = dict(model_manager.get_search_provider_config(search_provider))
        for key in (
            "evidence_min_similarity",
            "evidence_entity_gate",
            "evidence_ranker_multiplier",
        ):
            if key in current_defaults:
                search_config[key] = current_defaults[key]
        searxng_url = search_config.get("base_url", "http://localhost:8090")
        
        # 使用侧边栏的设置覆盖默认值
        max_tokens = 1000  # 固定值，简化配置

        # 初始化FactChecker
        fact_checker = FactChecker(
            api_base=api_base,
            model=chat_model,
            temperature=temperature,
            max_tokens=max_tokens,
            embedding_base_url=api_base,
            embedding_model=embedding_model,
            embedding_api_key=embedding_api_key,
            search_engine=search_provider,
            searxng_url=searxng_url,
            output_language=selected_language,
            search_config=search_config,
        )

        # 三节点事实核查流水线：按节点完成顺序逐步渲染
        claim_placeholder.markdown("### 节点 1：正在抽取新闻中心声明...")
        evidence_placeholder.empty()
        verdict_placeholder.empty()

        # 证据 Top K：节点 2 会先按相关度阈值和实体命中过滤，再取前 K 条
        search_max_results = search_config.get("max_results", 5)
        evidence_top_k = int(current_defaults.get("evidence_top_k", 8))

        result = None
        claim_info = None
        evidence_info = None
        verification = None
        claim = ""
        evidence_md = ""
        verdict_md = ""
        evaluation_evidence = []

        for event_name, payload in fact_checker.check_news_stream(
            user_input,
            search_results_per_query=search_max_results,
            evidence_top_k=evidence_top_k,
        ):
            if event_name == "claim":
                claim_info = payload
                claim = claim_info.claim
                claim_placeholder.markdown(format_claim_markdown(claim_info))
                evidence_placeholder.markdown("### 节点 2：正在改写查询并检索证据...")
            elif event_name == "evidence":
                evidence_info = payload
                evaluation_evidence = evidence_info.evidence_chunks
                evidence_md = format_evidence_markdown(evidence_info)
                evidence_placeholder.markdown(evidence_md)
                verdict_placeholder.markdown("### 节点 3：正在分解声明并逐步验证...")
            elif event_name == "verification":
                verification = payload
                verdict_md = format_verdict_markdown(verification)
                verdict_placeholder.markdown(verdict_md)
            elif event_name == "complete":
                result = payload

        if result is None or claim_info is None or evidence_info is None or verification is None:
            st.error("事实核查流程未能完整执行，请检查模型、检索和向量服务状态。")
            st.stop()

        verdict = verification.final_verdict

        # 整合完整的响应内容用于保存到聊天历史
        full_response = f"""
### 🔍 提取新闻的核心声明

{claim}

---

{evidence_md}

---

{verdict_md}
"""

        # 添加助手响应到聊天历史
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

        # 保存到数据库
        db_utils.save_fact_check(
            st.session_state.user_id,
            user_input,
            claim,
            verdict,
            verification.reasoning,
            evaluation_evidence,
        )


def show_history_page():
    """显示历史记录页面"""
    st.header("历史记录")
    st.write("以下是您过去进行的事实核查记录")

    # 分页控制
    items_per_page = 5
    total_items = db_utils.count_user_history(st.session_state.user_id)

    if "history_page" not in st.session_state:
        st.session_state.history_page = 0

    total_pages = (total_items + items_per_page - 1) // items_per_page

    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("上一页", disabled=(st.session_state.history_page == 0)):
                st.session_state.history_page -= 1
                st.rerun()
        with col2:
            st.write(f"第 {st.session_state.history_page + 1} 页，共 {total_pages} 页")
        with col3:
            if st.button(
                "下一页",
                disabled=(
                    st.session_state.history_page == total_pages - 1 or total_pages == 0
                ),
            ):
                st.session_state.history_page += 1
                st.rerun()

    # 获取用户历史记录
    history_items = db_utils.get_user_history(
        st.session_state.user_id,
        limit=items_per_page,
        offset=st.session_state.history_page * items_per_page,
    )

    if not history_items:
        st.info("您还没有任何历史记录")
        return

    # 显示历史记录
    for item in history_items:
        with st.container():
            cols = st.columns([4, 1, 1])
            with cols[0]:
                st.subheader(
                    f"{item['claim'][:100]}..."
                    if len(item["claim"]) > 100
                    else item["claim"]
                )

                # 添加判断结果和时间
                verdict = item["verdict"].upper().replace("_", " ")
                if verdict == "TRUE":
                    emoji = "✅"
                    verdict_cn = "正确"
                elif verdict == "FALSE":
                    emoji = "❌"
                    verdict_cn = "错误"
                elif verdict == "PARTIALLY TRUE":
                    emoji = "⚠️"
                    verdict_cn = "部分正确"
                else:
                    emoji = "❓"
                    verdict_cn = "无法验证"

                st.write(f"结论: {emoji} {verdict_cn}")
                st.write(f"时间: {item['created_at']}")

            with cols[1]:
                if st.button("查看详情", key=f"view_{item['id']}"):
                    st.session_state.current_history_id = item["id"]
                    st.session_state.page = "details"
                    st.rerun()

            st.divider()


def show_history_detail_page():
    """显示历史记录详情页面"""
    if st.session_state.current_history_id is None:
        st.error("未找到历史记录")
        if st.button("返回历史列表"):
            st.session_state.page = "history"
            st.rerun()
        return

    # 获取历史记录详情
    history_item = db_utils.get_history_by_id(st.session_state.current_history_id)

    if not history_item:
        st.error("未找到历史记录")
        if st.button("返回历史列表"):
            st.session_state.page = "history"
            st.rerun()
        return

    # 显示返回按钮
    if st.button("返回历史列表"):
        st.session_state.page = "history"
        st.rerun()

    # 显示历史记录详情
    st.header("核查详情")

    st.subheader("原始文本")
    st.write(history_item["original_text"])

    st.subheader("🔍 提取的核心声明")
    st.write(history_item["claim"])

    # 显示证据
    st.subheader("🔗 证据来源")
    for j, chunk in enumerate(history_item["evidence"]):
        st.markdown(f"**[{j+1}]:**")
        st.markdown(f"{chunk['text']}")
        st.markdown(f"来源: {chunk['source']}")
        if "similarity" in chunk and chunk["similarity"] is not None:
            st.markdown(f"相关性: {chunk['similarity']:.2f}")
        st.markdown("---")

    # 显示判断结果
    verdict = history_item["verdict"].upper().replace("_", " ")
    if verdict == "TRUE":
        emoji = "✅"
        verdict_cn = "正确"
    elif verdict == "FALSE":
        emoji = "❌"
        verdict_cn = "错误"
    elif verdict == "PARTIALLY TRUE":
        emoji = "⚠️"
        verdict_cn = "部分正确"
    else:
        emoji = "❓"
        verdict_cn = "无法验证"

    st.subheader(f"{emoji} 结论: {verdict_cn}")

    st.subheader("推理过程")
    st.write(history_item["reasoning"])

    # 显示导出选项
    st.divider()
    st.subheader("导出报告")

    # 创建PDF导出按钮
    try:
        pdf_data = generate_fact_check_pdf(history_item)

        # 生成文件名
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"事实核查报告_{current_time}.pdf"

        # 使用HTML强制下载
        pdf_b64 = base64.b64encode(pdf_data).decode()
        href = f"""
        <a href="data:application/pdf;base64,{pdf_b64}" 
        download="{filename}" 
        target="_blank"
        style="display: inline-block; padding: 0.25em 0.5em; 
        background-color: #4CAF50; color: white; 
        text-decoration: none; border-radius: 4px;">
        导出为PDF
        </a>
        """
        st.markdown(href, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"PDF生成错误: {str(e)}")
        st.info("请确保已安装ReportLab库: pip install reportlab")


# 全局状态初始化
if "page" not in st.session_state:
    st.session_state.page = "home"  # 可能的值: 'home', 'history', 'details'

if "current_history_id" not in st.session_state:
    st.session_state.current_history_id = None

# 早期检查持久登录状态 - 在任何UI显示之前
if "user_id" not in st.session_state or st.session_state.user_id is None:
    saved_login = auth.check_saved_login()
    if saved_login:
        st.session_state.user_id = saved_login["user_id"]
        st.session_state.username = saved_login["username"]
        st.session_state.persisted_login = saved_login

# 检查是否已登录，否则显示登录界面
is_authenticated = auth.show_auth_ui()

if is_authenticated:
    # 用户已登录，检查是否需要配置
    
    # 检查用户配置状态
    if not check_user_config_status():
        # 显示配置向导
        show_initial_config_wizard()
    else:
        # 配置完成，显示主应用程序
        # 显示顶部导航栏
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.title("AI虚假新闻检测器")
        with col2:
            if st.button("首页", use_container_width=True):
                st.session_state.page = "home"
                st.rerun()
        with col3:
            if st.button("历史记录", use_container_width=True):
                st.session_state.page = "history"
                st.rerun()
        with col4:
            if st.button("登出", use_container_width=True):
                auth.logout()
                st.rerun()

        # 显示当前用户信息
        st.write(f"已登录用户: {st.session_state.username}")

        # 根据当前页面显示不同的内容
        if st.session_state.page == "home":
            # 主页 - 使用简化的事实核查界面
            show_simplified_fact_check_page()
        elif st.session_state.page == "history":
            # 历史记录页面
            show_history_page()
        elif st.session_state.page == "details":
            # 历史详情页面
            show_history_detail_page()
