import json
import os
from typing import Dict, Any, Optional
import streamlit as st


class UserConfigManager:
    """用户配置管理器"""

    def __init__(self, user_id: int):
        """
        初始化用户配置管理器

        Args:
            user_id: 用户ID
        """
        self.user_id = user_id
        self.config_dir = "data/user_configs"
        self.config_file = os.path.join(self.config_dir, f"user_{user_id}.json")
        self._ensure_config_dir()

    def _ensure_config_dir(self):
        """确保配置目录存在"""
        os.makedirs(self.config_dir, exist_ok=True)

    def get_user_config(self) -> Dict[str, Any]:
        """
        获取用户配置

        Returns:
            用户配置字典
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            st.warning(f"读取用户配置失败: {e}")
        return {}

    def save_user_config(self, config: Dict[str, Any]):
        """
        保存用户配置

        Args:
            config: 用户配置字典
        """
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            st.toast("配置已保存", icon="💾")
        except Exception as e:
            st.error(f"保存用户配置失败: {e}")

    def update_user_config(self, updates: Dict[str, Any]):
        """
        更新用户配置

        Args:
            updates: 要更新的配置项
        """
        current_config = self.get_user_config()
        current_config.update(updates)
        self.save_user_config(current_config)

    def get_model_config(self) -> Dict[str, Any]:
        """
        获取用户模型配置

        Returns:
            模型配置字典
        """
        return self.get_user_config().get("model_config", {})

    def save_model_config(self, model_config: Dict[str, Any]):
        """
        保存用户模型配置

        Args:
            model_config: 模型配置字典
        """
        user_config = self.get_user_config()
        user_config["model_config"] = model_config
        self.save_user_config(user_config)

    def get_search_config(self) -> Dict[str, Any]:
        """
        获取用户搜索配置

        Returns:
            搜索配置字典
        """
        return self.get_user_config().get("search_config", {})

    def save_search_config(self, search_config: Dict[str, Any]):
        """
        保存用户搜索配置

        Args:
            search_config: 搜索配置字典
        """
        user_config = self.get_user_config()
        user_config["search_config"] = search_config
        self.save_user_config(user_config)

    def get_default_config(self) -> Dict[str, Any]:
        """
        获取用户默认配置

        Returns:
            默认配置字典
        """
        return self.get_user_config().get("default_config", {})

    def save_default_config(self, default_config: Dict[str, Any]):
        """
        保存用户默认配置

        Args:
            default_config: 默认配置字典
        """
        user_config = self.get_user_config()
        user_config["default_config"] = default_config
        self.save_user_config(user_config)

    def reset_config(self):
        """重置用户配置"""
        try:
            if os.path.exists(self.config_file):
                os.remove(self.config_file)
            st.toast("配置已重置", icon="🔄")
        except Exception as e:
            st.error(f"重置配置失败: {e}")


def get_user_config_manager() -> Optional[UserConfigManager]:
    """
    获取当前用户的配置管理器

    Returns:
        用户配置管理器实例，如果用户未登录则返回None
    """
    if hasattr(st.session_state, "user_id") and st.session_state.user_id:
        return UserConfigManager(st.session_state.user_id)
    return None
