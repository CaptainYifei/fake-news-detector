import streamlit as st
import db_utils
import hashlib
import time
import json
import os

def login_required(func):
    """
    装饰器：确保用户已登录，否则重定向到登录页面
    """
    def wrapper(*args, **kwargs):
        if not is_logged_in():
            st.warning("请先登录")
            show_login_form()
            return None
        return func(*args, **kwargs)
    return wrapper

def init_auth_state():
    """
    初始化认证相关的session状态
    """
    # 检查是否已经有登录状态
    if 'user_id' not in st.session_state or st.session_state.user_id is None:
        # 检查session state中是否有持久化登录数据
        if 'persisted_login' in st.session_state:
            saved_login = st.session_state.persisted_login
            # 检查是否过期
            if saved_login["expires"] > int(time.time()):
                st.session_state.user_id = saved_login['user_id']
                st.session_state.username = saved_login['username']
            else:
                # 过期，清除
                del st.session_state.persisted_login
                st.session_state.user_id = None
                st.session_state.username = None
        else:
            # 检查文件中是否有保存的登录状态
            saved_login = check_saved_login()
            if saved_login:
                st.session_state.user_id = saved_login['user_id']
                st.session_state.username = saved_login['username']
                # 同时保存到session state以便下次使用
                st.session_state.persisted_login = saved_login
            else:
                st.session_state.user_id = None
                st.session_state.username = None

    if 'username' not in st.session_state:
        st.session_state.username = None

    if 'auth_page' not in st.session_state:
        st.session_state.auth_page = 'login'  # 可选值: 'login', 'register'

def generate_login_token(username: str) -> str:
    """
    生成登录令牌
    """
    timestamp = str(int(time.time()))
    token_string = f"{username}:{timestamp}:fake_news_detector"
    return hashlib.md5(token_string.encode()).hexdigest()

def get_login_cache_file():
    """获取登录缓存文件路径"""
    return "data/.login_cache.json"

def save_login_state(username: str, user_id: int, remember: bool = False):
    """
    保存登录状态到本地文件和session state
    """
    if remember:
        token = generate_login_token(username)
        login_data = {
            "username": username,
            "user_id": user_id,
            "token": token,
            "expires": int(time.time()) + (30 * 24 * 3600)  # 30天
        }

        # 确保data目录存在
        os.makedirs("data", exist_ok=True)

        # 保存到文件
        try:
            cache_file = get_login_cache_file()
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(login_data, f)

            # 保存到session state - 这里是关键！
            st.session_state.persisted_login = login_data

        except Exception as e:
            st.warning(f"保存登录状态失败: {e}")
    else:
        # 清除保存的登录状态
        try:
            cache_file = get_login_cache_file()
            if os.path.exists(cache_file):
                os.remove(cache_file)
        except Exception:
            pass

        # 清除session state
        if "persisted_login" in st.session_state:
            del st.session_state.persisted_login

def check_saved_login():
    """
    检查是否有有效的保存登录状态
    """
    try:
        cache_file = get_login_cache_file()
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                saved = json.load(f)

            # 检查是否过期
            current_time = int(time.time())
            if saved["expires"] > current_time:
                # 返回保存的登录数据
                return saved
            else:
                # 过期，删除缓存文件
                os.remove(cache_file)
    except Exception:
        # 文件损坏或读取失败，删除缓存文件
        try:
            if os.path.exists(get_login_cache_file()):
                os.remove(get_login_cache_file())
        except Exception:
            pass

    return None

def is_logged_in() -> bool:
    """
    检查用户是否已登录
    
    Returns:
        用户是否已登录
    """
    return st.session_state.user_id is not None

def login(username: str, password: str):
    """
    验证用户并设置会话状态

    Args:
        username: 用户名
        password: 密码

    Returns:
        成功时返回user_id，失败时返回None
    """
    user_id = db_utils.verify_user(username, password)

    if user_id:
        st.session_state.user_id = user_id
        st.session_state.username = username
        return user_id
    else:
        return None

def logout():
    """
    登出用户，清除会话状态
    """
    st.session_state.user_id = None
    st.session_state.username = None

    # 清除保存的登录状态
    save_login_state("", 0, remember=False)

    # 清除自动登录检查标记，下次启动时重新检查
    if "auto_login_checked" in st.session_state:
        del st.session_state.auto_login_checked

    # 清除聊天历史
    if 'messages' in st.session_state:
        st.session_state.messages = []

def register(username: str, password: str, confirm_password: str) -> tuple[bool, str]:
    """
    注册新用户
    
    Args:
        username: 用户名
        password: 密码
        confirm_password: 确认密码
        
    Returns:
        (成功标志, 错误消息)
    """
    # 验证用户输入
    if not username or len(username) < 3:
        return False, "用户名至少需要3个字符"
        
    if not password or len(password) < 6:
        return False, "密码至少需要6个字符"
        
    if password != confirm_password:
        return False, "两次输入的密码不匹配"
    
    # 尝试创建用户
    success = db_utils.create_user(username, password)
    
    if success:
        return True, ""
    else:
        return False, "用户名已存在，请选择其他用户名"

def show_login_form():
    """
    显示登录表单
    """
    st.subheader("登录")

    with st.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        remember_me = st.checkbox("记住登录状态", help="勾选后30天内无需重新登录")
        submit = st.form_submit_button("登录")

        if submit:
            result = login(username, password)
            if result:
                # 保存登录状态
                save_login_state(username, result, remember_me)
                st.success("登录成功！" + ("已保存登录状态" if remember_me else ""))
                st.rerun()  # 重新运行应用以更新UI
            else:
                st.error("用户名或密码错误")
    
    # 表单外部的导航按钮
    st.markdown("---")
    st.markdown("还没有账号？")
    
    if st.button("注册新账号"):
        st.session_state.auth_page = 'register'
        st.rerun()

def show_register_form():
    """
    显示注册表单
    """
    st.subheader("注册新账号")
    
    with st.form("register_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        confirm_password = st.text_input("确认密码", type="password")
        submit = st.form_submit_button("注册")
        
        if submit:
            success, error_msg = register(username, password, confirm_password)
            if success:
                st.success("注册成功！现在您可以登录了")
                st.session_state.auth_page = 'login'
                st.rerun()
            else:
                st.error(error_msg)
    
    # 表单外部的导航按钮
    st.markdown("---")
    st.markdown("已有账号？")
    
    if st.button("返回登录"):
        st.session_state.auth_page = 'login'
        st.rerun()

def show_auth_ui():
    """
    显示认证UI（登录或注册）
    """
    init_auth_state()

    # 如果已登录，直接返回
    if is_logged_in():
        return True

    # 如果未登录，显示登录或注册表单
    if st.session_state.auth_page == 'login':
        show_login_form()
    else:
        show_register_form()

    return False