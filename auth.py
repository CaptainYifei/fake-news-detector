import streamlit as st
import db_utils

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
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
        
    if 'username' not in st.session_state:
        st.session_state.username = None
        
    if 'auth_page' not in st.session_state:
        st.session_state.auth_page = 'login'  # 可选值: 'login', 'register'

def is_logged_in() -> bool:
    """
    检查用户是否已登录
    
    Returns:
        用户是否已登录
    """
    return st.session_state.user_id is not None

def login(username: str, password: str) -> bool:
    """
    验证用户并设置会话状态
    
    Args:
        username: 用户名
        password: 密码
        
    Returns:
        登录是否成功
    """
    user_id = db_utils.verify_user(username, password)
    
    if user_id:
        st.session_state.user_id = user_id
        st.session_state.username = username
        return True
    else:
        return False

def logout():
    """
    登出用户，清除会话状态
    """
    st.session_state.user_id = None
    st.session_state.username = None
    
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
        submit = st.form_submit_button("登录")
        
        if submit:
            if login(username, password):
                st.success("登录成功！")
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
    
    if is_logged_in():
        return True
    
    # 如果未登录，显示登录或注册表单
    if st.session_state.auth_page == 'login':
        show_login_form()
    else:
        show_register_form()
    
    return False