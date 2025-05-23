import streamlit as st
import os
from datetime import datetime
import time
from fact_checker import FactChecker
import auth
import db_utils
from pdf_export import generate_fact_check_pdf

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
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# 定义函数
def show_fact_check_page():
    """显示主页的事实核查功能"""
    st.markdown("""
    本应用程序使用本地Qwen2.5模型验证陈述的准确性。
    请在下方输入需要核查的新闻，系统将检索网络证据进行新闻核查。
    """)
    
    # 侧边栏配置
    with st.sidebar:
        st.header("配置")
        
        # 本地API端点设置
        api_base = st.text_input("API基础URL", value="http://localhost:8000/v1", 
                                help="您的本地API端点的基础URL")
        
        # 模型选择 - 已更新为本地Qwen模型
        model_option = st.selectbox(
            "选择模型",
            ["Qwen2.5-14B-Instruct-AWQ"],
            index=0,
            help="使用本地Qwen2.5模型"
        )
        
        # 高级设置折叠部分
        with st.expander("高级设置"):
            temperature = st.slider("温度", min_value=0.0, max_value=1.0, value=0.0, step=0.1, 
                                  help="较低的值使响应更确定，较高的值使响应更具创造性")
            max_tokens = st.slider("最大响应长度", min_value=100, max_value=4000, value=1000, step=100,
                                  help="响应中的最大标记数")
        
        st.divider()
        st.markdown("### 关于")
        st.markdown("虚假新闻检测器:")
        st.markdown("1. 从新闻中提取核心声明")
        st.markdown("2. 在网络上搜索证据")
        st.markdown("3. 使用BGE-M3按相关性对证据进行排名")
        st.markdown("4. 基于证据提供结论")
        st.markdown("使用Streamlit、BGE-M3和LLM开发 ❤️")
    
    # 如果不存在，初始化会话状态以存储聊天历史
    if 'messages' not in st.session_state:
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
        
        # 初始化FactChecker
        fact_checker = FactChecker(api_base, model_option, temperature, max_tokens)
        
        # 第1步：提取声明
        claim_placeholder.markdown("### 🔍 正在提取新闻的核心声明...")
        claim = fact_checker.extract_claim(user_input)
        # 处理claim字符串，提取"claim:"后面的内容
        if "claim:" in claim.lower():
            claim = claim.split("claim:")[-1].strip()
        claim_placeholder.markdown(f"### 🔍 提取新闻的核心声明\n\n{claim}")
        
        # 第2步：搜索证据
        evidence_placeholder.markdown("### 🌐 正在搜索相关证据...")
        evidence_docs = fact_checker.search_evidence(claim)
        
        # 第3步：获取相关证据块
        evidence_placeholder.markdown("### 🌐 正在分析证据相关性...")
        evidence_chunks = fact_checker.get_evidence_chunks(evidence_docs, claim)
        
        # 显示证据结果
        evidence_md = "### 🔗 证据来源\n\n"
        for j, chunk in enumerate(evidence_chunks[:-1]):  # 跳过最后一个，与原始代码保持一致
            evidence_md += f"**[{j+1}]:**\n"
            evidence_md += f"{chunk['text']}\n"
            evidence_md += f"来源: {chunk['source']}\n\n"
        
        evidence_placeholder.markdown(evidence_md)
        
        # 第4步：评估声明
        verdict_placeholder.markdown("### ⚖️ 正在评估声明真实性...")
        evaluation = fact_checker.evaluate_claim(claim, evidence_chunks)
        
        # 确定结论表情符号
        verdict = evaluation["verdict"]
        if verdict.upper() == "TRUE":
            emoji = "✅"
            verdict_cn = "正确"
        elif verdict.upper() == "FALSE":
            emoji = "❌"
            verdict_cn = "错误"
        elif verdict.upper() == "PARTIALLY TRUE":
            emoji = "⚠️"
            verdict_cn = "部分正确"
        else:
            emoji = "❓"
            verdict_cn = "无法验证"
        
        # 显示最终结论
        verdict_md = f"### {emoji} 结论: {verdict_cn}\n\n"
        verdict_md += f"### 推理过程\n\n{evaluation['reasoning']}\n\n"
        
        verdict_placeholder.markdown(verdict_md)
        
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
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # 保存到数据库
        db_utils.save_fact_check(
            st.session_state.user_id,
            user_input,
            claim,
            verdict,
            evaluation["reasoning"],
            evidence_chunks
        )

def show_history_page():
    """显示历史记录页面"""
    st.header("历史记录")
    st.write("以下是您过去进行的事实核查记录")
    
    # 分页控制
    items_per_page = 5
    total_items = db_utils.count_user_history(st.session_state.user_id)
    
    if 'history_page' not in st.session_state:
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
            if st.button("下一页", disabled=(st.session_state.history_page == total_pages - 1 or total_pages == 0)):
                st.session_state.history_page += 1
                st.rerun()
    
    # 获取用户历史记录
    history_items = db_utils.get_user_history(
        st.session_state.user_id,
        limit=items_per_page,
        offset=st.session_state.history_page * items_per_page
    )
    
    if not history_items:
        st.info("您还没有任何历史记录")
        return
    
    # 显示历史记录
    for item in history_items:
        with st.container():
            cols = st.columns([4, 1, 1])
            with cols[0]:
                st.subheader(f"{item['claim'][:100]}..." if len(item['claim']) > 100 else item['claim'])
                
                # 添加判断结果和时间
                verdict = item['verdict'].upper()
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
                    st.session_state.current_history_id = item['id']
                    st.session_state.page = 'details'
                    st.rerun()
            
            st.divider()

def show_history_detail_page():
    """显示历史记录详情页面"""
    if st.session_state.current_history_id is None:
        st.error("未找到历史记录")
        if st.button("返回历史列表"):
            st.session_state.page = 'history'
            st.rerun()
        return
    
    # 获取历史记录详情
    history_item = db_utils.get_history_by_id(st.session_state.current_history_id)
    
    if not history_item:
        st.error("未找到历史记录")
        if st.button("返回历史列表"):
            st.session_state.page = 'history'
            st.rerun()
        return
    
    # 显示返回按钮
    if st.button("返回历史列表"):
        st.session_state.page = 'history'
        st.rerun()
    
    # 显示历史记录详情
    st.header("核查详情")
    
    st.subheader("原始文本")
    st.write(history_item['original_text'])
    
    st.subheader("🔍 提取的核心声明")
    st.write(history_item['claim'])
    
    # 显示证据
    st.subheader("🔗 证据来源")
    for j, chunk in enumerate(history_item['evidence']):
        st.markdown(f"**[{j+1}]:**")
        st.markdown(f"{chunk['text']}")
        st.markdown(f"来源: {chunk['source']}")
        if 'similarity' in chunk and chunk['similarity'] is not None:
            st.markdown(f"相关性: {chunk['similarity']:.2f}")
        st.markdown("---")
    
    # 显示判断结果
    verdict = history_item['verdict'].upper()
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
    st.write(history_item['reasoning'])
    
    # 显示导出选项
    st.divider()
    st.subheader("导出报告")

    # 创建PDF导出按钮
    try:
        pdf_data = generate_fact_check_pdf(history_item)
        
        # 生成文件名
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"事实核查报告_{current_time}.pdf"
        # print("*****",filename)
        # print("*****",pdf_data)
        with open("test.pdf", "wb") as f:
            f.write(pdf_data)
        import base64

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
        # test_data = "这是测试文本".encode('utf-8')
        # st.download_button("测试下载", test_data, "test.txt", "text/plain")
    except Exception as e:
        st.error(f"PDF生成错误: {str(e)}")
        st.info("请确保已安装ReportLab库: pip install reportlab")



# 全局状态初始化
if 'page' not in st.session_state:
    st.session_state.page = 'home'  # 可能的值: 'home', 'history', 'details'

if 'current_history_id' not in st.session_state:
    st.session_state.current_history_id = None

# 检查是否已登录，否则显示登录界面
is_authenticated = auth.show_auth_ui()

if is_authenticated:
    # 用户已登录，显示主应用程序
    
    # 显示顶部导航栏
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.title("AI虚假新闻检测器")
    with col2:
        if st.button("首页", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
    with col3:
        if st.button("历史记录", use_container_width=True):
            st.session_state.page = 'history'
            st.rerun()
    with col4:
        if st.button("登出", use_container_width=True):
            auth.logout()
            st.rerun()
    
    # 显示当前用户信息
    st.write(f"已登录用户: {st.session_state.username}")
    
    # 根据当前页面显示不同的内容
    if st.session_state.page == 'home':
        # 主页 - 事实核查界面
        show_fact_check_page()
    elif st.session_state.page == 'history':
        # 历史记录页面
        show_history_page()
    elif st.session_state.page == 'details':
        # 历史详情页面
        show_history_detail_page()