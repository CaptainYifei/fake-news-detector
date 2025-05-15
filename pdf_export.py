from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from io import BytesIO
from datetime import datetime
import os
import re
import tempfile
import logging
import platform

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_export")

# 检测系统
system = platform.system()
logger.info(f"当前操作系统: {system}")

# 尝试加载中文字体
try:
    # 首先尝试注册思源宋体（Adobe Source Han Sans）
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    chinese_font = 'STSong-Light'
    logger.info("成功加载内置中文字体: STSong-Light")
except Exception as e:
    logger.warning(f"无法加载内置中文字体 STSong-Light: {e}")
    
    # 尝试使用系统中文字体作为备选
    font_paths = []
    
    if system == 'Linux':
        font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        ]
    elif system == 'Darwin':  # macOS
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc"
        ]
    elif system == 'Windows':
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/msyh.ttf"
        ]
    
    # 添加可能与应用程序一起分发的字体
    font_paths.extend([
        os.path.join(os.path.dirname(__file__), "fonts/simhei.ttf"),
        os.path.join(os.path.dirname(__file__), "fonts/wqy-microhei.ttc"),
        "simhei.ttf",
        "wqy-microhei.ttc"
    ])
    
    font_loaded = False
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                logger.info(f"尝试加载字体: {font_path}")
                pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                chinese_font = 'ChineseFont'
                font_loaded = True
                logger.info(f"成功加载中文字体: {font_path}")
                break
        except Exception as e:
            logger.warning(f"加载字体 {font_path} 失败: {e}")
    
    # 如果没有找到任何可用的中文字体，使用默认字体
    if not font_loaded:
        chinese_font = 'Helvetica'
        logger.warning("未找到可用的中文字体，将使用Helvetica（可能导致中文乱码）")


def clean_html(text):
    """清理可能的HTML标签并处理内容"""
    if text is None:
        return ""
        
    # 将text转换为字符串
    text = str(text)
    
    # 清理HTML标签
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    
    # 移除不可见字符
    text = ''.join(c for c in text if ord(c) >= 32 or ord(c) == 9)
    
    # 处理换行和空格
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    return text


def generate_fact_check_pdf(history_item):
    """生成事实核查报告的PDF
    
    Args:
        history_item: 包含核查历史详情的字典
    
    Returns:
        BytesIO: 包含PDF数据的BytesIO对象
    """
    logger.info("开始生成PDF报告")
    
    try:
        # 尝试使用xhtml2pdf作为备选方案
        logger.info("尝试使用直接canvas方法生成PDF")
        return generate_pdf_with_canvas(history_item)
    except Exception as e:
        logger.error(f"使用canvas方法生成PDF失败: {e}")
        logger.info("尝试使用reportlab的SimpleDocTemplate方法生成PDF")
        try:
            return generate_pdf_with_template(history_item)
        except Exception as e:
            logger.error(f"使用SimpleDocTemplate方法生成PDF失败: {e}")
            # 生成一个最简单的PDF以确保功能正常
            return generate_simple_pdf(history_item)


def generate_pdf_with_canvas(history_item):
    """使用Canvas直接绘制PDF（适用于中文）"""
    buffer = BytesIO()
    
    # 页面设置
    page_width, page_height = A4
    margin = 72  # 1英寸边距
    
    # 创建Canvas
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setTitle("事实核查报告")
    
    # 设置中文字体
    c.setFont(chinese_font, 18)
    
    # 添加标题
    title = "事实核查报告"
    title_width = c.stringWidth(title, chinese_font, 18)
    c.drawString((page_width - title_width) / 2, page_height - margin, title)
    
    # 添加生成时间
    c.setFont(chinese_font, 10)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(margin, page_height - margin - 30, f"生成时间: {current_time}")
    
    # 当前Y位置（从上到下递减）
    y_position = page_height - margin - 60
    
    # 文本绘制函数
    def draw_text_block(title, content, start_y):
        # 绘制小标题
        c.setFont(chinese_font, 14)
        c.drawString(margin, start_y, title)
        start_y -= 20
        
        # 绘制内容
        c.setFont(chinese_font, 10)
        
        # 处理多行文本
        content = clean_html(content)
        lines = []
        
        # 手动切分行
        current_line = ""
        words = content.split()
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            # 检查是否超出页面宽度
            if c.stringWidth(test_line, chinese_font, 10) < (page_width - 2 * margin):
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # 如果没有成功拆分行，则按固定长度拆分
        if not lines:
            chunk_size = 80  # 每行大约80个字符
            lines = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        
        # 绘制文本行
        for line in lines:
            if start_y < margin:  # 如果到达页面底部，则添加新页面
                c.showPage()
                c.setFont(chinese_font, 10)
                start_y = page_height - margin
            
            c.drawString(margin, start_y, line)
            start_y -= 15
        
        return start_y - 15  # 返回下一部分的起始Y位置
    
    # 绘制原始文本
    y_position = draw_text_block("原始文本", history_item['original_text'], y_position)
    
    # 绘制核心声明
    y_position = draw_text_block("核心声明", history_item['claim'], y_position)
    
    # 获取判断结果对应的表情和中文
    verdict = history_item['verdict'].upper()
    if verdict == "TRUE":
        emoji = "✓"
        verdict_cn = "正确"
    elif verdict == "FALSE":
        emoji = "✗"
        verdict_cn = "错误"
    elif verdict == "PARTIALLY TRUE":
        emoji = "!"
        verdict_cn = "部分正确"
    else:
        emoji = "?"
        verdict_cn = "无法验证"
    
    # 绘制结论
    c.setFont(chinese_font, 14)
    if y_position < margin:  # 检查是否需要新页面
        c.showPage()
        c.setFont(chinese_font, 14)
        y_position = page_height - margin
    
    c.drawString(margin, y_position, f"结论: {emoji} {verdict_cn}")
    y_position -= 20
    
    # 绘制推理过程
    y_position = draw_text_block("推理过程", history_item['reasoning'], y_position)
    
    # 绘制证据来源
    c.setFont(chinese_font, 14)
    if y_position < margin:  # 检查是否需要新页面
        c.showPage()
        c.setFont(chinese_font, 14)
        y_position = page_height - margin
    
    c.drawString(margin, y_position, "证据来源")
    y_position -= 20
    
    # 绘制每条证据
    for j, chunk in enumerate(history_item['evidence']):
        c.setFont(chinese_font, 10)
        if y_position < margin:  # 检查是否需要新页面
            c.showPage()
            c.setFont(chinese_font, 10)
            y_position = page_height - margin
        
        c.drawString(margin, y_position, f"[{j+1}]:")
        y_position -= 15
        
        # 绘制证据文本
        text_lines = clean_html(chunk['text']).split('\n')
        for line in text_lines:
            if y_position < margin:  # 检查是否需要新页面
                c.showPage()
                c.setFont(chinese_font, 10)
                y_position = page_height - margin
            
            c.drawString(margin, y_position, line)
            y_position -= 15
        
        # 绘制来源
        if y_position < margin:  # 检查是否需要新页面
            c.showPage()
            c.setFont(chinese_font, 10)
            y_position = page_height - margin
        
        c.drawString(margin, y_position, f"来源: {clean_html(chunk['source'])}")
        y_position -= 15
        
        # 绘制相关性
        if 'similarity' in chunk and chunk['similarity'] is not None:
            if y_position < margin:  # 检查是否需要新页面
                c.showPage()
                c.setFont(chinese_font, 10)
                y_position = page_height - margin
            
            c.drawString(margin, y_position, f"相关性: {chunk['similarity']:.2f}")
            y_position -= 15
        
        y_position -= 10  # 证据之间的额外间距
    
    # 保存PDF
    c.save()
    
    # 重置缓冲区位置
    buffer.seek(0)
    
    # 返回PDF数据
    return buffer.getvalue()


def generate_pdf_with_template(history_item):
    """使用SimpleDocTemplate生成PDF（传统方法）"""
    buffer = BytesIO()
    
    # 创建PDF文档
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # 定义样式
    styles = getSampleStyleSheet()
    
    # 创建中文标题样式
    title_style = ParagraphStyle(
        'ChineseTitle',
        parent=styles['Title'],
        fontName=chinese_font,
        fontSize=18,
        alignment=1,  # 居中
        wordWrap='CJK'  # 关键：使用CJK单词换行模式
    )
    
    # 创建中文正文样式
    normal_style = ParagraphStyle(
        'ChineseNormal',
        parent=styles['Normal'],
        fontName=chinese_font,
        fontSize=10,
        leading=14,  # 行间距
        wordWrap='CJK'  # 关键：使用CJK单词换行模式
    )
    
    # 创建中文小标题样式
    heading_style = ParagraphStyle(
        'ChineseHeading',
        parent=styles['Heading2'],
        fontName=chinese_font,
        fontSize=14,
        wordWrap='CJK'  # 关键：使用CJK单词换行模式
    )
    
    # 获取判断结果对应的表情和中文
    verdict = history_item['verdict'].upper()
    if verdict == "TRUE":
        emoji = "✓"  # PDF中使用简单符号
        verdict_cn = "正确"
    elif verdict == "FALSE":
        emoji = "✗"
        verdict_cn = "错误"
    elif verdict == "PARTIALLY TRUE":
        emoji = "!"
        verdict_cn = "部分正确"
    else:
        emoji = "?"
        verdict_cn = "无法验证"
    
    # 报告内容列表
    content = []
    
    # 添加标题
    content.append(Paragraph("事实核查报告", title_style))
    content.append(Spacer(1, 12))
    
    # 添加生成时间
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content.append(Paragraph(f"生成时间: {current_time}", normal_style))
    content.append(Spacer(1, 12))
    
    # 添加原始文本
    content.append(Paragraph("原始文本", heading_style))
    content.append(Spacer(1, 6))
    
    # 使用安全的文本处理方式
    try:
        original_text = clean_html(history_item['original_text'])
        content.append(Paragraph(original_text, normal_style))
    except Exception as e:
        logger.error(f"处理原始文本时出错: {e}")
        content.append(Paragraph("无法显示原始文本", normal_style))
    
    content.append(Spacer(1, 12))
    
    # 添加核心声明
    content.append(Paragraph("核心声明", heading_style))
    content.append(Spacer(1, 6))
    try:
        claim_text = clean_html(history_item['claim'])
        content.append(Paragraph(claim_text, normal_style))
    except Exception as e:
        logger.error(f"处理核心声明时出错: {e}")
        content.append(Paragraph("无法显示核心声明", normal_style))
    
    content.append(Spacer(1, 12))
    
    # 添加判断结果
    content.append(Paragraph(f"结论: {emoji} {verdict_cn}", heading_style))
    content.append(Spacer(1, 12))
    
    # 添加推理过程
    content.append(Paragraph("推理过程", heading_style))
    content.append(Spacer(1, 6))
    try:
        reasoning_text = clean_html(history_item['reasoning'])
        content.append(Paragraph(reasoning_text, normal_style))
    except Exception as e:
        logger.error(f"处理推理过程时出错: {e}")
        content.append(Paragraph("无法显示推理过程", normal_style))
    
    content.append(Spacer(1, 12))
    
    # 添加证据来源
    content.append(Paragraph("证据来源", heading_style))
    content.append(Spacer(1, 6))
    
    for j, chunk in enumerate(history_item['evidence']):
        try:
            content.append(Paragraph(f"[{j+1}]:", normal_style))
            chunk_text = clean_html(chunk['text'])
            content.append(Paragraph(chunk_text, normal_style))
            source_text = clean_html(chunk['source'])
            content.append(Paragraph(f"来源: {source_text}", normal_style))
            if 'similarity' in chunk and chunk['similarity'] is not None:
                content.append(Paragraph(f"相关性: {chunk['similarity']:.2f}", normal_style))
            content.append(Spacer(1, 6))
        except Exception as e:
            logger.error(f"处理证据 {j+1} 时出错: {e}")
            content.append(Paragraph(f"无法显示证据 {j+1}", normal_style))
            content.append(Spacer(1, 6))
    
    # 构建PDF
    doc.build(content)
    
    # 重置缓冲区位置
    buffer.seek(0)
    
    # 返回PDF数据
    return buffer.getvalue()


def generate_simple_pdf(history_item):
    """生成一个简单的PDF，确保至少有一些内容可以下载"""
    logger.info("生成简单PDF作为最后的备选方案")
    buffer = BytesIO()
    
    # 创建Canvas
    c = canvas.Canvas(buffer, pagesize=A4)
    
    # 设置字体为Helvetica（内置字体，支持ASCII）
    c.setFont("Helvetica", 16)
    c.drawString(72, A4[1]-108, "Fact Check Report")
    
    c.setFont("Helvetica", 12)
    c.drawString(72, A4[1]-140, "Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # 获取判断结果
    verdict = history_item['verdict'].upper()
    if verdict == "TRUE":
        verdict_text = "TRUE"
    elif verdict == "FALSE":
        verdict_text = "FALSE"
    elif verdict == "PARTIALLY TRUE":
        verdict_text = "PARTIALLY TRUE"
    else:
        verdict_text = "UNVERIFIABLE"
    
    c.drawString(72, A4[1]-180, "Verdict: " + verdict_text)
    
    # 添加注释说明PDF有问题
    c.setFont("Helvetica", 10)
    c.drawString(72, 72, "Note: There was an issue generating the complete PDF with Chinese characters.")
    c.drawString(72, 58, "Please check the application interface for complete results.")
    
    c.save()
    
    # 重置缓冲区位置
    buffer.seek(0)
    
    # 返回PDF数据
    return buffer.getvalue()