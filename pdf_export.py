import io
from fpdf import FPDF
from datetime import datetime

class PDF(FPDF):
    """为虚假新闻检测结果定制的PDF生成类"""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_font('simhei', '', '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', uni=True)
        self.add_font('simsun', '', '/usr/share/fonts/truetype/arphic/uming.ttc', uni=True)
        
    def header(self):
        # 页眉
        self.set_font('simhei', '', 12)
        self.cell(0, 10, 'AI虚假新闻检测报告', 0, 1, 'C')
        self.line(10, 18, 200, 18)
        self.ln(10)
        
    def footer(self):
        # 页脚
        self.set_y(-15)
        self.set_font('simhei', '', 8)
        self.cell(0, 10, f'第 {self.page_no()} 页', 0, 0, 'C')
        
    def chapter_title(self, title):
        # 章节标题
        self.set_font('simhei', '', 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 9, title, 0, 1, 'L', 1)
        self.ln(5)
        
    def chapter_body(self, text):
        # 正文
        self.set_font('simsun', '', 10)
        self.multi_cell(0, 5, text)
        self.ln()

def generate_fact_check_pdf(history_item):
    """
    生成事实核查结果的PDF文件
    
    Args:
        history_item: 历史记录数据
        
    Returns:
        PDF文件的二进制数据
    """
    pdf = PDF()
    pdf.add_page()
    
    # 添加报告时间
    report_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
    pdf.set_font('simhei', '', 10)
    pdf.cell(0, 10, f'生成时间: {report_time}', 0, 1, 'R')
    pdf.ln(5)
    
    # 原始文本
    pdf.chapter_title("原始新闻")
    pdf.chapter_body(history_item['original_text'])
    
    # 核心声明
    pdf.chapter_title("核心声明")
    pdf.chapter_body(history_item['claim'])
    
    # 判断结果
    verdict = history_item['verdict'].upper()
    if verdict == "TRUE":
        verdict_cn = "正确"
    elif verdict == "FALSE":
        verdict_cn = "错误"
    elif verdict == "PARTIALLY TRUE":
        verdict_cn = "部分正确"
    else:
        verdict_cn = "无法验证"
    
    pdf.chapter_title(f"结论: {verdict_cn}")
    
    # 推理过程
    pdf.chapter_title("推理过程")
    pdf.chapter_body(history_item['reasoning'])
    
    # 证据来源
    pdf.chapter_title("证据来源")
    for i, chunk in enumerate(history_item['evidence']):
        pdf.set_font('simhei', '', 11)
        pdf.cell(0, 6, f"证据 [{i+1}]:", 0, 1)
        
        pdf.set_font('simsun', '', 10)
        pdf.multi_cell(0, 5, chunk['text'])
        
        pdf.set_font('simsun', '', 9)
        pdf.cell(0, 5, f"来源: {chunk['source']}", 0, 1)
        
        if 'similarity' in chunk and chunk['similarity'] is not None:
            pdf.cell(0, 5, f"相关性: {chunk['similarity']:.2f}", 0, 1)
        
        pdf.ln(3)
    
    # 转换为二进制
    pdf_bytes = io.BytesIO()
    pdf.output(pdf_bytes)
    return pdf_bytes.getvalue()