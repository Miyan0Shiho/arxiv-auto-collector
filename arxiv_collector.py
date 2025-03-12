import arxiv
import datetime
import os
import requests
from PyPDF2 import PdfReader
from urllib.parse import urlparse
import time

# 配置常量
TEMP_PDF_PATH = os.path.join(os.getcwd(), "temp.pdf")  # 适配GitHub Actions环境

def sanitize_filename(title):
    """生成安全的文件名"""
    return "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in title)[:50].strip()

def main():
    # 计算日期范围
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=7)
    
    # 配置arXiv搜索
    search = arxiv.Search(
        query=f'rag AND submittedDate:[{start_date.strftime("%Y%m%d")} TO {end_date.strftime("%Y%m%d")}]',
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    # 初始化客户端
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)
    
    # 处理每篇论文
    for result in client.results(search):
        try:
            # 生成周目录
            published_date = result.published
            year, week_num, _ = published_date.isocalendar()
            folder_name = f"{year}-W{week_num:02d}"
            os.makedirs(folder_name, exist_ok=True)
            
            # 下载PDF
            pdf_url = result.pdf_url
            response = requests.get(pdf_url, timeout=10)
            response.raise_for_status()
            
            # 保存临时PDF
            with open(TEMP_PDF_PATH, "wb") as f:
                f.write(response.content)
            
            # 解析PDF文本
            text_content = []
            with open(TEMP_PDF_PATH, "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    if page_text := page.extract_text():
                        text_content.append(page_text)
            
            # 生成Markdown内容
            md_content = (
                f"# {result.title}\n\n"
                f"**Authors**: {', '.join([a.name for a in result.authors])}\n\n"
                f"**Published**: {published_date.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"**PDF URL**: [{pdf_url}]({pdf_url})\n\n"
                "## Abstract\n"
                f"{result.summary}\n\n"
                "## Full Text\n"
                f"\n\n<!-- PDF content starts -->\n\n{'\n\n'.join(text_content)}"
            )
            
            # 保存Markdown文件
            safe_title = sanitize_filename(result.title)
            md_filename = os.path.join(folder_name, f"{safe_title}.md")
            
            with open(md_filename, "w", encoding="utf-8") as f:
                f.write(md_content)
            
            print(f"✅ 成功保存：{md_filename}")
            
        except Exception as e:
            print(f"❌ 处理论文《{result.title}》失败：{str(e)}")
            time.sleep(5)  # 错误后暂停
        finally:
            if os.path.exists(TEMP_PDF_PATH):
                os.remove(TEMP_PDF_PATH)  # 清理临时文件

if __name__ == "__main__":
    main()
