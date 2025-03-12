# arxiv_collector.py
import arxiv
import datetime
import os
import requests
from PyPDF2 import PdfReader
import time

TEMP_PDF_PATH = os.path.join(os.getcwd(), "temp.pdf")

def sanitize_filename(title):
    return "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in title)[:50].strip()

def main():
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=7)
    
    search = arxiv.Search(
        query=f'rag AND submittedDate:[{start_date.strftime("%Y%m%d")} TO {end_date.strftime("%Y%m%d")}]',
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)
    
    for result in client.results(search):
        try:
            published_date = result.published
            year, week_num, _ = published_date.isocalendar()
            folder_name = f"{year}-W{week_num:02d}"
            os.makedirs(folder_name, exist_ok=True)
            
            pdf_url = result.pdf_url
            response = requests.get(pdf_url, timeout=10)
            response.raise_for_status()
            
            with open(TEMP_PDF_PATH, "wb") as f:
                f.write(response.content)
            
            text_content = []
            with open(TEMP_PDF_PATH, "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    if page_text := page.extract_text():
                        text_content.append(page_text)
            
            # 修复点：将换行符连接移到f-string外部
            full_text_separator = '\n\n'
            full_text = full_text_separator.join(text_content)
            
            md_content = (
                f"# {result.title}\n\n"
                f"**Authors**: {', '.join([a.name for a in result.authors])}\n\n"
                f"**Published**: {published_date.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"**PDF URL**: [{pdf_url}]({pdf_url})\n\n"
                "## Abstract\n"
                f"{result.summary}\n\n"
                "## Full Text\n"
                f"\n\n<!-- PDF content starts -->\n\n{full_text}"
            )
            
            safe_title = sanitize_filename(result.title)
            md_filename = os.path.join(folder_name, f"{safe_title}.md")
            
            with open(md_filename, "w", encoding="utf-8") as f:
                f.write(md_content)
            
            print(f"✅ 成功保存：{md_filename}")
            
        except Exception as e:
            print(f"❌ 处理论文《{result.title}》失败：{str(e)}")
            time.sleep(5)
        finally:
            if os.path.exists(TEMP_PDF_PATH):
                os.remove(TEMP_PDF_PATH)

if __name__ == "__main__":
    main()
