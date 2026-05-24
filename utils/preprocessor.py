import re
from io import BytesIO
from bs4 import BeautifulSoup
from pypdf import PdfReader
from docx import Document

def clean_text(text: str) -> str:
    """Cleans text by normalizing whitespace and removing non-printable characters."""
    if not text:
        return ""
    # Normalize whitespaces (spaces, tabs, newlines)
    text = re.sub(r'\s+', ' ', text)
    # Remove control characters or weird symbols
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50, metadata_source: dict = None) -> list[dict]:
    """
    Splits text into chunks of specified character count while attempting to respect
    word boundaries. Returns a list of dictionaries with page_content and metadata.
    """
    if not text or len(text.strip()) == 0:
        return []
    
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 10  # Ensure overlap is smaller than chunk size
        
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # If we are not at the end of the text, look back for a space/newline to avoid cutting a word.
        if end < text_len:
            # Look back up to 20% of the chunk size
            lookback_limit = max(start, end - int(chunk_size * 0.2))
            boundary = -1
            for idx in range(end - 1, lookback_limit - 1, -1):
                if text[idx] in ['\n', ' ', '\r', '\t']:
                    boundary = idx
                    break
            if boundary != -1:
                end = boundary + 1  # Include the boundary character (space)
        
        chunk_content = text[start:end].strip()
        
        if chunk_content:
            chunk_metadata = (metadata_source or {}).copy()
            chunk_metadata.update({
                "char_count": len(chunk_content),
                "word_count": len(chunk_content.split()),
            })
            chunks.append({
                "page_content": chunk_content,
                "metadata": chunk_metadata
            })
            
        start = end - chunk_overlap
        # Prevent infinite loops in case boundary search doesn't progress
        if start >= end:
            start = end
        if end == text_len:
            break
            
    # Add index IDs to the chunks
    for idx, chunk in enumerate(chunks):
        chunk["metadata"]["chunk_index"] = idx
        
    return chunks

def parse_pdf(file_bytes: bytes) -> str:
    """Extracts text from a PDF file."""
    pdf_file = BytesIO(file_bytes)
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def parse_docx(file_bytes: bytes) -> str:
    """Extracts text from a Word document (.docx)."""
    docx_file = BytesIO(file_bytes)
    doc = Document(docx_file)
    text = ""
    for para in doc.paragraphs:
        if para.text:
            text += para.text + "\n"
    return text

def parse_txt(file_bytes: bytes) -> str:
    """Extracts text from a plain text file."""
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # Fallback to latin-1
        return file_bytes.decode("latin-1")

def parse_html(html_content: str) -> str:
    """Extracts text from HTML and strips templates/tags using BeautifulSoup."""
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove script and style elements
    for script in soup(["script", "style", "meta", "noscript", "header", "footer"]):
        script.decompose()
        
    # Get text
    text = soup.get_text()
    
    # Break into lines and remove leading/trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return text
