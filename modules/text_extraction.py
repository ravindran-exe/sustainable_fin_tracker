import pdfplumber
import pytesseract
from PIL import Image
import tempfile
import re
import streamlit as st

def extract_text_from_pdf(file):
    """Enhanced PDF text extraction with better formatting preservation"""
    text = ""
    tables_data = []
    
    with pdfplumber.open(file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Extract text
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text + "\n"
            
            # Extract tables
            tables = page.extract_tables()
            if tables:
                for table_num, table in enumerate(tables):
                    tables_data.append({
                        'page': page_num + 1,
                        'table': table_num + 1,
                        'data': table
                    })
    
    return text, tables_data

def extract_text_with_ocr(file):
    """Enhanced OCR extraction with preprocessing"""
    full_text = ""
    
    with pdfplumber.open(file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Convert to image with higher resolution
            img = page.to_image(resolution=400)
            
            # Save temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                img.save(tmp_img.name, format="PNG")
                pil_img = Image.open(tmp_img.name)
                
                # OCR with better configuration
                custom_config = r'--oem 3 --psm 6'
                ocr_text = pytesseract.image_to_string(pil_img, config=custom_config)
                
                full_text += f"\n--- Page {page_num + 1} (OCR) ---\n"
                full_text += ocr_text + "\n"
    
    return full_text

def smart_chunk_text(text, max_chunk_size=1500, overlap=200):
    """Smart chunking that preserves context"""
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Add overlap
                words = current_chunk.split()
                if len(words) > overlap // 10:
                    current_chunk = " ".join(words[-(overlap // 10):]) + " " + sentence + ". "
                else:
                    current_chunk = sentence + ". "
            else:
                current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks