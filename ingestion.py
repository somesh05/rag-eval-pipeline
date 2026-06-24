
import os
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

def load_pdf(file_path: str) -> str:
    reader= PdfReader(file_path)
    text_parts= []
    for page_num, page in enumerate(reader.pages):
        page_text=page.extract_text() or ""
        text_parts.append(page_text)
    return "\n".join(text_parts)

def load_txt(file_path:str) -> str:
    with open (file_path, "r", encoding="utf-8") as f:
        return f.read()
    
def load_document(file_path:str) -> str:
    ext=os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext in (".txt", ".md"):
        return load_txt(file_path)
    else:
        raise ValueError (f"Unsupported file type: {ext}")
    
def chunk_document(text:str, source_name:str, chunk_size:int =500, chunk_overlap:int=50) -> List[Dict]:
    splitter= RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators= ["\n\n", "\n", ". ", ", ", ""]
    )
    raw_chunks= splitter.split_text(text)
    chunks= []

    for i, chunk_text in enumerate(raw_chunks):
        chunks.append({
            "id": f"{source_name}_chunk_{i}",
            "text": chunk_text,
            "source": source_name,
            "chunk_index": i,
        })

    return chunks

def process_uploaded_files(file_paths:List[str], chunk_size:int=500, chunk_overlap:int=50) -> List[Dict]:
    all_chunks= []
    for file_path in file_paths:
        source_name= os.path.basename(file_path)
        try:
            text= load_document(file_path)
            chunks= chunk_document(text, source_name, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
            print (f"Loaded {len(chunks)} chunks from {source_name}")
        except Exception as e:
            print (f"Error processing {file_path}: {e}")
    return all_chunks
            
