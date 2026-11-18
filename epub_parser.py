import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import json
import os

def extract_text(html_content):
    """Parse HTML and extract clean text."""
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator='\n')
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        
        if i + chunk_size >= len(words):
            break
            
    return chunks

def process_epub(file_path, book_id, user_id):
    """Read EPUB, extract chapters, chunk text, and attach metadata."""
    book = epub.read_epub(file_path)
    
    chapter_number = 1
    all_chunks = []
    
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        html_content = item.get_content()
        raw_text = extract_text(html_content)
        
        # Skip empty or very short sections (like title pages)
        if len(raw_text.split()) < 50:
            continue
            
        text_chunks = chunk_text(raw_text, chunk_size=500, overlap=50)
        
        for chunk_index, chunk_content in enumerate(text_chunks):
            chunk_data = {
                "book_id": book_id,
                "user_id": user_id,
                "chapter_number": chapter_number,
                "chunk_index": chunk_index,
                "content": chunk_content
            }
            all_chunks.append(chunk_data)
            
        chapter_number += 1
        
    return all_chunks

if __name__ == "__main__":
    test_file = 'test_book.epub'
    output_file = 'book_chunks.json'
    
    if os.path.exists(test_file):
        print(f"Processing {test_file}...")
        chunks = process_epub(test_file, "book_001", "user_123")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=4, ensure_ascii=False)
            
        print(f"Success! Generated {len(chunks)} chunks.")
        print(f"Output saved to {output_file}")
    else:
        print(f"Error: Please place an EPUB file named '{test_file}' in this directory.")