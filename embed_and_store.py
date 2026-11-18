import json
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# 1. Database Connection (Ensure your Supabase URL is correct!)
DB_URL = "postgresql+psycopg2://postgres:57%256%247%2BU4tNi%40Xc@db.ekdkhndqzaacimuvhchb.supabase.co:5432/postgres"
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)

print("Loading AI model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_paragraphs(epub_path):
    """Extracts raw text paragraph by paragraph instead of by chapter."""
    book = epub.read_epub(epub_path)
    paragraphs = []
    
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_body_content(), 'html.parser')
            # Extract every single paragraph tag
            for p in soup.find_all('p'):
                text_content = p.get_text(strip=True)
                if len(text_content) > 30: # Ignore tiny blank lines or page numbers
                    paragraphs.append(text_content)
    return paragraphs

def main():
    print("Extracting granular paragraphs from EPUB...")
    paragraphs = extract_paragraphs("test_book.epub")
    total_chunks = len(paragraphs)
    print(f"Found {total_chunks} valid paragraphs.")

    session = Session()
    try:
        print("Rebuilding database table for granular tracking...")
        # Drop the old chapter-based table
        session.execute(text("DROP TABLE IF EXISTS story_chunks;"))
        
        # Create the new percentage-based table
        session.execute(text("""
            CREATE TABLE story_chunks (
                id SERIAL PRIMARY KEY,
                book_id VARCHAR(50),
                progress_marker FLOAT,
                content TEXT,
                embedding vector(384)
            );
        """))
        session.commit()

        print("Embedding and storing chunks. This may take a minute...")
        
        for i, para_text in enumerate(paragraphs):
            # Calculate exactly where this paragraph sits in the book (0.0 to 1.0)
            progress_marker = i / total_chunks 
            
            vector = model.encode(para_text).tolist()
            
            session.execute(text("""
                INSERT INTO story_chunks (book_id, progress_marker, content, embedding)
                VALUES (:book_id, :progress_marker, :content, :embedding)
            """), {
                "book_id": "book_001",
                "progress_marker": progress_marker,
                "content": para_text,
                "embedding": str(vector)
            })
            
            # Print a progress indicator every 100 paragraphs
            if i % 100 == 0:
                print(f"Stored {i}/{total_chunks} paragraphs...")

        session.commit()
        print(f"Success! {total_chunks} granular chunks securely stored in Supabase.")

    finally:
        session.close()

if __name__ == "__main__":
    main()