import os
import uuid
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from groq import Groq
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load hidden variables from .env file
load_dotenv()

# Fetch keys securely from environment
DB_URL = os.getenv("SUPABASE_DB_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)

app = FastAPI(title="FictionVerse BYOB API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=GROQ_API_KEY)

class QuestionRequest(BaseModel):
    book_id: str
    current_progress: float 
    question: str

def extract_paragraphs(epub_path):
    book = epub.read_epub(epub_path)
    paragraphs = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_body_content(), 'html.parser')
            for p in soup.find_all('p'):
                text_content = p.get_text(strip=True)
                if len(text_content) > 30: 
                    paragraphs.append(text_content)
    return paragraphs

@app.post("/upload")
async def upload_and_process_book(file: UploadFile = File(...)):
    # Generate a unique ID for this specific book
    book_id = f"book_{uuid.uuid4().hex[:8]}"
    temp_path = f"temp_{book_id}.epub"

    # Save the file temporarily
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Extract and embed the text just like your old script did
        paragraphs = extract_paragraphs(temp_path)
        total_chunks = len(paragraphs)
        if total_chunks == 0:
            raise HTTPException(status_code=400, detail="Could not extract text from EPUB.")

        session = Session()
        try:
            for i, para_text in enumerate(paragraphs):
                progress_marker = i / total_chunks 
                vector = model.encode(para_text).tolist()
                
                session.execute(text("""
                    INSERT INTO story_chunks (book_id, progress_marker, content, embedding)
                    VALUES (:book_id, :progress_marker, :content, :embedding)
                """), {
                    "book_id": book_id,
                    "progress_marker": progress_marker,
                    "content": para_text,
                    "embedding": str(vector)
                })
            session.commit()
        finally:
            session.close()

        return {"book_id": book_id, "message": f"Successfully processed {total_chunks} paragraphs."}
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/ask")
def ask_story_question(req: QuestionRequest):
    session = Session()
    try:
        question_vector = model.encode(req.question).tolist()
        
        query = text("""
            SELECT progress_marker, content, 1 - (embedding <=> :vector) AS similarity
            FROM story_chunks
            WHERE book_id = :book_id 
              AND progress_marker <= :current_progress
            ORDER BY embedding <=> :vector
            LIMIT 3;
        """)
        
        result = session.execute(query, {
            "vector": str(question_vector),
            "book_id": req.book_id,
            "current_progress": req.current_progress
        }).fetchall()
        
        if not result:
            raise HTTPException(status_code=404, detail="No relevant context found.")

        context_texts = []
        for row in result:
            context_texts.append(f"[Progress {round(row.progress_marker * 100)}%]: {row.content}")
            
        compiled_context = "\n\n".join(context_texts)

        user_prompt = f"""You are a strict reading companion. You only know the information in the Story Context below.

Story Context:
{compiled_context}

User Question: {req.question}

CRITICAL INSTRUCTION: Analyze the Story Context above. Does the context contain the details to answer the User Question? 
If NO, you are strictly forbidden from answering from your memory. You MUST reply EXACTLY with this phrase and nothing else: "I don't want to spoil anything, but you'll have to keep reading to find out!"
If YES, answer using ONLY the Story Context."""

        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": user_prompt}],
            model="openai/gpt-oss-120b", 
            temperature=0.0
        )

        return {
            "question": req.question,
            "answer": chat_completion.choices[0].message.content,
            "retrieved_context": context_texts
        }
    finally:
        session.close()