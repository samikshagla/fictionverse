import os
import uuid
import shutil
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from groq import Groq
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("SUPABASE_DB_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

HF_EMBEDDING_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

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

groq_client = Groq(api_key=GROQ_API_KEY)

class QuestionRequest(BaseModel):
    book_id: str
    current_progress: float
    question: str

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Fetches 384-dim embeddings from Hugging Face Inference API."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(
        HF_EMBEDDING_URL,
        headers=headers,
        json={"inputs": texts, "options": {"wait_for_model": True}},
        timeout=60
    )
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Hugging Face API error: {response.text}"
        )
    return response.json()

def extract_paragraphs(epub_path: str) -> list[str]:
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
    book_id = f"book_{uuid.uuid4().hex[:8]}"
    temp_path = f"temp_{book_id}.epub"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        paragraphs = extract_paragraphs(temp_path)
        total_chunks = len(paragraphs)
        if total_chunks == 0:
            raise HTTPException(status_code=400, detail="Could not extract text from EPUB.")

        session = Session()
        try:
            # Batch embed 50 paragraphs per API call for speed
            batch_size = 50
            for start in range(0, total_chunks, batch_size):
                batch_texts = paragraphs[start:start + batch_size]
                batch_vectors = get_embeddings(batch_texts)

                for offset, (para_text, vector) in enumerate(zip(batch_texts, batch_vectors)):
                    progress_marker = (start + offset) / total_chunks
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
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/ask")
def ask_story_question(req: QuestionRequest):
    session = Session()
    try:
        question_vector = get_embeddings([req.question])[0]

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

        context_texts = [
            f"[Progress {round(row.progress_marker * 100)}%]: {row.content}"
            for row in result
        ]
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