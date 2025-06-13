import json
import base64
import asyncio
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

OPENAI_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjIwMDI0ODZAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.3Z-upltp4wPclJfjNf0C-8YLsqu2HExG26AMxIGJYcI"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_COMPLETION_MODEL = "gpt-4o-mini"  # or gpt-3.5-turbo

app = FastAPI()

# ---------------------------
# Pydantic models
# ---------------------------

class Link(BaseModel):
    url: str
    text: str

class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64 image optional

class AnswerResponse(BaseModel):
    answer: str
    links: List[Link]

# ---------------------------
# Load and preprocess data
# ---------------------------

def load_json_file(filename: str):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

content_data = load_json_file("cleaned-content.json")  # Already a list of dicts
discourse_data = load_json_file("cleaned-discourse.json")  # List of dicts

# ---------------------------
# Chunking text for embedding
# ---------------------------

def chunk_text(text: str, max_chars: int = 2000) -> List[str]:
    # Simple chunk by approx max_chars
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # Try to break at last newline before end
        chunk = text[start:end]
        last_newline = chunk.rfind("\n")
        if last_newline != -1 and end != len(text):
            end = start + last_newline
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]

# ---------------------------
# Async OpenAI API calls with httpx
# ---------------------------

async def get_embedding_async(text: str) -> List[float]:
    url = "https://aipipe.org/openrouter/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    json_data = {
        "model": EMBEDDING_MODEL,
        "input": text
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, headers=headers, json=json_data)
    if response.status_code != 200:
        raise RuntimeError(f"OpenAI Embedding API error: {response.text}")
    data = response.json()
    return data["data"][0]["embedding"]

async def openai_chat_completion_async(system_prompt: str, user_prompt: str) -> str:
    url = "https://aipipe.org/openrouter/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    json_data = {
        "model": CHAT_COMPLETION_MODEL,
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.3
    }
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(url, headers=headers, json=json_data)
    if response.status_code != 200:
        raise RuntimeError(f"OpenAI Chat Completion API error: {response.text}")
    data = response.json()
    return data["choices"][0]["message"]["content"]

# ---------------------------
# Document chunk structure
# ---------------------------

class DocumentChunk:
    def __init__(self, source_type: str, source_id: str, text: str, url: Optional[str]):
        self.source_type = source_type  # "content" or "discourse"
        self.source_id = source_id
        self.text = text
        self.url = url
        self.embedding = None  # to be filled later

chunks: List[DocumentChunk] = []

async def prepare_chunks_and_embeddings():
    global chunks
    chunks = []

    # Process content.json
    for doc in content_data:
        doc_id = doc.get("id", "")
        content = doc.get("content", "")
        url = None  # no direct URL in content.json
        for ctext in chunk_text(content):
            chunks.append(DocumentChunk("content", doc_id, ctext, url))

    # Process discourse.json
    for post in discourse_data:
        post_id = str(post.get("id", ""))
        post_url = post.get("post_url", "")
        full_url = f"https://discourse.onlinedegree.iitm.ac.in{post_url}" if post_url else None
        content = post.get("content", "")
        for ctext in chunk_text(content):
            chunks.append(DocumentChunk("discourse", post_id, ctext, full_url))

    # Generate embeddings concurrently
    async def embed_chunk(chunk: DocumentChunk):
        chunk.embedding = await get_embedding_async(chunk.text)

    await asyncio.gather(*(embed_chunk(c) for c in chunks))

# Run once on startup
@app.on_event("startup")
async def startup_event():
    print("Loading and embedding documents, please wait...")
    await prepare_chunks_and_embeddings()
    print(f"Embedded {len(chunks)} chunks.")

# ---------------------------
# Cosine similarity search (brute force)
# ---------------------------

def cosine_similarity(vec1, vec2):
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def semantic_search(query_embedding: List[float], top_k: int = 5) -> List[DocumentChunk]:
    similarities = []
    for chunk in chunks:
        if chunk.embedding is None:
            continue
        sim = cosine_similarity(query_embedding, chunk.embedding)
        similarities.append((sim, chunk))
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in similarities[:top_k]]

# ---------------------------
# FastAPI endpoint
# ---------------------------

@app.post("/api/", response_model=AnswerResponse)
async def answer_question(req: QuestionRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Optional: decode image if provided (not used here)
    if req.image:
        try:
            _ = base64.b64decode(req.image.split(",")[-1])
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

    # Embed query
    try:
        query_emb = await get_embedding_async(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")

    # Search top relevant chunks
    relevant_chunks = semantic_search(query_emb, top_k=5)

    # Prepare context and links
    context_texts = [chunk.text for chunk in relevant_chunks]
    links = []
    for chunk in relevant_chunks:
        if chunk.url:
            links.append({"url": chunk.url, "text": chunk.text[:100].replace("\n", " ") + "..."})

    context_str = "\n\n---\n\n".join(context_texts)

    system_prompt = (
        "You are a helpful virtual teaching assistant for the Tools in Data Science course at IIT Madras. "
        "Use the following course content and discussion excerpts to answer the student's question."
    )
    user_prompt = f"Context:\n{context_str}\n\nQuestion:\n{question}\n\nAnswer clearly and provide relevant links if possible."

    try:
        answer = await openai_chat_completion_async(system_prompt, user_prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

    return AnswerResponse(answer=answer, links=links)

# ---------------------------
# Run with:
# uvicorn main:app --reload
# ---------------------------
