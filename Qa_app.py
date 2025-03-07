import os
import faiss
import numpy as np
import pypdf
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from functools import wraps

# -------------------
# CONFIGURATION
# -------------------
load_dotenv()

# Authentication
API_KEY = os.getenv("API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("API_KEY is not set in environment or .env file")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in environment or .env file")

# Groq Client
client = Groq(api_key=GROQ_API_KEY)

# Embedding model and FAISS index
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
INDEX_PATH = "data/faiss_index.bin"
TEXT_STORE_PATH = "data/text_chunks.npy"

# Load FAISS index & stored text chunks
faiss_index = faiss.read_index(INDEX_PATH)
faq_texts = np.load(TEXT_STORE_PATH, allow_pickle=True)

# Path to the full document (used for full-document QA)
PDF_PATH = "/home/sharanya/Flask_app/UNIZH Profile.pdf"

# -------------------
# AUTH DECORATOR
# -------------------
def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({"message": "Token is missing!"}), 403
        if token != f"Bearer {API_KEY}":
            return jsonify({"message": "Invalid token!"}), 403
        return f(*args, **kwargs)
    return decorated_function

# -------------------
# TEXT EXTRACTION
# -------------------
def extract_text_from_pdf(pdf_path):
    """Extracts all text from a PDF."""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

full_document_text = extract_text_from_pdf(PDF_PATH)

# -------------------
# RETRIEVAL FUNCTION (FAISS)
# -------------------
def retrieve_relevant_chunk(query):
    """Finds the most relevant chunk using FAISS similarity search."""
    query_embedding = np.array([embedding_model.encode(query)])
    _, indices = faiss_index.search(query_embedding, 1)
    return faq_texts[indices[0][0]]

# -------------------
# QA FUNCTION (Groq)
# -------------------
def ask_groq(question, context):
    """Generates an answer using Groq LLM based on provided context."""
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": """You are an AI assistant answering questions **only** based on the provided official document. If the document does not contain relevant information, say: "The document does not contain this information." Do not make assumptions."""},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content.strip()

# -------------------
# FLASK APP
# -------------------
app = Flask(__name__)

@app.route('/ask', methods=['POST'])
@token_required
def answer_question():
    """Unified API to answer questions using either full document or chunk retrieval."""
    data = request.json
    question = data.get("question", "")
    mode = data.get("mode", "chunk")  # "full" or "chunk"

    if not question:
        return jsonify({"error": "Please provide a question"}), 400

    # if mode == "full":
    #     # Use entire document
    #     context = full_document_text
    # else:
    #     # Use chunk retrieval (default)
    #     context = retrieve_relevant_chunk(question)
    if len(full_document_text.split()) < 25000:  # Approx 25k words = 32k tokens
        context = full_document_text  # Full document for small files
    else:
        context = retrieve_relevant_chunk(question)  # Chunk retrieval for large files


    answer = ask_groq(question, context)

    return jsonify({
        "question": question,
        "answer": answer,
        "context": context[:500]  # Limit context length in response for readability
    })

# -------------------
# START SERVER
# -------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
