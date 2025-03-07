import os
import pypdf
import groq
from flask import Flask, request, jsonify
from functools import wraps
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key for header authentication and Groq API key from environment variables
API_KEY = os.getenv("API_KEY")  # Your header API key for authentication
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment or .env file")

if not API_KEY:
    raise ValueError("API_KEY is not set in the environment or .env file")

# Groq client setup
client = groq.Client(api_key=GROQ_API_KEY)

# Decorator to check for API key in headers
def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')  # Get token from Authorization header
        if not token:
            return jsonify({"message": "Token is missing!"}), 403
        if token != f"Bearer {API_KEY}":  # Check if the token matches the API key
            return jsonify({"message": "Invalid token!"}), 403
        return f(*args, **kwargs)
    return decorated_function

# Load and extract text from the PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Path to the uploaded PDF
pdf_path = "/home/sharanya/Flask_app/UNIZH Profile.pdf"
document_text = extract_text_from_pdf(pdf_path)

# Function to get an answer from Groq
def ask_groq(question, context):
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",  # Change this based on available models in Groq
        messages=[
            {"role": "system", "content": "You are a helpful assistant answering questions based on the provided document."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content.strip()

# Flask API for QA
app = Flask(__name__)

@app.route('/ask', methods=['POST'])
@token_required  # Apply the API key check to this route
def answer_question():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Please provide a question"}), 400
    
    answer = ask_groq(question, document_text)
    return jsonify({"question": question, "answer": answer})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
