from flask import Flask, render_template, request, jsonify
import os
from google import genai
from google.genai import types
import faiss
import numpy as np
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# Configure GenAI Client
api_key = os.environ.get("GOOGLE_API_KEY")
client = None
if not api_key:
    print("Warning: GOOGLE_API_KEY not found in environment.")
else:
    client = genai.Client(api_key=api_key)

app = Flask(__name__)

# Global state
vector_index = None
chunks = []
# For this simple stateless example, we won't maintain a complex chat session object 
# but simply append context to the prompt each time.
chat_history = [] 

def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - chunk_overlap)
        
    return chunks

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/load', methods=['POST'])
def load_pdf():
    global vector_index, chunks, chat_history
    data = request.json
    pdf_path = data.get('path')
    
    if not pdf_path:
        return jsonify({"error": "Path is required"}), 400
        
    if not os.path.exists(pdf_path):
        return jsonify({"error": f"File not found at {pdf_path}"}), 404

    try:
        # Load PDF
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
            
        # Split
        chunks = get_text_chunks(text)
        
        if not chunks:
             return jsonify({"error": "No text extracted from PDF."}), 400

        # Embed
        model = 'text-embedding-004' # Newer model, check availability. Or use "models/text-embedding-004"
        embeddings = []
        
        # Batch process
        batch_size = 50 
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            # google-genai SDK call
            response = client.models.embed_content(
                model=model,
                contents=batch,
            )
            # Response has .embeddings attribute which is a list of Embedding objects
            batch_embeddings = [e.values for e in response.embeddings]
            embeddings.extend(batch_embeddings)

        # Create FAISS index
        if embeddings:
            dimension = len(embeddings[0])
            vector_index = faiss.IndexFlatL2(dimension)
            vector_index.add(np.array(embeddings).astype('float32'))
        
        chat_history = [] # Reset history
        
        return jsonify({"message": f"PDF Loaded! Processed {len(chunks)} chunks."})
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    global vector_index, chunks, chat_history
    if vector_index is None:
        return jsonify({"error": "Please load a PDF first"}), 400
        
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        # Embed Query
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=query,
        )
        query_embedding = response.embeddings[0].values
        
        # Search Vector Store
        k = 3
        D, I = vector_index.search(np.array([query_embedding]).astype('float32'), k)
        
        retrieved_context = ""
        for idx in I[0]:
            if idx < len(chunks):
                retrieved_context += chunks[idx] + "\n\n"
        
        # Construct Prompt
        prompt = (
            "You are a helpful assistant. Answer the user's question based ONLY on the following context.\n"
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{retrieved_context}\n\n"
            f"Question: {query}"
        )
        
        # Send to Gemini
        model_name = "gemini-2.0-flash-lite"
        
        # We can pass simple history if we wanted, but for strict RAG we mostly care about the current prompt + context.
        # To maintain chat illusion, we could append previous turns to 'contents' list.
        # For simplicity here, we just send the single prompt.
        
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        
        answer = response.text
        
        return jsonify({"answer": answer})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
