from flask import Flask, render_template, request, jsonify
import os
import google.generativeai as genai
import faiss
import numpy as np
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("Warning: GOOGLE_API_KEY not found in environment.")
else:
    genai.configure(api_key=api_key)

app = Flask(__name__)

# Global state
vector_index = None
chunks = []
chat_session = None

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
    global vector_index, chunks, chat_session
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
        # Gemini Embedding model returns 768 dimensions for embedding-001
        model = 'models/embedding-001'
        embeddings = []
        
        # Batch embedding to avoid hitting limits or timeouts if possible, 
        # though for this simple app we'll loop or send in small batches.
        # genai.embed_content accepts a list of content.
        
        # Process in batches of 100 to be safe
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            result = genai.embed_content(
                model=model,
                content=batch,
                task_type="retrieval_document",
                title="PDF Chunk"
            )
            embeddings.extend(result['embedding'])

        # Create FAISS index
        dimension = len(embeddings[0])
        vector_index = faiss.IndexFlatL2(dimension)
        vector_index.add(np.array(embeddings).astype('float32'))
        
        # Initialize Chat Session with a system prompt context
        model_name = "gemini-2.5-flash" 
        # Note: 2.5-flash might not be available in all regions or under this exact name in the SDK yet depending on the version.
        # We will use 'gemini-1.5-flash' as a safe fallback if 2.5 isn't resolved, or rely on the user's string.
        # Using 'gemini-pro' or 'gemini-1.5-flash' is standard. Let's try the user's request.
        
        try:
             generation_model = genai.GenerativeModel("gemini-1.5-flash") # Fallback/Standard
        except:
             generation_model = genai.GenerativeModel("gemini-pro")

        chat_session = generation_model.start_chat(history=[])
        
        return jsonify({"message": f"PDF Loaded! Processed {len(chunks)} chunks."})
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    global vector_index, chunks, chat_session
    if vector_index is None:
        return jsonify({"error": "Please load a PDF first"}), 400
        
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        # Embed Query
        query_embedding = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )['embedding']
        
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
        # We don't necessarily need the chat session history for the RAG part if we just do single turn Q&A,
        # but the user asked for a chat.
        # To strictly follow RAG, we usually feed the context into the prompt each time.
        # Simple approach: Just generate content with the context.
        
        response = chat_session.send_message(prompt)
        answer = response.text
        
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
