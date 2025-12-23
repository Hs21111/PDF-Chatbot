from flask import Flask, render_template, request, jsonify
import os
import getpass
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

# Ensure API Key
if "GOOGLE_API_KEY" not in os.environ:
    # In a real web app, we might ask via UI, but for this local demo we assume env or console input
    print("GOOGLE_API_KEY not found in env. Please set it in .env or environment.")

app = Flask(__name__)

# Global state
vectorstore = None
chat_history = ChatMessageHistory()
rag_chain = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/load', methods=['POST'])
def load_pdf():
    global vectorstore, rag_chain, chat_history
    data = request.json
    pdf_path = data.get('path')
    
    if not pdf_path:
        return jsonify({"error": "Path is required"}), 400
        
    if not os.path.exists(pdf_path):
        return jsonify({"error": f"File not found at {pdf_path}"}), 404

    try:
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # Split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Embed
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        
        # Setup Chain
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
        retriever = vectorstore.as_retriever()
        
        # Contextualize prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # Answer prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        chat_history = ChatMessageHistory() # Reset history
        
        return jsonify({"message": "PDF Loaded & Processed Successfully!"})
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    global rag_chain, chat_history
    if not rag_chain:
        return jsonify({"error": "Please load a PDF first"}), 400
        
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        response = rag_chain.invoke({"input": query, "chat_history": chat_history.messages})
        answer = response["answer"]
        
        # Update history
        chat_history.add_user_message(query)
        chat_history.add_ai_message(answer)
        
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
