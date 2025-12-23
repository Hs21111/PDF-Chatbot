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

def get_api_key():
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API Key: ")

def load_and_process_pdf(pdf_path):
    print(f"Loading PDF from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    print("Creating vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

def main():
    get_api_key()
    
    pdf_path = input("Enter the path to the PDF file: ").strip()
    if not os.path.exists(pdf_path):
        print("File not found.")
        return

    try:
        vectorstore = load_and_process_pdf(pdf_path)
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return

    retriever = vectorstore.as_retriever()
    
    # LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

    # Contextualize question
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

    # Answer question
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

    # Chat history
    chat_history = ChatMessageHistory()
    
    print("\nChat with your PDF! Type 'exit' to quit.")
    
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit", "q"]:
            break
            
        response = rag_chain.invoke({"input": query, "chat_history": chat_history.messages})
        print(f"Assistant: {response['answer']}")
        
        chat_history.add_user_message(query)
        chat_history.add_ai_message(response["answer"])

if __name__ == "__main__":
    main()
