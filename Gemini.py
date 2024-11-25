import os
import streamlit as st  # type: ignore
from PyPDF2 import PdfReader  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
import google.generativeai as genai  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
from langchain.chains.question_answering import load_qa_chain  # type: ignore
from langchain.prompts import PromptTemplate  # type: ignore
from dotenv import load_dotenv  # type: ignore
import fastapi  # type: ignore
from fastapi import HTTPException  # type: ignore
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from contextlib import asynccontextmanager
from loguru import logger
import ngrok
import sys
from pathlib import Path

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not found in environment variables")
    sys.exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

# Ngrok configuration
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
if not NGROK_AUTH_TOKEN:
    logger.error("NGROK_AUTH_TOKEN not found in environment variables")
    sys.exit(1)

PORT = int(os.getenv("PORT", "8000"))

# Create source directory if it doesn't exist
source_dir = Path("source")
source_dir.mkdir(exist_ok=True)

# Update PDF paths to be more flexible
pdf_paths = []
if os.path.exists(source_dir):
    pdf_paths = [str(p) for p in source_dir.glob("*.pdf")]
    if not pdf_paths:
        logger.warning("No PDF files found in source directory")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer, but use your back knowledge ans answer the user question as much as possible even if the content is not provided\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    # Set up ngrok
    try:
        logger.info("Setting up Ngrok Tunnel")
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        logger.info(f"Using Ngrok auth token: {'*' * len(NGROK_AUTH_TOKEN)}")
        
        # Connect to ngrok
        listener = await ngrok.connect(PORT, authtoken=NGROK_AUTH_TOKEN)
        logger.info(f"Ngrok tunnel established at: {listener.url()}")
    except Exception as e:
        logger.error(f"Failed to setup ngrok tunnel: {str(e)}")
        raise

    # Initialize vector store
    global vector_store, vector_store_loaded
    raw_text = ""
    try:
        if pdf_paths:
            for pdf_path in pdf_paths:
                with open(pdf_path, "rb") as f:
                    raw_text += get_pdf_text([f])
            text_chunks = get_text_chunks(raw_text)
            vector_store = get_vector_store(text_chunks)
            vector_store_loaded = True
            logger.info("PDF reading and processing completed. Ready for POST requests.")
        else:
            logger.warning("No PDFs to process")
            vector_store_loaded = False
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        vector_store_loaded = False
    
    yield
    
    # Cleanup ngrok
    try:
        logger.info("Tearing Down Ngrok Tunnel")
        await ngrok.disconnect()
    except Exception as e:
        logger.error(f"Error during ngrok shutdown: {str(e)}")

app = fastapi.FastAPI(lifespan=lifespan)

# cors 
# cors issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PDFRequest(BaseModel):
    question: str     
    subject: Optional[str] = None
    content: Optional[str] = None

vector_store = None
vector_store_loaded = False  # Flag to track if the vector store has been loaded

@app.post("/process_pdf")
async def process_pdf(request: PDFRequest):
    try:
        if not vector_store_loaded:
            raise HTTPException(status_code=500, detail="Vector store not loaded. Please try again later.")
        question = request.question
        answer = user_input(question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the PDF Question Answering API!"}

def streamlit_interface():
    st.set_page_config("Chat PDF")
    st.header("Multi-PDF Chat using Gemini")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        st.write(f"PDF Files defined in code: {', '.join(pdf_paths)}")

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = ""
                for pdf_path in pdf_paths:
                    with open(pdf_path, "rb") as f:
                        raw_text += get_pdf_text([f])
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
                st.write("PDF reading and processing is finished. Ready for querying.")
                
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Gemini:app", host="127.0.0.1", port=PORT, reload=True)