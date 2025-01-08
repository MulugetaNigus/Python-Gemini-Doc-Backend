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
import sys

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not found in environment variables")
    sys.exit(1)

# get the env variable from the .env file
genai.configure(api_key=GOOGLE_API_KEY)
PORT = int(os.getenv("PORT", "8000"))

# Subject mapping
subject_mapping = {
    "flutter": ["source/flutter_tutorial.pdf"],
    "embedded": ["source/Lecture 1.pdf"],
}

# Define pdf_paths based on subject_mapping
pdf_paths = [pdf for subject in subject_mapping.values() for pdf in subject]

# get the content from the pdf file
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# chunk the content
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# vectorization
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# define the conversation schema
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context try different questions", don't provide the wrong answer, \n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # the main ai engine for the capabilities of the process the pdf content
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# get the user input
def user_input(user_question, pdf_docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# start looking for to start the pdf read
@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    listener = None
    try:
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
        
    finally:
        # Cleanup
        try:
            logger.info("Something went wrong")
        except Exception as e:
            logger.error(f"Error during startups: {str(e)}")

# init the fastAPI
app = fastapi.FastAPI(lifespan=lifespan)

# cors 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# our pydantic model
class PDFRequest(BaseModel):
    question: str     
    subject: Optional[str] = None

# vectors
vector_store = None
vector_store_loaded = False

# a fast api endpoints for the pdf processing
@app.post("/process_pdf")
async def process_pdf(request: PDFRequest):
    try:
        if not vector_store_loaded:
            raise HTTPException(status_code=500, detail="Vector store not loaded. Please try again later.")
        
        # Get the subject and corresponding PDF paths
        subject = request.subject
        if subject not in subject_mapping:
            raise HTTPException(status_code=400, detail="Invalid subject provided.")
        
        pdf_paths = subject_mapping[subject]
        raw_text = get_pdf_text(pdf_paths)
        text_chunks = get_text_chunks(raw_text)
        vector_store = get_vector_store(text_chunks)

        question = request.question
        answer = user_input(question, pdf_paths)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# get request test
@app.get("/")
def read_root():
    return {"message": "Welcome to the PDF Question Answering API!"}
                
# main entry
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Gemini:app", host="127.0.0.1", port=PORT, reload=True)