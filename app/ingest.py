import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)


def build_vectorstore_from_pdf(pdf_path: str) -> FAISS:
    """
    Loads a PDF, splits it into chunks, embeds them, and builds a FAISS
    vector store in memory.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text could be extracted from this PDF.")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, len(chunks)