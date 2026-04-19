import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def build_vector_store(pdf_directory: str, persist_directory: str):
    print(f"Loading PDFs from '{pdf_directory}'...")
    
    loader = PyPDFDirectoryLoader(pdf_directory)
    docs = loader.load()
    print(f"Success: Loaded {len(docs)} pages.")

    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Success: Created {len(chunks)} text chunks.")
    
    print("Embedding and storing in ChromaDB... (This may take a minute or two)")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    
    print("Ingestion complete. Database saved successfully.")

if __name__ == "__main__":
    build_vector_store("./papers", "./chroma_db")