from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def get_retriever(persist_directory: str = "./chroma_db"):
    """Returns a retriever optimized for diverse academic chunks using MMR."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )
    
    # MMR pulls 10 chunks, filters to the 3 most diverse to prevent redundant context
    return vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 3, "fetch_k": 10}
    )