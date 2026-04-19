import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from retrieval import get_retriever
import evaluation
from dotenv import load_dotenv

load_dotenv()

# --- 1. State Definition (Refactored to separate intent from query) ---
class GraphState(TypedDict):
    original_intent: str  # The user's actual question (never changes)
    search_query: str     # The query used for DB search (gets rewritten)
    generation: str
    context: str
    retries: int
    rewrite_retries: int

# --- 2. Initialize Tools ---
retriever = get_retriever()
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# --- 3. Node Functions ---
def retrieve_node(state: GraphState):
    print("\n>> NODE: Retrieving Documents")
    
    # Use the rewritten query if it exists, otherwise use the original intent
    current_query = state.get("search_query", state["original_intent"])
    docs = retriever.invoke(current_query)
    
    # NEW: Empty Retrieval Bypass to save LLM tokens
    if not docs:
        print("   [!] No documents found in database.")
        return {"context": "NO_DOCUMENTS_FOUND", "search_query": current_query}

    context = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in docs])
    
    print("\n   [X-RAY VISION] Snippet of retrieved context:")
    print(f"   \"{context[:200]}...\"\n")
    
    return {"context": context, "search_query": current_query}

def generate_node(state: GraphState):
    print("\n>> NODE: Generating Answer")
    prompt = PromptTemplate(
        template="Use the context to answer the original question. Cite your sources if possible.\nQuestion: {question}\nContext: {context}\nAnswer:",
        input_variables=["question", "context"]
    )
    # Always generate the answer based on the original intent!
    generation = llm.invoke(prompt.format(question=state["original_intent"], context=state["context"])).content
    return {"generation": generation, "retries": state.get("retries", 0) + 1}

def rewrite_node(state: GraphState):
    print("\n>> NODE: Rewriting Query")
    # Base the rewrite on the original intent so it doesn't drift
    better_query = evaluation.rewrite_query(state["original_intent"])
    print(f"   New Keywords: {better_query}")
    
    current_retries = state.get("rewrite_retries", 0)
    return {"search_query": better_query, "rewrite_retries": current_retries + 1}

# --- 4. Routing Logic ---
def check_relevance_route(state: GraphState):
    print("\n>> EVAL: Checking Context Relevance...")
    
    # Bypass the LLM judge completely if the DB returned nothing
    if state["context"] == "NO_DOCUMENTS_FOUND":
        print("   Status: Empty Context. Routing directly to Rewrite.")
        if state.get("rewrite_retries", 0) >= 2: return "max_retries"
        return "rewrite"

    is_relevant = evaluation.check_relevance(state["search_query"], state["context"])
    
    if is_relevant == "yes":
        print("   Status: Relevant. Proceeding to Generation.")
        return "generate"
    else:
        print("   Status: Irrelevant Context. Routing to Rewrite.")
        if state.get("rewrite_retries", 0) >= 2: 
            print("   [!] Max rewrite limit reached. Exiting to prevent loop.")
            return "max_retries"
        return "rewrite"

def check_hallucination_route(state: GraphState):
    print("\n>> EVAL: Checking Faithfulness (Hallucination)...")
    is_hallucinated = evaluation.check_faithfulness(state["context"], state["generation"])
    
    if is_hallucinated == "yes":
        print("   Status: Hallucinated! Forcing regeneration.")
        if state.get("retries", 0) >= 3: 
            print("   [!] Max generation limit reached.")
            return "max_retries"
        return "regenerate"
    else:
        print("   Status: Answer is faithful.")
        return "pass"

# --- 5. Build the Graph ---
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("rewrite", rewrite_node)

workflow.set_entry_point("retrieve")
workflow.add_conditional_edges("retrieve", check_relevance_route, {
    "generate": "generate",
    "rewrite": "rewrite",
    "max_retries": END
})
workflow.add_edge("rewrite", "retrieve")
workflow.add_conditional_edges("generate", check_hallucination_route, {
    "regenerate": "generate",
    "pass": END,
    "max_retries": END
})

app = workflow.compile()

# --- 6. Execution ---
if __name__ == "__main__":
    # Test Question
    initial_question = "How do Generative Agents form memories and interact according to the Park et al. paper?"
    
    inputs = {
        "original_intent": initial_question, 
        "search_query": initial_question, 
        "retries": 0,
        "rewrite_retries": 0
    }
    
    print(f"\n🚀 STARTING AGENTIC RAG SYSTEM...")
    print(f"User Question: {initial_question}")
    
    result = app.invoke(inputs)
    
    print("\n================== FINAL OUTPUT ==================")
    final_answer = result.get("generation", "Agent failed to generate an answer due to maximum retries or lack of relevant documents.")
    print(final_answer)
    print("==================================================")