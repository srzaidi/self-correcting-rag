# Self-Correcting Agentic RAG

An AI pipeline that reads private documents, answers questions, and double-checks its own work to prevent hallucinated facts. 

## 💡 What It Does
Standard AI models often make up facts (hallucinate). Standard RAG models pull documents but blindly trust whatever answer the AI drafts. 

This project solves that by acting like a strict researcher. It searches a local database of PDFs, drafts an answer, and then uses an **LLM-as-a-Judge** to grade its own draft. If the draft contains facts not found in the documents, the system rejects it and forces a rewrite.

## ⚙️ How It Works
1. **Retrieve:** Searches a local database of research papers.
2. **Evaluate Context:** Checks if the retrieved text is actually relevant. If not, it rewrites the user's query into SEO-style keywords and searches again.
3. **Generate:** Drafts an answer using the Groq Llama-3.1 API.
4. **Evaluate Factuality:** A strict AI judge compares the draft to the original text. If a hallucination is detected, it loops back and tries again.

## 🛠️ Built With
* **Python**
* **LangGraph & LangChain** (Agentic State Machine routing)
* **ChromaDB** (Local Vector Database)
* **HuggingFace** (Local Embeddings: `all-MiniLM-L6-v2`)
* **Groq API** (`Llama-3.1-8b-instant`)

## 🚀 How to Run It Locally

1. Clone this repository.
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the main folder and add your Groq API key:
   ```text
   GROQ_API_KEY=your_key_here
   ```
4. Create a folder named `papers/`, add some PDF research papers, and build the database:
   ```bash
   python ingestion.py
   ```
5. Run the agent:
   ```bash
   python agent.py
   ```
