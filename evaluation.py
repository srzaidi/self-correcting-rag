from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Initialize the judge LLM
eval_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

def check_relevance(question: str, context: str) -> str:
    """Checks if the retrieved context contains any information to answer the question."""
    prompt = PromptTemplate(
        template="""You are a lenient grader assessing the relevance of a retrieved document to a user question.
        Question/Keywords: {question}
        Context: {context}
        GOAL: Does the context contain ANY information, concepts, or keywords related to the question?
        If it contains even a partial answer or mentions the core topic, output exactly 'yes'. 
        If it is completely unrelated, output exactly 'no'. 
        Output nothing else.
        """,
        input_variables=["question", "context"]
    )
    return eval_llm.invoke(prompt.format(question=question, context=context)).content.strip().lower()

def check_faithfulness(context: str, generation: str) -> str:
    """Checks if the generated answer is strictly grounded in the retrieved context."""
    prompt = PromptTemplate(
        template="""You are a strict grader. Check if the drafted answer contains hallucinations.
        Context: {context}
        Draft: {generation}
        If the answer contains claims NOT in the context, output exactly 'yes' (it is hallucinated).
        If it is fully supported, output exactly 'no'. Output nothing else.
        """,
        input_variables=["context", "generation"]
    )
    return eval_llm.invoke(prompt.format(context=context, generation=generation)).content.strip().lower()

def rewrite_query(original_intent: str) -> str:
    """Rewrites the original question into dense keywords for vector search."""
    prompt = PromptTemplate(
        template="""You are an expert at optimizing search queries for vector databases.
        The user's original question failed to retrieve relevant documents.
        Extract only the most important keywords and core concepts from the question.
        
        Original Question: {original_intent}
        
        CRITICAL: OUTPUT ONLY A COMMA-SEPARATED LIST OF 3 TO 5 KEYWORDS. NO SENTENCES.
        Keywords:""",
        input_variables=["original_intent"]
    )
    return eval_llm.invoke(prompt.format(original_intent=original_intent)).content.strip()