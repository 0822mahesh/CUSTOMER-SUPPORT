""" Tools files"""
import os
from langchain import hub
from langchain.tools import tool
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from .config import OPENAI_API_KEY, get_llm
from .knowledge_base import create_knowledge_base


@tool
def answer_using_rag(query: str) -> dict:
    """this tool used the knowlwdge base to give answer"""
    if not os.path.exists("data/vectorstore"):
        create_knowledge_base()
    vector_store = FAISS.load_local("data/vectorstore",
                                    embeddings=OpenAIEmbeddings(api_key=OPENAI_API_KEY),
                                    allow_dangerous_deserialization=True)
    results = vector_store.similarity_search(query=query,k=3)
    print(results)
    context = " ".join([doc.page_content for doc in results])
    prompt = hub.pull("rlm/rag-prompt")
    llm = get_llm()
    parser = StrOutputParser()
    chain = prompt | llm | parser
    result = chain.invoke(
        {"context": context,
         "question": query}
    )
    return {"input": query,"output": result}

@tool
def answer_billing_query(query: str) -> str:
    """Answer billing related query."""
    return "This is a billing-related response to: '{query}'"

@tool
def answer_technical_query(query: str) -> str:
    """Answer technical support related query."""
    return f"This is a technical support response to: '{query}'"

@tool
def answer_general_query(query: str) -> str:
    """Answer general purpose related query."""
    return f"This is a general response to: '{query}'"

@tool
def escalate_to_human(query: str) -> str:
    """ Escalating to live agent"""
    return f"Escalating to human support for query: '{query}'"

@tool
def log_query(query: str) -> str:
    """ logging the Query"""
    print(f"Logged Query: {query}")
    return "Query logged"
