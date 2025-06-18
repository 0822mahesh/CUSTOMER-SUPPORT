import os
from langchain.agents import create_openai_functions_agent, initialize_agent, AgentType
from langchain.prompts import ChatPromptTemplate
from .tools import (
    answer_using_rag,
    answer_billing_query,
    answer_general_query,
    answer_technical_query,
    escalate_to_human,
    log_query,
)
from dotenv import load_dotenv
from .config import OPENAI_API_KEY,get_llm
from langchain.prompts import PromptTemplate

load_dotenv()

def get_tools():
    return [
        answer_using_rag,
        answer_billing_query,
        answer_technical_query,
        answer_general_query,
        escalate_to_human,
        log_query,
        
    ]

def get_agent():
    """
    """
    agent = initialize_agent(tools=get_tools(),llm=get_llm(), agent=AgentType.OPENAI_FUNCTIONS, verbose= True)
    return agent

