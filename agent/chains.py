""" Chains file"""
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from .config import get_llm
from .tools import (answer_billing_query, answer_general_query,
                    answer_technical_query, answer_using_rag,
                    escalate_to_human, log_query)

load_dotenv()

def get_tools():
    """
    Returns a list of tools available to the support agent.
    """
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
    Initializes and returns a support agent using predefined tools and an LLM.
    """
    agent = initialize_agent(tools=get_tools(),llm=get_llm(), agent=AgentType.OPENAI_FUNCTIONS, verbose= True)
    return agent