""" Chains file"""
from langchain.agents import AgentType, initialize_agent

from constants import logger
from .config import get_llm
from .tools import (answer_billing_query, answer_technical_query,
                    answer_using_rag, escalate_to_human, general_query)


def get_tools():
    """
    Returns a list of tools available to the support agent.
    """
    return [
        answer_using_rag,
        answer_billing_query,
        answer_technical_query,
        general_query,
        escalate_to_human,
        ]

def get_agent():
    """
    Initializes and returns a LangChain agent configured with tools and an LLM.
    Returns: LangChain agent instance
    """
    try:
        tools = get_tools()
        llm = get_llm()

        logger.info("Initializing support agent with tools and LLM.")
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True
        )
        return agent

    except Exception as e:
        logger.exception("Failed to initialize agent.")
        raise RuntimeError(f"Agent initialization error: {str(e)}")