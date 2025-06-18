""" Agent Executer"""
from .chains import get_agent

def run_support_agent(user_query:str):
    """ 
    This function retrives an agent and invokes it with user question.
    """
    try:
        agent = get_agent()
        return agent.invoke(user_query)
    except (ValueError,AttributeError) as e:
        return {"message":f"unable to call the agent{e}"}