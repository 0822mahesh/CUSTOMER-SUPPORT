from .chains import get_agent
from .tools import answer_using_rag

def run_support_agent(user_query:str):
    try:
        agent = get_agent()
        return agent.invoke(user_query)
    except Exception as e:
        return {"message":f"unable to call the agent{e}"}