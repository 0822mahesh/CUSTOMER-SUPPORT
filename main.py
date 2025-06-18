from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from agent.agent_executor import run_support_agent
from agent.chains import get_agent

app = FastAPI()

class Query(BaseModel):
    user_query: str

@app.post("/support")
def get_support(query: Query):
    try:
        result = run_support_agent(query.user_query)
        return result
      
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

