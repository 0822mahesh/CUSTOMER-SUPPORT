import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

if not OPENAI_API_KEY:
    raise EnvironmentError("OpenAI_API_KEY is not available in environment")

def get_llm():
    """
    This function will initiate ChatOpenAI model.
    """
    return ChatOpenAI(model="gpt-4",temperature=0, api_key=OPENAI_API_KEY)
