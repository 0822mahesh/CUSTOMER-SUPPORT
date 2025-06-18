""" Knowledge base file """
import os
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

DATA_DIR = "data/docs/"
obs_path= os.path.abspath(os.path.join(os.path.dirname(__file__), DATA_DIR))

def create_knowledge_base():
    """
    This function will help you create vectore store
    """
    try:
        loader = DirectoryLoader(obs_path,glob="*.txt",loader_cls=TextLoader)
        docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size = 500, chunk_overlap= 100)
        split_docs = text_splitter.split_documents(documents=docs)
        vector_store = FAISS.from_documents(split_docs,embedding=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")))
        vector_store.save_local("data/vectorstore")
    except Exception as e:
        raise e   
if __name__=="__main__":
    create_knowledge_base()
