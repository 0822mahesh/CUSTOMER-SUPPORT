from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader,TextLoader
import os

data_directory = "data/docs/"
obs_path= os.path.abspath(os.path.join(os.path.dirname(__file__), data_directory))

def create_knowledge_base():
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
