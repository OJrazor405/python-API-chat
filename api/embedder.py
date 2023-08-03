from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

import os

# Class to embed a single pdf file
class Embedder:

    # # Constructor
    def __init__(self):
        # Load the .env file with your API keys
        # script_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of the current script
        # dotenv_path = os.path.join(script_dir, '..', '.env.local')  # Gets the path to the .env.local file
        # load_dotenv(dotenv_path)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
    
    def embedder(self):
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        
        return embeddings
    
    # Embed the provided pdf
    def embedPdf(self, url: str):
        self.url = url
        loader = PyPDFLoader(self.url)
        pages = loader.load_and_split()


        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(pages)

        return texts