from langchain.vectorstores import Pinecone
from dotenv import load_dotenv

import os


class SemanticSearch:
    def __init__(self):
        # Load the .env file with your API keys
        script_dir = os.path.dirname(
            os.path.abspath(__file__)
        )  # Gets the directory of the current script
        dotenv_path = os.path.join(
            script_dir, "..", ".env"
        )  # Gets the path to the .env.local file
        load_dotenv(dotenv_path)
        # Load the OpenAI API key from the .env file
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def runSemanticSearch(self, conversationchain, docsearch: Pinecone, query: str):
        # Initialize the SemanticSearch object with a Pinecone handle and a search query
        self.docsearch = docsearch
        self.query = query
        # Search the Pinecone index for documents similar to the search query
        docs = self.docsearch.similarity_search(self.query)
        # Run the question-answering model on the retrieved documents and the search query
        return conversationchain.run(input_documents=docs, question=self.query)
