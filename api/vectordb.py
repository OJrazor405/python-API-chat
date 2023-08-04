from langchain.vectorstores import Pinecone

import os
import pinecone

# This class is used to push embeddings to Pinecone
class VectorDB:

    # Constructor
    def __init__(self, embeddings, namespace, index_name):
        self.embeddings = embeddings
        self.namespace = namespace
        self.index_name = index_name
        # Load the .env file with your API keys
        # script_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of the current script
        # dotenv_path = os.path.join(script_dir, '..', '.env.local')  # Gets the path to the .env.local file
        # load_dotenv(dotenv_path)

        # # Load the API keys from the .env file
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_env_key = os.getenv("PINECONE_API_ENV")        
    
    # Function for pushing embeddings to Pinecone. Returns the docsearch object
    def pushTextEmbeddings(self, texts):
        self.texts = texts
        # initialize pinecone
        pinecone.init(
            api_key=self.pinecone_api_key,  # find at app.pinecone.io
            environment=self.pinecone_env_key  # next to api key in console
        )

        # THIS IS THE FUNCTIONALITY FOR CREATING A NEW INDEX EVERY TIME WE PUSH EMBEDDINGS - IMPLEMENT LATER
        # # check if 'openai' index already exists (only create index if not)
        # if 'openai' not in pinecone.list_indexes():
        #     pinecone.create_index('openai', dimension=len(embeds[0]))
        # # connect to index
        # index = pinecone.Index('openai')

        # Here we are creating a new index every time we push embeddings
        # We return the docsearch to use it for querying later
        docsearch = Pinecone.from_texts([t.page_content for t in self.texts], self.embeddings, index_name=self.index_name, namespace=self.namespace)
        
        return docsearch
  
    def pullVectorstore(self):
        # initialize pinecone
        pinecone.init(
            api_key=self.pinecone_api_key,  # find at app.pinecone.io
            environment=self.pinecone_env_key  # next to api key in console
        )
        vectorstore = Pinecone.from_existing_index(embedding=self.embeddings, index_name=self.index_name, namespace=self.namespace)
        
        return vectorstore
    
    def deleteAllVectors(self):
        pinecone.init(
            api_key=self.pinecone_api_key,  # find at app.pinecone.io
            environment=self.pinecone_env_key  # next to api key in console
        )
        index = pinecone.Index(self.index_name)
        index.delete(delete_all=True, namespace=self.namespace)
