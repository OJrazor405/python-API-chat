from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

import os

class SemanticSearch:
    def __init__(self, docsearch: Pinecone, query: str):
        # Initialize the SemanticSearch object with a Pinecone handle and a search query
        self.docsearch = docsearch
        self.query = query

        # Load the .env file with your API keys
        # script_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of the current script
        # dotenv_path = os.path.join(script_dir, '..', '.env')  # Gets the path to the .env.local file
        # load_dotenv(dotenv_path)

        # Load the OpenAI API key from the .env file
        # self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def runSemanticSearch(self):        
        # Create an OpenAI language model with the loaded API key
        llm = ChatOpenAI(temperature=0, model="gpt-4", streaming=True)
        
        # Set the prompt template for the question-answering model
        prompt_template = """You are a AI assistant called Egde AI. Use the following pieces of context to answer the question at the end. All answers should be answered with a gentle and informative tone.

        If you can't find the answer in the document, use the chatgpt knowledge base to answer the question. Remember to inform the user that you are using the chatgpt knowledge base for that given answer.

        Example: "I couldn't find any information about this topic in the document provided, but I found this answer in the chatgpt knowledge base: <answer>"

        If the user ask for the source of the given information provided, return the page number of the given page the content was retrieved from.

        Answer the question in the same language as the context.

        {context}

        Question: {question}
        """

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # Load a pre-trained question-answering model from OpenAI
        chain = load_qa_chain(llm, chain_type="stuff", prompt = PROMPT)

        # Search the Pinecone index for documents similar to the search query
        docs = self.docsearch.similarity_search(self.query)

        # Run the question-answering model on the retrieved documents and the search query
        return chain.run(input_documents=docs, question=self.query)