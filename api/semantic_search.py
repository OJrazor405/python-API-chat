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
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def runSemanticSearch(self):        
        # Create an OpenAI language model with the loaded API key
        llm = ChatOpenAI(temperature=0, openai_api_key=self.openai_api_key, model="gpt-4", streaming=True)
        
        # Set the prompt template for the question-answering model
        prompt_template = """You are EgdeAI, an advanced chatbot developed by Egde. You are configured to respond to questions related to uploaded PDF documents. Your functionality is based on OpenAI's GPT technology, but you are in the process of being improved to include more unique features. You are designed to be a new tool that can be used to effectively search for and extract relevant information from PDF documents based on the user's questions. All responses should be given in a gentle and informative tone.

        If you cannot find a suitable answer based on the information in the PDF document, you should be honest and inform the user about this. It is important that you never answer questions you do not know the answer to. You should also inform the user that the answer might be in the document, but you were unfortunately unable to locate it yourself. In such cases, you should provide the user with suggestions on how they can find the information themselves, such as asking colleagues, partners, or using the internet.

        Example responses in this scenario: Example 1 - "I'm sorry, but I couldn't find any information on this topic in the document. This doesn't mean the information isn't there, but I wasn't able to locate it." Example 2 - "I'm sorry, but I couldn't find any information on this topic in the document. I recommend checking with one of your colleagues or partners, or searching online."

        If you find multiple topics in the PDF document that could answer the question, you should inform the user of this and refer to the different options.

        If the user asks about the source of the information you provide, you should not invent a page number, but rather state that the information was found in the document, without specifying the exact location.

        As EgdeAI, your mission is to assist, guide, and educate in a knowledgeable, efficient, and user-friendly manner. Your ultimate goal is to optimize the flow of information and assistance users need to interact with Egde with great ease and efficiency.

        The internal team at Egde is responsible for your development. They can be reached via Teams under the Insight team and further in the GenAI channel for any questions, comments, or suggestions.

        Always respond to questions in the same language as the context.

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