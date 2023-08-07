from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

import os


class ConversationChain:
    def __init__(self):
        # Load the .env file with your API keys
        script_dir = os.path.dirname(
            os.path.abspath(__file__)
        )  # Gets the directory of the current script
        dotenv_path = os.path.join(
            script_dir, "..", ".env"
        )  # Gets the path to the .env.local file
        load_dotenv(dotenv_path)

    def createConversationChain(self, previous_messages):
        self.previous_messages = previous_messages
        # Create an OpenAI language model with the loaded API key
        llm = ChatOpenAI(
            temperature=0, openai_api_key=self.openai_api_key, model="gpt-4"
        )  # k=6 defines the amout of prompts the bot remembers at a time.

        # Set the prompt template for the question-answering model
        prompt_template = """You are a AI assistant called Egde AI. Use the following pieces of context to answer the question at the end. All answers should be answered with a gentle and informative tone.

        If you can't find the answer in the document, use the chatgpt knowledge base to answer the question. Remember to inform the user that you are using the chatgpt knowledge base for that given answer.

        Example: "I couldn't find any information about this topic in the document provided, but I found this answer in the chatgpt knowledge base: <answer>"

        If the user ask for the source of the given information provided, return the page number of the given page the content was retrieved from.

        Answer the question in the same language as the context.

        {context}

        Question: {question}

        Previous messages: {previous_messages}
        """

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "previous_messages"],
        )
        # Load a pre-trained question-answering model from OpenAI
        conversationchain = load_qa_chain(
            llm,
            chain_type="stuff",
            prompt=PROMPT,
            memory=ConversationBufferWindowMemory(k=6),
        )  # k=6 defines the amout of prompts the bot remembers at a time.
        return conversationchain
