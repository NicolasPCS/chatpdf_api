from dotenv import load_dotenv
from pyprojroot import here
from uuid import uuid4
from langsmith import Client
from langchain.docstore.document import Document

from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_xml_agent, create_tool_calling_agent
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain import hub
from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.utils.function_calling import convert_to_openai_function

import csv
import os
import unicodedata
import dill as pickle

# INICIAR LANGSMITH Y API KEYS
#dotenv_path = here() / ".env"
#load_dotenv(dotenv_path=dotenv_path)

load_dotenv()

client = Client()

embeddings = AzureOpenAIEmbeddings(model="text-embedding-ada-002")

# Definición de agente
openai  = AzureChatOpenAI(
    deployment_name="gpt-35-turbo-16k",
    temperature=0.0
)

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", """Task: You are a helpful assistant, expert on the core of Bantotal and Genexus 8, 9 y 16. You must answer users question IN SPANISH. 
        Instructions: All information in your answers must be retrieved from your knowledge or based on previous information from the chat history. In case the question can’t be answered based on your knowledge, you must ask the user to provide more information or context. Otherwise honestly say that you can not answer that question.
        Be detailed in your answers but stay focused to the question. Add all details that are useful to provide a complete answer, but do not add details beyond the scope of the question."""),
        MessagesPlaceholder(variable_name="chat_history"),("human", "{input}"),
    ]
)

chain = chat_template|openai

#agent = AgentExecutor(agent=openai, prompt_template=chat_template, tools=tools, verbose=False)
#agent = create_openai_tools_agent(openai, tools, chat_template)
#agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)