#APP
# Libraries for Queryrun
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun

#for wrappers
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper

#inbuilt tools
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

#import for rag
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

#Rag operations
loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
vectordb = FAISS.from_documents(documents=documents, embedding=OpenAIEmbeddings())
retriver = vectordb.as_retriever()
retriver

#create retriever tool
from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(retriver, "langsmith-search", "Search any information about Langsmith")
retriever_tool

# create the tool
tools = [arxiv, wiki, retriever_tool]
tools

## Run all these tools with Agents and LLM Models
## Tools, LLM-->AgentExecutor

from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# create a prompt template by using hub kyoki hme readymade prompt use krna hain
from langchain import hub
prompt = hub.pull("hwchase17/openai-functions-agent")
prompt.messages

#now create the agent
from langchain.agents import create_openai_tools_agent
agent = create_openai_tools_agent(llm, tools, prompt)

#now create agent executor
#Agent sirf "dimag" hai >> AgentExecutor us dimag ko "body + hands" deta hai
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# now invoke this
agent_executor.invoke({"input":"Tell me about Langsmith"})

# we can try it by asking anything
agent_executor.invoke({"input":"What is machine learning"})
agent_executor.invoke({"input":"What's the paper 1706.03762 about?"})

