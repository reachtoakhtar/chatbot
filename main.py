import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import create_retriever_tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
AZURE_OPENAI_EMBEDDING_MODEL = os.getenv('AZURE_OPENAI_EMBEDDING_MODEL')


def chatbot():
    llm = AzureChatOpenAI(
        model=AZURE_OPENAI_DEPLOYMENT_NAME,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        temperature=0
    )

    filepath = 'docs'
    certification_policy_tool = create_tool(os.path.join(filepath, 'certification_policy.pdf'), 'certification_policy')
    leave_policy_tool = create_tool(os.path.join(filepath, 'leave_policy.pdf'), 'leave_policy')
    reimbursement_policy_tool = create_tool(os.path.join(filepath, 'reimbursement_policy.pdf'), 'reimbursement_policy')
    rewards_and_rec_guide_tool = create_tool(os.path.join(filepath, 'rewards_and_recognition_guide.pdf'),
                                             'rewards_and_recognition_guide')

    tools = [leave_policy_tool, certification_policy_tool, reimbursement_policy_tool, rewards_and_rec_guide_tool]

    memory = MemorySaver()
    config = {"configurable": {"thread_id": "tid"}}

    index = 0
    while True:
        print('\n----------------')
        print(f"Query number: {index + 1}")
        print('----------------')

        input_query = input("What's your query? \n")
        agent_executor = create_react_agent(llm, tools, checkpointer=memory)
        response = agent_executor.invoke({"messages": [HumanMessage(content=input_query)]}, config=config)

        ai_messages = [message.content for message in response["messages"] if isinstance(message, AIMessage)]
        filtered_response = list(filter(lambda content: content != "", ai_messages))
        print(f"AI : {filtered_response[index]}")

        index += 1


def create_tool(filepath, toolname):
    loader = PyPDFLoader(filepath)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    vectorstore = InMemoryVectorStore.from_documents(
        documents=splits, embedding=AzureOpenAIEmbeddings(model=AZURE_OPENAI_EMBEDDING_MODEL)
    )

    retriever = vectorstore.as_retriever()
    tool = create_retriever_tool(
        retriever,
        toolname,
        description=toolname
    )

    return tool


if __name__ == '__main__':
    chatbot()
