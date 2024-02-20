import logging
import sys
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
    ServiceContext,
    Document,
)

from llama_index.llms import AzureOpenAI
from llama_index.node_parser import (
    SentenceWindowNodeParser,
    HierarchicalNodeParser,
    get_leaf_nodes,
)
from llama_index.text_splitter import SentenceSplitter
from llama_index.embeddings import AzureOpenAIEmbedding, HuggingFaceEmbedding
from llama_index.schema import MetadataMode
from llama_index.postprocessor import MetadataReplacementPostProcessor
from llama_index import set_global_service_context
from llama_index.llms.types import ChatMessage
import chromadb
from llama_index.vector_stores import ChromaVectorStore

file_content = ""
with open("docs/detrazioni-miste.txt", "r") as file:
    file_content = file.read()

documents = [Document(text=file_content)]


api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("OPENAI_API_VERSION")

llm = AzureOpenAI(
    engine="chat",
    temperature=0,
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

# messages = [
#     ChatMessage(role="system", content="You are a pirate with colorful personality."),
#     ChatMessage(role="user", content="Hello"),
# ]

# response = llm.chat(messages)
# print(response)


# You need to deploy your own embedding model as well as your own chat completion model
embed_model = AzureOpenAIEmbedding(
    azure_deployment="embeddings",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)


# save to disk
# db = chromadb.PersistentClient(path="./chroma_db")
# chroma_collection = db.get_or_create_collection("detrazioni")
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

# index = VectorStoreIndex.from_documents(
#     documents, storage_context=storage_context, service_context=service_context
# )


# load from disk
db2 = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db2.get_or_create_collection("detrazioni")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    service_context=service_context,
)


from llama_index.llms import ChatMessage, MessageRole
from llama_index.prompts import ChatPromptTemplate

# Text QA Prompt
chat_text_qa_msgs = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=("You are an helpful chat assistant. You are here to help the user."),
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the question: {query_str}\n"
        ),
    ),
]
text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)

# Refine Prompt
chat_refine_msgs = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=("Always answer the question, even if the context isn't helpful."),
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=(
            "We have the opportunity to refine the original answer "
            "(only if needed) with some more context below.\n"
            "------------\n"
            "{context_msg}\n"
            "------------\n"
            "Given the new context, refine the original answer to better "
            "answer the question: {query_str}. "
            "If the context isn't useful, output the original answer again.\n"
            "Original Answer: {existing_answer}"
        ),
    ),
]
refine_template = ChatPromptTemplate(chat_refine_msgs)


query_engine = index.as_query_engine(
    text_qa_template=text_qa_template, refine_template=refine_template
)
response = query_engine.query("Question")

print(response)
