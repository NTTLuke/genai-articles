# https://www.youtube.com/watch?v=oDzWsynpOyI


import logging
import sys
import os
from dotenv import load_dotenv

load_dotenv()


from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
    ServiceContext,
    Document,
)

import json
import llama_index
from llama_index.llms import AzureOpenAI
from llama_index.node_parser import (
    SentenceWindowNodeParser,
    HierarchicalNodeParser,
    get_leaf_nodes,
)
from llama_index.text_splitter import SentenceSplitter
from llama_index.embeddings import AzureOpenAIEmbedding, HuggingFaceEmbedding
from llama_index.schema import MetadataMode
from llama_index.postprocessor import (
    MetadataReplacementPostProcessor,
    SimilarityPostprocessor,
)
from llama_index import set_global_service_context
from llama_index.llms.types import ChatMessage
import chromadb
from llama_index.vector_stores import ChromaVectorStore


### THE LLM
api_key = os.getenv("OPENAI_API_KEY")
azure_endpoint = os.getenv("OPENAI_API_BASE")
api_version = os.getenv("OPENAI_API_VERSION")

llm = AzureOpenAI(
    engine="chat",
    temperature=0.1,
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)


embed_model = AzureOpenAIEmbedding(
    azure_deployment="embeddings",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)


def _print_docs(docs):
    # inspect documents
    print("length of documents: ", str(len(docs)))
    print("-----")
    print(docs)

    print("-----Metadata-----")
    for doc in docs:
        print(doc.metadata)


def _print_nodes(name, nodes):
    print("-----" + name + "-----")
    counter = 1
    for node in nodes:
        print(f"-----Node {counter}")
        dict_node = dict(node)

        print(dict_node)
        counter += 1

    print("-----")


def _create_text_qa_template():
    from llama_index.llms import ChatMessage, MessageRole

    from llama_index.prompts import ChatPromptTemplate

    # Text QA Prompt
    chat_text_qa_msgs = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "You are an helpful chat assistant. You are here to help the user.Answer must be in the original language."
            ),
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=(
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the context information and not prior knowledge,"
                "answer the question: {query_str}\n"
            ),
        ),
    ]
    text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)

    return text_qa_template


def _create_refine_template():
    from llama_index.llms import ChatMessage, MessageRole

    from llama_index.prompts import ChatPromptTemplate

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
    return refine_template


def create_window_nodes(path="./sample-docs/"):
    # get the file
    documents = SimpleDirectoryReader(path).load_data()
    # _print_docs(documents)

    sentence_node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    window_nodes = sentence_node_parser.get_nodes_from_documents(documents)

    # _print_nodes("WINDOW NODES", window_nodes)
    return window_nodes


def create_base_nodes(path="./sample-docs/"):
    # get the file
    documents = SimpleDirectoryReader(path).load_data()
    # _print_docs(documents)

    base_node_parser = SentenceSplitter()

    base_nodes = base_node_parser.get_nodes_from_documents(documents)

    # _print_nodes("BASE NODES", base_nodes)
    return base_nodes


def save_on_chroma_and_get_index(nodes, collection_name):
    ### CREATE THE VECTOR STORES
    ### SAVING VECTORS ON DISK
    db = chromadb.PersistentClient(path="./chroma_db")

    vector_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=vector_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    ctx = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model, node_parser=nodes
    )

    index = VectorStoreIndex(
        nodes, storage_context=storage_context, service_context=ctx
    )

    return index


def get_index(collection_name):
    db2 = chromadb.PersistentClient(path="./chroma_db")
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

    collection = db2.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        service_context=service_context,
    )

    return index


def run_window_index_sample(question):
    window_index = get_index("window-detrazioni")
    text_qa_template = _create_text_qa_template()
    refine_template = _create_refine_template()

    window_query_engine = window_index.as_query_engine(
        similarity_top_k=5,
        verbose=True,
        text_qa_template=text_qa_template,
        # refine_template=refine_template,
        node_postprocessor=MetadataReplacementPostProcessor(
            target_metadata_key="window",
        )
        # node_postprocessors=[
        #     SimilarityPostprocessor(similarity_cutoff=0.7),
        #     MetadataReplacementPostProcessor(
        #         target_metadata_key="window",
        #     ),
        # ],
    )

    base_response = window_query_engine.query(question)
    print(base_response)


def run_base_index_sample(question):
    base_index = get_index("base-detrazioni")
    text_qa_template = _create_text_qa_template()
    refine_template = _create_refine_template()

    # Query engine
    # base_query_engine = base_index.as_query_engine(
    #     verbose=True,
    #     text_qa_template=text_qa_template,
    #     # refine_template=refine_template,
    # )

    # chat engine

    base_query_engine = base_index.as_chat_engine()

    base_response = base_query_engine.chat(question)
    print(base_response)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    # windows_node = create_window_nodes()
    # window_index = save_on_chroma_and_get_index(windows_node, "window-detrazioni")

    ### INFERENCE
    question = "question!!!"
    # window_index = run_window_index_sample(question=question)
    base_index = run_base_index_sample(question=question)

# ### TODO : TO INVESTIGATE
# ### SAVING INDEX DEFINITION ON DISK
# ### this is useful to avoid having to recreate the index every time so we can save money
# ### from embedding calls
# window_index.storage_context.persist(persist_dir="./window-indexes")
# base_index.storage_context.persist(persist_dir="./base-indexes")

# ### RELOAD INDEXES FROM DISK
# SC_retrieved_window = storage_context_window.from_defaults(
#     persist_dir="./window-indexes"
# )
# SC_retrieved_base = storage_context_base.from_defaults(persist_dir="./base-indexes")

# retrieved_window_index = load_index_from_storage(SC_retrieved_window)
# retrieved_base_index = load_index_from_storage(SC_retrieved_base)
