from operator import itemgetter

from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import os
from dotenv import load_dotenv

load_dotenv()

persist_directory = "data/new_chroma"
collection_name = "rag_with_lcel"

# embedding
embedding = AzureOpenAIEmbeddings(
    deployment=os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    chunk_size=1,
    embedding_ctx_length=1000,
)

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("OPENAI_CHAT_DEPLOYMENT_NAME"),
    openai_api_type="azure",
    temperature=0.0,
)


def load_from_texts():
    # vector db
    vectorstore = Chroma.from_texts(
        ["harrison worked at kensho"],
        persist_directory=persist_directory,
        embedding=embedding,
        collection_name=collection_name,
    )

    return vectorstore


def get_retriever():
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding,
        collection_name=collection_name,
    )
    retriever = vector_db.as_retriever()

    return retriever


# functions
def ask_v1(question: str):
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": get_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    chain.invoke(question)


def ask_v2(question: str, language: str):
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}

    Answer in the following language: {language}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {
            "context": itemgetter("question") | get_retriever(),
            "question": itemgetter("question"),
            "language": itemgetter("language"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke({"question": question, "language": language})
    return result


def ask_v3(question: str, language: str):
    """Conversational retrieval chain"""
    from langchain.prompts.prompt import PromptTemplate
    from langchain.schema import format_document
    from langchain_core.runnables import RunnableParallel
    from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string

    # template for question condensation
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    # template for answer
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

    def _combine_documents(
        docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
    ):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)

    _inputs = RunnableParallel(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: get_buffer_string(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    )

    _context = {
        "context": itemgetter("standalone_question")
        | get_retriever()
        | _combine_documents,
        "question": lambda x: x["standalone_question"],
    }

    conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | llm

    conversational_qa_chain.invoke(
        {
            "question": "where did harrison work?",
            "chat_history": [],
        }
    )


# Main
# load data

# load_from_texts()

# result = ask_v1("where did harrison work?")
result = ask_v2("where did harrison work?", "italian")

print(result)
