from operator import itemgetter

from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain.vectorstores.chroma import Chroma
import os
from dotenv import load_dotenv

load_dotenv()

# embedding
embedding = AzureOpenAIEmbeddings(
    deployment=os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    chunk_size=1,
    embedding_ctx_length=1000,
)

# llm
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("OPENAI_CHAT_DEPLOYMENT_NAME"),
    openai_api_type="azure",
    temperature=0.0,
)


def _get_retriever():
    vectorstore = Chroma.from_texts(
        ["Sam is passionate about LLMs and GenerativeAi"], embedding=embedding
    )
    retriever = vectorstore.as_retriever()
    return retriever


def simple_rag(question: str):
    retriever = _get_retriever()

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke(question)
    return result


def rag_with_different_inputs(question: str, style: str):
    retriever = _get_retriever()

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}

    Remember to speak as {style} when giving your final answer.
    Use emojis if you want.
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "style": itemgetter("style"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke({"question": question, "style": style})
    return result


if __name__ == "__main__":
    question = "What is Sam passionate about?"

    # result = simple_rag(question=question)
    style = "Heavy Metal Fan"
    result = rag_with_different_inputs(question=question, style=style)
    print(result)
