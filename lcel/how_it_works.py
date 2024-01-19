from types import UnionType
from typing import Any
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.chat_models.azure_openai import AzureChatOpenAI
import os

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


# This is demostrate how pipe operator works behind the scene
class Runnable:
    def __init__(self, func):
        self.func = func

    def __or__(self, other):
        def chained_func(*args, **kwargs):
            return other(self.func(*args, **kwargs))

        return Runnable(chained_func)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def add_one(x):
    return x + 1


def multiply_by_two(x):
    return x * 2


def test_simple_use():
    add_one_chain = Runnable(add_one)
    multiply_by_two_chain = Runnable(multiply_by_two)

    # using the method
    # chain = add_one_chain.__or__(multiply_by_two_chain)
    # print(chain(5))

    # using the operator
    chain = add_one_chain | multiply_by_two_chain
    print(chain(5))


def run_in_parallel_simple_example():
    from langchain_core.runnables import (
        RunnableParallel,
        RunnablePassthrough,
        RunnableLambda,
    )
    from langchain.vectorstores.docarray import DocArrayInMemorySearch
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser

    vecstore_a = DocArrayInMemorySearch.from_texts(
        ["Tom is a senior software dev", "Tom was born XXX"], embedding=embedding
    )
    vecstore_b = DocArrayInMemorySearch.from_texts(
        ["Sam was born in 1922", "Sam is a musician"], embedding=embedding
    )

    retriever_a = vecstore_a.as_retriever()
    retriever_b = vecstore_b.as_retriever()

    prompt_str = """Answer the question below using the context:
    
    Context: 
    {context_a}
    {context_b}

    Question: {question}    
    
    Answer: """

    prompt = ChatPromptTemplate.from_template(prompt_str)

    retrieval = RunnableParallel(
        {
            "context_a": retriever_a,
            "context_b": retriever_b,
            "question": RunnablePassthrough(),
        }
    )

    # demostrate runnable lambda with custom function
    def return_when_point(x):
        if "." in x:
            return "\n".join(x.split(".")[:-1]) + "."
        else:
            return x

    # Chain
    chain = (
        retrieval | prompt | llm | StrOutputParser() | RunnableLambda(return_when_point)
    )

    out = chain.invoke("Question")
    print(out)


run_in_parallel_simple_example()
