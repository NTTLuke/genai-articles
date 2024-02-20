# Simple rag with simple "approach" doesn't work well when the context
# of document referring to the same topic for different business context
# Example : Searching for "XXXX" in a document where "XXXX"is mentioned
# for different scenario

import os
from operator import itemgetter
from typing import List, Tuple

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, format_document
from langchain.vectorstores import Pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.chat_models.azure_openai import AzureChatOpenAI

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

index_name = "cdr"
vector_store = AzureSearch(
    azure_search_endpoint=os.getenv("AZURE_SEARCH_SERVICE"),
    azure_search_key=os.getenv("AZURE_SEARCH_KEY"),
    index_name=index_name,
    embedding_function=embedding.embed_query,
)

# Condense a chat history and follow-up question into a standalone question
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# RAG answer synthesis prompt
template = """Answer the question based only on the following context:
<context>
{context}
</context>"""
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)

# Conversational Retrieval Chain
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


def _get_docs_metadata(docs) -> List:
    metadata = []
    for doc in docs:
        metadata.append({"id": doc["metadata"].page})

    return metadata


# User input
class ChatHistory(BaseModel):
    chat_history: List[Tuple[str, str]] = Field(..., extra={"widget": {"type": "chat"}})
    question: str


_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(itemgetter("question")),
)

#
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3, "fetch_k": 20, "score_threshold": 0.90},
)

_inputs = RunnableParallel(
    {
        "question": lambda x: x["question"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "context": _search_query | retriever | _combine_documents,
    }
).with_types(input_type=ChatHistory)

chain = _inputs | ANSWER_PROMPT | llm | StrOutputParser()


result = chain.invoke(
    {
        "question": "Chi fa parte del gruppo di lavoro bilanci dello studio?",
        "chat_history": [],
    }
)


print(result)


#####  LANGSERVE  #####
#!/usr/bin/env python
# from fastapi import FastAPI
# from langserve import add_routes


# app = FastAPI(
#     title="LangChain Server",
#     version="1.0",
#     description="A simple api server using Langchain's Runnable interfaces",
# )

# add_routes(
#     app,
#     chain,
#     path="/rag",
# )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
