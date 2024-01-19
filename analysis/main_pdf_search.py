from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import requests
from bs4 import BeautifulSoup
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
import os
from dotenv import load_dotenv
from langchain.utilities import BingSearchAPIWrapper
from pypdf import PdfReader

from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

import json

load_dotenv()


persist_directory = "data/chroma"
pdf_path = "./docs/Perizia.pdf"
collection_name = "perizia"

# embedding
embedding = AzureOpenAIEmbeddings(
    deployment=os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    chunk_size=1,
    embedding_ctx_length=1000,
)
# vector db
vector_db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding,
    collection_name=collection_name,
)


def load_pdf(doc_path: str, embedding):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=15, separators=["\n\n", "\n", " "]
    )
    pdf = PdfReader(doc_path)

    docs = []
    for page_num in range(len(pdf.pages)):
        pdf_page = pdf.pages[page_num]
        pdf_page_text = pdf_page.extract_text()

        # create metadata
        # page_nr = int(page_num + 1)
        # name_without_extension = file_name.split(".")[0]
        # source = f"{name_without_extension}-page-{page_nr}.pdf"

        # split page
        text_splitted = text_splitter.split_text(pdf_page_text)
        # metadatas = [{"source": source, "page": page_nr} for _ in text_splitted]

        # convert to document
        documents = text_splitter.create_documents(texts=text_splitted)
        if len(documents) > 0:
            docs.extend(documents)

    db = vector_db.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )


# load_pdf(pdf_path, embedding)

SUMMARY_TEMPLATE = """{text}
-----------
Using above text, answer in short the following question:

> {question}
-----------

if the question cannot be answered, imply summarize the text. Include all factual information, numbers, statistics, etc.
Use same language as the question.
"""

SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("OPENAI_CHAT_DEPLOYMENT_NAME"),
    openai_api_type="azure",
    temperature=0.0,
)


def doc_search(question: str):
    search_result = vector_db.similarity_search(query=question)
    return [doc.page_content for doc in search_result]


doc_search_chain = (
    RunnablePassthrough.assign(text=lambda x: doc_search(x["question"]))
    | SUMMARY_PROMPT
    | llm
    # | (lambda x: [{"question": x["question"], "text": t} for t in x["texts"]])
    | StrOutputParser()
)


SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Write 3 search queries to search into my documents that form an"
            "objective opinion from the following: {question}\n"
            "Use the same language as the question.",
        ),
        "You must respond with a list of strings in the following format:"
        '["query1", "query2", "query3"]',
    ]
)

search_question_chain = SEARCH_PROMPT | llm | StrOutputParser() | json.loads

full_research_chain = (
    search_question_chain
    | (lambda x: [{"question": q} for q in x])
    | doc_search_chain.map()
)

WRITER_SYSTEM_PROMPT = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."

# Report prompts from https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/master/prompts.py
RESEARCH_REPORT_TEMPLATE = """Information:
--------
{research_summary}
--------
Using the above information, answer the following question or topic: "{question}" in a detailed report -- \
The report should focus on the answer to the question, should be well structured, informative, \
in depth, with facts and numbers if available and a minimum of 1,200 words.
You should strive to write the report as long as you can using all relevant and necessary information provided.
You must write the report with markdown syntax.
You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
You must write the report in apa format.
Respond with the same language as the given question.
Please do your best, this is very important to my career."""  # noqa: E501

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_PROMPT),
        ("user", RESEARCH_REPORT_TEMPLATE),
    ]
)


def collapse_list_of_lists(list_of_lists):
    content = []
    for l in list_of_lists:
        content.append("\n\n".join(l))
    return "\n\n".join(content)


chain = (
    RunnablePassthrough.assign(research_summary=full_research_chain)
    | prompt
    | llm
    | StrOutputParser()
)


# chain.invoke({"question": "What are Actors in Dapr ?"})

# question = "question"
# chain.invoke({"question": question})


#####  LANGSERVE  #####
RUN_LANGSERVE = True

if RUN_LANGSERVE:
    #!/usr/bin/env python
    from fastapi import FastAPI
    from langserve import add_routes

    app = FastAPI(
        title="LangChain Server",
        version="1.0",
        description="A simple api server using Langchain's Runnable interfaces",
    )

    add_routes(
        app,
        chain,
        path="/research-assistant",
    )

    if __name__ == "__main__":
        import uvicorn

        uvicorn.run(app, host="localhost", port=8000)
