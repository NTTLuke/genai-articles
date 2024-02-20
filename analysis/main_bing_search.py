from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import requests
from bs4 import BeautifulSoup
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
import os
from dotenv import load_dotenv
from langchain.utilities import BingSearchAPIWrapper
import json

load_dotenv()


RESULT_PER_QUESTION = 3

bing_search = BingSearchAPIWrapper()


def web_search(query: str, num_results: int = RESULT_PER_QUESTION):
    results = bing_search.results(query, num_results)
    return [r["link"] for r in results]


SUMMARY_TEMPLATE = """{text}
-----------
Using above text, answer in short the following question:

> {question}
-----------

if the question cannot be answered, imply summarize the text. Include all factual information, numbers, statistics, etc.
"""

SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)


llm = AzureChatOpenAI(
    azure_deployment=os.getenv("OPENAI_CHAT_DEPLOYMENT_NAME"),
    openai_api_type="azure",
    temperature=0.0,
)


def scrape_text(url: str):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator="", strip=True)
            return text
        else:
            return f"Failed to scrape text : {response.status_code}"

    except Exception as e:
        print(e)
        return f"Failed to scrape text {e}"


# get the content of the page and use it as the context
# url = "https://blog.langchain.dev/announcing-langsmith/"
scrape_and_summarize_chain = (
    RunnablePassthrough.assign(text=lambda x: scrape_text(x["url"])[:10000])
    | SUMMARY_PROMPT
    | llm
    | StrOutputParser()
)

web_search_chain = (
    RunnablePassthrough.assign(urls=lambda x: web_search(x["question"]))
    | (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]])
    | scrape_and_summarize_chain.map()
)


SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Write 3 google search queries to search online that form an"
            "objective opinion from the following: {question}\n",
        ),
        "You must respond with a list of strings in the following format:"
        '["query1", "query2", "query3"]',
    ]
)

search_question_chain = SEARCH_PROMPT | llm | StrOutputParser() | json.loads

full_research_chain = (
    search_question_chain
    | (lambda x: [{"question": q} for q in x])
    | web_search_chain.map()
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
    RunnablePassthrough.assign(
        research_summary=full_research_chain | collapse_list_of_lists
    )
    | prompt
    | llm
    | StrOutputParser()
)

#####  LANGSERVE  #####
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
