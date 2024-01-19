from dotenv import load_dotenv
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from langchain.schema.output_parser import StrOutputParser
from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser


class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


def get_simple_chain(query: str, parser: None):
    # llm
    model = AzureChatOpenAI(
        azure_deployment=os.getenv("OPENAI_CHAT_DEPLOYMENT_NAME"),
        openai_api_type="azure",
        temperature=0.0,
    )

    prompt = ChatPromptTemplate.from_template(query)

    # simple chain without parser
    chain = prompt | model

    # add parser if provided
    if parser is not None:
        chain = prompt | model | parser

    return chain


if __name__ == "__main__":
    query = "tell me a joke about {topic}."

    # simple
    chain = get_simple_chain(query=query, parser=StrOutputParser())
    response = chain.invoke({"topic": "bears"})
    print(response)

    # stream
    # chain = get_simple_chain(query=query, parser=None)
    # for s in chain.stream({"topic": "programming"}):
    #     print(s.content, end="", flush=True)

    # batch
    # chain = get_simple_chain(query=query, parser=StrOutputParser())
    # response = chain.batch(
    #     [{"topic": "bears"}, {"topic": "cats"}], config={"max_concurrency": 5}
    # )
    # print(response)
