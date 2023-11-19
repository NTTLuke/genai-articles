from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
import os
from dotenv import load_dotenv

load_dotenv()

persist_directory = "data/chroma"
pdf_path = os.getenv("FILE_PATH")

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

# vector db
vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding)


def load_pdf(doc_path: str, embedding):
    pdf = PyPDFLoader(doc_path)
    doc = pdf.load()

    # Split
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    token_splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=10)
    splits = token_splitter.split_documents(doc)

    db = Chroma.from_documents(
        documents=splits, embedding=embedding, persist_directory=persist_directory
    )


def ask_question(question: str):
    memory = ConversationSummaryMemory(
        llm=llm, memory_key="chat_history", return_messages=True
    )

    retriever = vector_db.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

    result = qa(question)
    print(result["answer"])


def retrieval_transform(inputs: dict) -> dict:
    import logging
    from langchain.retrievers.multi_query import MultiQueryRetriever

    # to see query
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_db.as_retriever(), llm=llm
    )
    docs = retriever.get_relevant_documents(query=inputs["question"])
    docs = [d.page_content for d in docs]
    docs_dict = {"query": inputs["question"], "contexts": "\n---\n".join(docs)}
    return docs_dict


def ask_question_multiquery(question: str):
    from langchain.chains import TransformChain

    # docs = retriever.get_relevant_documents(query=question)
    # print(docs)

    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

    QA_PROMPT = PromptTemplate(
        input_variables=["query", "contexts"],
        template="""You are a helpful assistant who answers user queries using the
        contexts provided. If the question cannot be answered using the information
        provided say "I don't know".

        Contexts:
        {contexts}

        Question: {query}""",
    )

    # Chain
    qa_chain = LLMChain(llm=llm, prompt=QA_PROMPT)

    retrieval_chain = TransformChain(
        input_variables=["question"],
        output_variables=["query", "contexts"],
        transform=retrieval_transform,
    )

    from langchain.chains import SequentialChain

    rag_chain = SequentialChain(
        chains=[retrieval_chain, qa_chain],
        input_variables=["question"],  # we need to name differently to output "query"
        output_variables=["query", "contexts", "text"],
    )

    out = rag_chain({"question": question})
    print(out["text"])


if __name__ == "__main__":
    ## LOAD PDF
    # load_doc(pdf_path, embedding)

    question = "Quale strumento suona Fred?"
    # question = "Qual'è il ruolo di Fred nella Band del Bosco?"
    # question = "Qual'è il ruolo di Fred, come musicista, nella Band del Bosco?"

    ## ASK QUESTION WITH ConversationalRetrievalChain
    # ask_question(question=question)

    ## ASK QUESTION WITH MULTIQUERY RETRIEVER
    # ask_question_multiquery(question=question)
