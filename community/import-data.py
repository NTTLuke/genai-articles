from langchain.chains import ConversationalRetrievalChain
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from pypdf import PdfReader
from langchain.document_loaders import PyPDFLoader

import os
from dotenv import load_dotenv

load_dotenv()

persist_directory = "data/chroma"
collection_name = "book_eng"

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

vector_db = Chroma(
    collection_name=collection_name,
    persist_directory=persist_directory,
    embedding_function=embedding,
)


def load_pdf(pdf_path: str):
    from io import BytesIO

    # Rec char Splitter
    character_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # sentence token splitter
    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=0, tokens_per_chunk=256
    )

    pdf_bytes = None
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    doc = PdfReader(BytesIO(pdf_bytes))
    docs = []
    for page_num in range(len(doc.pages)):
        pdf_page = doc.pages[page_num]
        pdf_page_text = pdf_page.extract_text()

        # skip empty pages
        if not pdf_page_text:
            continue

        # split text
        character_split_texts = character_splitter.split_text(pdf_page_text)

        token_split_texts = []
        for text in character_split_texts:
            token_split_texts += token_splitter.split_text(text)

        # create metadata from token split
        page_nr = int(page_num + 1)

        # set metadata for each split
        metadatas = [{"source": pdf_path, "page": page_nr} for _ in token_split_texts]

        # convert to document
        documents = character_splitter.create_documents(
            texts=token_split_texts, metadatas=metadatas
        )

        docs.extend(documents)

    db = Chroma.from_documents(
        collection_name=collection_name,
        documents=docs,
        embedding=embedding,
        persist_directory=persist_directory,
    )


# just for testing
def get_relevant_documents(question: str):
    docs = vector_db.similarity_search(question)
    return docs


def ask(
    query: str,
) -> str:
    """Ask to LLM and return the answer"""
    from langchain.retrievers import MultiQueryRetriever
    from langchain.prompts import PromptTemplate

    # Continue with the question using doc retriever
    # get the prompt
    final_prompt = """
        You are a friendly helpful assistant to help and maintain polite conversation. 
        Your users are asking questions about information retrieved from a book.
        Answer the user's question using only these information.
        Remember to be polite and friendly.
        
        Context: 
        {context}

        Question: 
        {question}

        Answer:
        """

    PROMPT = PromptTemplate(
        template=final_prompt,
        input_variables=["context", "question"],
    )

    # see on langsmith
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "score_threshold": 0.9},
        ),
        llm=llm,
    )

    chain_type_kwargs = {"prompt": PROMPT}
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs=chain_type_kwargs,
        rephrase_question=False,
    )

    llm_result = qa.invoke({"question": query, "chat_history": []})
    return llm_result


if __name__ == "__main__":
    ### code examples based on https://amzn.eu/d/5LZBc6p
    import logging

    # FOR LOGGING MULTIQUERY
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    # LOAD PDF
    pdf_path = os.getenv("FILE_PATH_ENG")
    print(pdf_path)

    # load_pdf(pdf_path)
    # print("pdf loaded")

    # result = get_relevant_documents("What's the name of the singer?")
    # print(result)

    # result = ask("What's the name of the singer?")
    result = ask("What is the instrument played by Fred?")
    print(result["answer"])
