from langchain.chains import ConversationalRetrievalChain
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from pypdf import PdfReader
from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
)
from typing import Any

import os
from dotenv import load_dotenv

load_dotenv()


class BookRag:
    def __init__(
        self,
        chroma_persistent_dir: str = "data/chroma",
        chroma_collection_name: str = "book_eng",
    ):
        self.CHROMA_PERSISTENT_DIR = chroma_persistent_dir
        self.CHROMA_COLLECTION_NAME = chroma_collection_name

        self.embeddings = AzureOpenAIEmbeddings(
            deployment=os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
            chunk_size=1,
            embedding_ctx_length=1000,
        )

        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("OPENAI_CHAT_DEPLOYMENT_NAME"),
            openai_api_type="azure",
            temperature=0.0,
        )

        self.vector_db = Chroma(
            collection_name=self.CHROMA_COLLECTION_NAME,
            persist_directory=self.CHROMA_PERSISTENT_DIR,
            embedding_function=self.embeddings,
        )

    def load_pdf(self, pdf_path: str):
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
            metadatas = [
                {"source": pdf_path, "page": page_nr} for _ in token_split_texts
            ]

            # convert to document
            documents = character_splitter.create_documents(
                texts=token_split_texts, metadatas=metadatas
            )

            docs.extend(documents)

        db = Chroma.from_documents(
            collection_name=self.CHROMA_COLLECTION_NAME,
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.CHROMA_PERSISTENT_DIR,
        )

    # just for testing
    def get_relevant_documents(self, question: str):
        docs = self.vector_db.similarity_search(question)
        return docs

    def get_retriever_parent_child(self) -> BaseConversationalRetrievalChain:
        pass

    def get_retriever_simple_rag(self) -> BaseConversationalRetrievalChain:
        """Ask to LLM and return the answer"""
        from langchain.prompts import PromptTemplate
        from langchain.memory import ConversationBufferMemory

        memory = ConversationBufferMemory(
            memory_key="chat_history", output_key="answer", return_messages=True
        )

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

        retriever = (
            self.vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3, "score_threshold": 0.9},
            ),
        )

        chain_type_kwargs = {"prompt": PROMPT}
        qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_db.as_retriever(),
            return_source_documents=True,
            verbose=True,
            combine_docs_chain_kwargs=chain_type_kwargs,
            rephrase_question=False,
            memory=memory,
        )

        return qa

    def get_retriever_using_multiquery(self) -> BaseConversationalRetrievalChain:
        """Ask to LLM and return the answer"""
        from langchain.retrievers import MultiQueryRetriever
        from langchain.prompts import PromptTemplate
        from langchain.memory import ConversationBufferMemory

        memory = ConversationBufferMemory(
            memory_key="chat_history", output_key="answer", return_messages=True
        )

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
            self.vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3, "score_threshold": 0.9},
            ),
            llm=self.llm,
        )

        chain_type_kwargs = {"prompt": PROMPT}
        qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever,
            return_source_documents=True,
            verbose=True,
            combine_docs_chain_kwargs=chain_type_kwargs,
            rephrase_question=False,
            memory=memory,
        )

        return qa

    def ask_base_rag(self, query: str):
        qa = self.get_retriever_simple_rag()

        llm_result = qa.invoke({"question": query})
        return llm_result

    def ask_multiquery(self, query: str) -> dict[str, Any]:
        qa = self.get_retriever_using_multiquery()

        llm_result = qa.invoke({"question": query})
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

    rag = BookRag()
    # result = rag.get_relevant_documents("What's the name of the singer?")
    # print(result)

    # question = "What's the name of the singer?"
    # question = "What is the instrument played by Fred?"
    # question = (
    #     "Provide the names of the band members along with the instruments they play."
    # )
    question = "Why is everyone in Starry Sky that day?"

    print("*** ask base rag *** ")
    result = rag.ask_base_rag(query=question)

    # print("*** ask multiquery ***")
    # result = rag.ask_multiquery(query=question)

    print(result["answer"])
