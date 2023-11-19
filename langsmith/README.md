# Test and Evaluation of ConversationalRetrieval and MultiQuery Approaches with Langsmith

## Overview

This script enables conversational question-answering from PDF documents using a combination of LangChain components and Azure OpenAI services. It first processes a PDF document, splits its content into manageable parts, embeds these parts into a vector space using Azure OpenAI embeddings, and then retrieves and answers questions based on this embedded content.

## Features

- **PDF Processing:** Uses `PyPDFLoader` to load PDF content.
- **Text Splitting:** Splits PDF text into chunks using `TokenTextSplitter`.
- **Embedding Generation:** Converts text chunks into embeddings with `AzureOpenAIEmbeddings`.
- **Vector Storage:** Stores embeddings in `Chroma` vector database.
- **Question Answering:** Supports both single and multi-query retrieval methods for answering questions.

## Dependencies

- langchain
- os
- dotenv

## Setup

Load environment variables:

```python
load_dotenv()
```

Set the path for the PDF file and the directory to persist embeddings:

```python
persist_directory = "data/chroma"
pdf_path = os.getenv("FILE_PATH")
```

## Usage

### Initialize Components

Create instances for embeddings, LangChain Large Language Model (LLM), and the vector database:

```python
embedding = AzureOpenAIEmbeddings(deployment="embedding deployment name", chunk_size=1, embedding_ctx_length=1000)
llm = AzureChatOpenAI(azure_deployment="azure deployment name", openai_api_type="azure", temperature=0.0)
vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
```

### Load and Process PDF

Function to load and process a PDF document:

```python
def load_pdf(doc_path: str, embedding):
    # Implementation details...
```

### Ask Questions

Function to ask a question and get an answer:

```python
def ask_question(question: str):
    # Implementation details...
```

### Multi-Query Retrieval

Function for multi-query retrieval and question answering:

```python
def ask_question_multiquery(question: str):
    # Implementation details...
```

### Main Execution

Example of how to use the script:

```python
if __name__ == "__main__":
    question = "Quale strumento suona Fred?"
    ask_question_multiquery(question=question)
```

> The question is about the PDF I used to check things. I used an online version of the picture book for children that I wrote. You can find it at https://amzn.eu/d/1gVvNUq.❤️

# Testing and Evaluating with Langsmith

see notebook : `book-chat-test-eval`

TODO DOCUMENTATION
