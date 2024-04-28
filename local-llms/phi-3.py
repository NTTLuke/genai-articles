# Example of using the OpenAI Python client to interact with a local OpenAI server with LM Studio (server mode enabled).
# Model used in this example: microsoft/Phi-3-mini-4k-instruct-gguf
# run the code with `python local-llms/phi-3.py`

from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


# question = "What kind of music does Luca like to play and what instrument does he play?"
question = "What is Luca's professions and where is he from?"
context_info = """I'm Luca and I'm a guitar player from Italy. I love to play rock music and I'm looking for a band to join. I'm also a music producer for an italian songwriter."""


def generate_questions():
    # Create 5 questions based on the context provided
    stream = client.chat.completions.create(
        model="microsoft/Phi-3-mini-4k-instruct-gguf",
        messages=[
            {
                "role": "system",
                "content": "Generate ONLY FIVE questions based on the context provided. The answer of each question can be retrieved from the context. Questions must be different. DO NOT add any new information.",
            },
            {
                "role": "user",
                "content": f"""Here the context: 
                --- {context_info} --- 
                Response must be in this format:
                Number of the question : Question
                Example: 1: What is your name?

                When you reach the number of the questions asked, please stop generating more questions.
                """,
            },
        ],
        temperature=0.2,
        stream=True,
    )

    return stream


# Ask a question about a person
def answer_with_context():
    stream = client.chat.completions.create(
        model="microsoft/Phi-3-mini-4k-instruct-gguf",
        messages=[
            {
                "role": "system",
                "content": "Always answer based on the provided context. Do not add any new information.",
            },
            {
                "role": "user",
                "content": f""" 
                
                Question: 
                {question} 
                
                Context: 
                {context_info}. 
                
                Do not add any extra information.""",
            },
        ],
        temperature=0.2,
        stream=True,
    )

    return stream


if __name__ == "__main__":
    # generate_questions()

    stream = answer_with_context()

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
