# SETUP

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="ls\_\_xxxxxxxxxxxxxxxxxxxxxx"
LANGCHAIN_PROJECT="the name of the project"

# Dataset

## Key-value datasets

> DEFAULT ONE

```
client.create_example(
 inputs={
   "a-question": "What is the largest mammal?",
   "user-context": "The user is a 1st grader writing a bio report.",
 },
 outputs = {
   "answer": "The blue whale is the largest mammal.",
   "source": "https://en.wikipedia.org/wiki/Blue_whale",
 }
```

## Chat datasets

Datasets with the "chat" data type correspond to messages and generations from LLMs that expect structured "chat" messages as inputs and outputs.
Each example row expects an "inputs" dictionary containing a single "input" key mapped to a list of serialized chat messages.
The "outputs" dictionary contains a single "output" key mapped to a single list of serialized chat messages.

```
client.create_example(
  inputs={
    "input": [
      {"data": {"content": "You are a helpful tutor AI."}, "type": "system"},
      {"data": {"content": "What is the largest mammal?"}, "type": "human"},
    ]},
  outputs={
    "output": {
      "data": {
        "content": "The blue whale is the largest mammal."
        },
      "type": "ai",
    },
  },
```

## LLM datasets

Datasets with the "llm" data type correspond to the string inpts and outputs from the "completion" style LLMS (string in, string out).

```
client.create_example(
  inputs={"input": "What is the largest mammal?"},
  outputs={"output": "The blue whale is the largest mammal."},
  dataset_id=dataset.id,
  # Or dataset_name="My LLM Dataset"
),
```
