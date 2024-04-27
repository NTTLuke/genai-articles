from crewai import Agent, Task, Crew, Process
from interpreter import interpreter
from langchain.tools import tool
from langchain_openai.chat_models.azure import AzureChatOpenAI
import os
import sys


# 1. Setup LLM and Interpreter
llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    temperature=0.1,
)


interpreter.auto_run = True
interpreter.llm.api_key = os.getenv("AZURE_OPENAI_API_KEY")
interpreter.llm.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
interpreter.llm.api_version = os.getenv("OPENAI_API_VERSION")
interpreter.llm.model = "azure/gpt-4"


# 2. Define Open Interpreter as client tool
class CLITool:
    @tool("Executor")
    def execute_command(command: str):
        """Create and Execute code using Open Interpreter."""
        result = interpreter.chat(command)
        return result


# 3. Creating an Agent for CLI tasks
cli_agent = Agent(
    role="Software Engineer",
    goal="Always use Executor Tool. Ability to perform CLI operations, write programs and execute using Exector Tool",
    backstory="Expert in command line operations, creating and executing code.",
    tools=[CLITool.execute_command],
    verbose=True,
    llm=llm,
)

# 4. Defining a Task for CLI operations
wallpapers_path = os.path.join(os.getcwd(), "wallpapers")
cli_task = Task(
    description=f"Identify the OS and then change background wallpaper with a random one retrieved from this directory: '{wallpapers_path}'",
    agent=cli_agent,
    tools=[CLITool.execute_command],
    expected_output="Response message about the task executed",
)

# 5. Creating a Crew with CLI focus
cli_crew = Crew(agents=[cli_agent], tasks=[cli_task])


# 6. Run the Crew
result = cli_crew.kickoff()
print(result)
