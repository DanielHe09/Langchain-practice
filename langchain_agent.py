import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Get API key from environment variable or use placeholder
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your-api-key-here")

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

# Define some basic tools for the agent
def search_tool(query: str) -> str:
    """A simple search tool example"""
    return f"Search results for: {query}"

def calculator_tool(expression: str) -> str:
    """A simple calculator tool"""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# Create tools list
tools = [
    Tool(
        name="Search",
        func=search_tool,
        description="Useful for searching information. Input should be a search query string."
    ),
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Useful for performing mathematical calculations. Input should be a valid Python expression."
    )
]

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def run_agent(query: str) -> str:
    """Run the agent with a query"""
    try:
        response = agent.run(query)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Example usage
    query = "What is 25 * 4?"
    print(f"Query: {query}")
    print(f"Response: {run_agent(query)}")
