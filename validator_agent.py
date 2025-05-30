from crewai import Agent
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name = "gpt-4-turbo", temperature = 0)

validator_agent = Agent(
    role = 'Clinical Output Validator',
    goal = 'Review generated summaries to ensure they match the provided patient clinical data and highlight any unsupported claims',
    backstory = "You ensure that the final summaries are trustworthy by validating each fact against original data sources.",
    llm = llm,
    verbose = True
)