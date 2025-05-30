from crewai import Agent
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name = "gpt-4-turbo", temperature = 0)

planner_agent = Agent(
    role = 'Timeline Planner',
    goal = 'Identify and organize key clinical events in the patient history chronologically',
    backstory = "You're an expert in clinical workflow and disease progression. You help organize medical data into clear timelines for downstream summarization.",
    llm = llm,
    verbose = True
)
