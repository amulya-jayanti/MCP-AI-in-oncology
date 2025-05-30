from crewai import Agent
from langchain.chat_models import ChatOpenAI
from crew_agents.tools.treatment_tool import TreatmentRecommendationTool

llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)

system_message = (
    "You are a treatment recommendation agent specialized in prostate cancer. "
    "You analyze structured patient data and provide treatment options using clinical model predictions. "
    "You must use the treatment recommendation tool to generate suggestions and avoid making assumptions."
)

treatment_tool = TreatmentRecommendationTool()

treatment_agent = Agent(
    role="Treatment Recommender",
    goal="Propose an evidence-based next-step treatment plan using PSA, Gleason, bone pain, and treatment history",
    backstory="You're a digital oncologist assistant that supports doctors in selecting therapies using ML outputs.",
    tools=[treatment_tool],
    llm=llm,
    verbose=True,
    system_message=system_message
)
