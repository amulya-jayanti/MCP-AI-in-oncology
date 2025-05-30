from crewai import Agent
from crew_agents.tools import EvidenceLookupTool, summarize_patient_context  
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name = "gpt-4-turbo", temperature = 0)

system_message = (
    "You are a clinical summarization agent. "
    "Never invent or fill in missing demographic information. "
    "Do not mention name, gender, date of birth, contact info, or address unless explicitly provided in the input. "
    "If a value is missing, either omit it or write 'Not available in the data'. "
    "Do not include placeholders like 'John Doe' or 'Dr. Smith'. "
    "Only report what is in the input or returned from the TemporalSummary tool."
)

evidence_tool = EvidenceLookupTool()

summarizer_agent = Agent(
    role = "LLM Clinical Summarizer",
    goal = "Create a temporally-aware summary using clinical records and supporting literature",
    backstory = "You're an expert in generating grounded clinical narratives. You use structured data and external publications to build trustworthy, clear summaries.",
    tools = [evidence_tool, summarize_patient_context],  
    llm = llm,
    verbose = True,
    system_message = system_message
)
