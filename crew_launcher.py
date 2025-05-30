from crewai import Crew, Task
from crew_agents.summarizer_agent import summarizer_agent
from crew_agents.treatment_agent import treatment_agent
from crew_agents.validator_agent import validator_agent
from crew_agents.planner_agent import planner_agent

def run_crew_pipeline(patient_context: str):
    # Summarization
    summary_task = Task(
        description=f"Summarize the clinical history for the patient using the context:\n{patient_context}",
        expected_output="A clean clinical summary including PSA, weight, bone pain, treatments.",
        agent=summarizer_agent
    )

    # Treatment Recommendation
    treatment_task = Task(
        description="Use the summarized patient data to recommend the next treatment using the treatment tool.",
        expected_output="The most appropriate next treatment step.",
        agent=treatment_agent
    )

    # Validation
    validation_task = Task(
        description="Review the summary and treatment recommendation for hallucination and clinical alignment.",
        expected_output="Valid or Invalid + list of hallucinations.",
        agent=validator_agent
    )

    crew = Crew(
        agents=[summarizer_agent, planner_agent, treatment_agent, validator_agent],
        tasks=[summary_task, treatment_task, validation_task],
        verbose=True
    )

    return crew.kickoff()
