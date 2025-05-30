import sys
import os
from mcp_server.crew_launcher import run_crew_pipeline

# Define python path
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from fastapi import FastAPI
from mcp_server.builder import get_patient_context
from crew_agents.tools import fetch_pubmed_abstracts, fetch_clinical_trials  

# Building mcp with FastAPI
app = FastAPI()

# Builds query for rag/trajectory tools - using only pubmed and clinical trials now
def build_dynamic_evidence_query(temporal_context: str) -> dict:
    context_lower = temporal_context.lower()

    query_terms = {
        "adt": "ADT prostate cancer",
        "psa": "PSA progression prostate cancer",
        "pirads": "PIRADS scoring prostate cancer",
        "bone pain": "bone metastases prostate cancer",
        "weight": "weight change prostate cancer"
    }

    selected_pubmed = []
    selected_trials = []

    for keyword, query in query_terms.items():
        if keyword in context_lower:
            selected_pubmed.append(query)
            selected_trials.append(query)

    if not selected_pubmed:
        selected_pubmed = ["prostate cancer progression"]
        selected_trials = ["prostate cancer treatment outcomes"]

    return {
        "pubmed": selected_pubmed,
        "trials": selected_trials
    }

# Dynamically makes query and retrieves relevant context
@app.get("/context/{patient_id}")
def read_context(patient_id: int):
    context = get_patient_context(patient_id)
    temporal_text = context.get("temporal_context", "").strip()

    search_queries = build_dynamic_evidence_query(temporal_text)

    pubmed_results = []
    trial_results = []

    for query in search_queries["pubmed"]:
        pubmed_results.extend(fetch_pubmed_abstracts(query, max_results=1))

    for query in search_queries["trials"]:
        trial_results.extend(fetch_clinical_trials(query, max_results=1))

    context["clinical_evidence"] = pubmed_results + trial_results
    return context

@app.post("/run-crew")
def run_all_agents(input: dict):
    patient_context = input["context"]
    result = run_crew_pipeline(patient_context)
    return {"crew_output": result}
