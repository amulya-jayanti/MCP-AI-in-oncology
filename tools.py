import pandas as pd
from crewai.tools.base_tool import BaseTool  
from pydantic import BaseModel, Field
from typing import Literal
from Bio import Entrez
import requests

#functions to query pubmed and clinical trials websites
Entrez.email = "zcalianos@uchicago.edu"

def fetch_pubmed_abstracts(query, max_results = 5):
    handle = Entrez.esearch(db = "pubmed", term = query, retmax = max_results)
    record = Entrez.read(handle)
    ids = record["IdList"]
    abstracts = []

    for pmid in ids:
        fetch_handle = Entrez.efetch(db = "pubmed", id=pmid, rettype = "abstract", retmode = "text")
        abstract = fetch_handle.read()
        abstracts.append(abstract.strip())
    return abstracts

#not using clinical trials now, but will add later
def fetch_clinical_trials(keyword, max_results = 5):
    url = f"https://clinicaltrials.gov/api/query/study_fields?expr={keyword}&fields=NCTId,BriefTitle,BriefSummary&min_rnk=1&max_rnk={max_results}&fmt=json"
    res = requests.get(url)

    if res.status_code != 200 or not res.text.strip():
        return ["No clinical trials found or API error."]

    try:
        data = res.json()
        return [
            f"{t['BriefTitle'][0]}: {t['BriefSummary'][0]}"
            for t in data["StudyFieldsResponse"]["StudyFields"]
            if t["BriefTitle"] and t["BriefSummary"]
        ]
    except Exception as e:
        return [f"Failed to parse clinical trials response: {str(e)}"]

#not using this tool now, but keeping in case
class EvidenceLookupToolSchema(BaseModel):
    query: str

class EvidenceLookupTool(BaseTool):
    name: str = "EvidenceLookup"
    description: str = (
        "Searches PubMed and ClinicalTrials.gov for evidence related to a medical query. "
        "Use a single quoted string as input. Example: 'prostate cancer MRI scan effectiveness'."
    )
    args_schema = EvidenceLookupToolSchema

    def _run(self, query: str) -> str:
        if not query or not isinstance(query, str):
            return "Error: 'query' must be a non-empty string describing a clinical concept (e.g., 'PIRADS 4 prostate cancer risk')."

        pubmed = fetch_pubmed_abstracts(query)
        trials = fetch_clinical_trials(query)
        return "\n\n".join(
            ["From PubMed:\n" + abs for abs in pubmed] + 
            ["From ClinicalTrials.gov:\n" + t for t in trials]
        )

#Uses MCP to get patient timelines
class TemporalSummaryToolSchema(BaseModel):
    patient_id: int = Field(..., description="The ID of the patient to summarize context for.")

class TemporalSummaryTool(BaseTool):
    name: str = "TemporalSummaryTool"
    description: str = (
        "Fetches structured longitudinal prostate cancer context for a patient using their ID. "
        "Returns a formatted string of dated clinical entries for summarization."
    )
    args_schema = TemporalSummaryToolSchema

    def _run(self, patient_id: int) -> str:
        try:
            url = f"http://localhost:8000/context/{patient_id}"
            response = requests.get(url)

            if response.status_code != 200:
                return f"API error for patient {patient_id}: {response.text}"

            data = response.json()
            context = data.get("temporal_context", "")

            if not context:
                return f"No clinical history found for patient {patient_id}."

            print(f"[DEBUG] Context fetched from MCP API for patient {patient_id}:\n{context}")  # Optional debug

            summary = context
            return summary

        except Exception as e:
            return f"Error fetching or summarizing context: {str(e)}"


summarize_patient_context = TemporalSummaryTool()

from xml.etree import ElementTree as ET

def extract_pubmed_metadata(pmid: str) -> dict:
    handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml")
    records = Entrez.read(handle)

    try:
        article = records["PubmedArticle"][0]["MedlineCitation"]["Article"]
        title = article["ArticleTitle"]
        authors = article["AuthorList"]
        author_name = authors[0].get("LastName", "Unknown") + " et al." if authors else "Unknown"
        pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year = pub_date.get("Year", "n.d.")
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        abstract = article.get("Abstract", {}).get("AbstractText", [""])[0]

        return {
            "study_title": title,
            "author": author_name,
            "year": year,
            "url": url,
            "finding_summary": abstract[:350].strip() + "..."
        }

    except Exception as e:
        return {
            "study_title": "Unknown",
            "author": "Unknown",
            "year": "n.d.",
            "url": "",
            "finding_summary": f"[Could not parse abstract: {str(e)}]"
        }

#parse patient context, query pubmed, makes citation - RAG tool
class TrajectoryEvidenceToolSchema(BaseModel):
    mcp_context: str = Field(..., description="Structured temporal clinical context from the MCP server")

def analyze_psa_trend(mcp_context: str) -> str:
    lines = mcp_context.splitlines()
    psa_values = []
    for line in lines:
        if "psa=" in line.lower():
            try:
                val = float(line.lower().split("psa=")[1].split("ng/ml")[0])
                psa_values.append(val)
            except:
                continue
    if len(psa_values) >= 2:
        if all(earlier >= later for earlier, later in zip(psa_values, psa_values[1:])):
            return "PSA steadily declining"
        elif any(later > earlier for earlier, later in zip(psa_values, psa_values[1:])):
            return "PSA rising"
    return "PSA stable or fluctuating"

def build_dynamic_clinical_query(mcp_context: str) -> str:
    context_lower = mcp_context.lower()
    query_terms = []

    if "adt" in context_lower:
        query_terms.append("ADT prostate cancer")
    if "psa" in context_lower:
        query_terms.append("PSA trajectory")
    if "weight" in context_lower or "kg" in context_lower:
        query_terms.append("weight change prostate cancer")
    if "pirads" in context_lower:
        query_terms.append("PIRADS progression")
    if "bone pain" in context_lower:
        query_terms.append("bone pain prognosis")
 
    query_terms.append(analyze_psa_trend(mcp_context))
    
    exclusion_terms = "NOT survey NOT preference NOT perception NOT qualitative NOT elbow NOT orthopedic NOT fracture"

    if query_terms:
        return f"({' OR '.join(set(query_terms))}) {exclusion_terms}"
    else:
        return f"prostate cancer disease progression {exclusion_terms}"

def format_rag_comment(study_title: str, author: str, year: str, url: str, finding_summary: str) -> str:
    return (
        f"**2. Clinical Literature Context** — According to "
        f"[{author}, {year}]({url}), {finding_summary} "
        f"(from *{study_title}*)."
    )


class TrajectoryEvidenceTool(BaseTool):
    name: str = "TrajectoryEvidenceTool"
    description: str = (
        "Dynamically generates a PubMed-style clinical citation to explain a patient's observed PSA, weight, or PIRADS trends. "
        "This tool is useful to contextualize progress or stagnation in a patient's condition over time."
    )
    args_schema = TrajectoryEvidenceToolSchema

    def _run(self, mcp_context: str) -> str:
        try:
            query = build_dynamic_clinical_query(mcp_context)

            print(f"[DEBUG] RAG search query = '{query}'")

            search_handle = Entrez.esearch(db="pubmed", term=query, retmax=1)
            search_results = Entrez.read(search_handle)
            ids = search_results.get("IdList", [])

            if not ids:
                return "**2. Clinical Literature Context** — No relevant PubMed study found for this patient."

            metadata = extract_pubmed_metadata(ids[0])
       
            abstract_lower = metadata["finding_summary"].lower()
            generic_phrases = [
                "most common cancer in men",
                "prostate cancer is the most commonly diagnosed",
                "prostate cancer is a leading cause",
            ]

            trajectory_terms = ["psa", "weight", "pirads", "response to therapy", "adt", "treatment outcome"]

            is_generic = any(phrase in abstract_lower for phrase in generic_phrases)
            has_trend_info = any(term in abstract_lower for term in trajectory_terms)

            if is_generic and not has_trend_info:
                return "**2. Clinical Literature Context** — No relevant PubMed study found for this patient."

            if not metadata["url"].startswith("http") or all(
                metadata.get(field, "").lower() in ["unknown", "n.d."]
                for field in ["study_title", "author", "year"]
            ):
                return "**2. Clinical Literature Context** — A relevant study was found but could not be cited due to missing metadata."

            return format_rag_comment(
                study_title=metadata["study_title"] or "Untitled",
                author=metadata["author"] or "Unknown",
                year=metadata["year"] or "n.d.",
                url=metadata["url"],
                finding_summary=metadata["finding_summary"]
            )

        except Exception as e:
            return f"[ERROR generating citation]: {str(e)}"

class TreatmentRecommendationTool:
    name = "treatment_recommendation_tool"
    description = "Use this tool to get the next recommended treatment based on structured clinical features."

    def __call__(self, patient_features: dict) -> str:
        try:
            res = requests.post("http://localhost:8000/recommend-treatment", json=patient_features)
            if res.status_code == 200:
                return res.json().get("recommendation", "No recommendation received.")
            else:
                return f"API call failed: {res.status_code}"
        except Exception as e:
            return f"Error: {e}"
