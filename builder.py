import pandas as pd
from datetime import datetime, timedelta
import re

#parse patient data - returns date and note
def parse_notes(raw_notes: str):
    events = []
    entries = raw_notes.split('|')
    for entry in entries:
        parts = entry.strip().split(': ', 1)
        if len(parts) != 2:
            continue
        date_str, content = parts
        try:
            date = datetime.strptime(date_str.strip(), "%Y-%m-%d")
        except ValueError:
            continue
        events.append({"date": date, "note": content.strip()})
    return events

#uses parse_notes to get patient data for patient_id
def get_patient_context(patient_id: int, months: int = 12):
    df = pd.read_csv("data/prostate_patient_data.csv")

    patient_row = df[df["PatientID"] == patient_id]

    if patient_row.empty:
        return {"error": "Patient not found"}

    raw_notes = patient_row.iloc[0]["Clinical Notes"]
    all_events = parse_notes(raw_notes)

    context_summary = "\n".join([f"{e['date'].date()}: {e['note']}" for e in all_events])

    return {
        "patient_id": patient_id,
        "event_count": len(all_events),
        "temporal_context": context_summary
}



