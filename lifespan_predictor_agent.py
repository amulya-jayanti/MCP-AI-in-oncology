from crew_agents.lifespan_model_utils import predict_lifespan

def run_lifespan_prediction(patient_data: dict) -> dict:
    try:
        patient_id = patient_data["patient_id"]
        prediction = predict_lifespan(patient_id)
        return {"lifespan_years": prediction, "status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# if __name__ == "__main__":
#     test_input = {"patient_id": 1}
#     result = run_lifespan_prediction(test_input)
#     print(result)


