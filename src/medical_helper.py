def detect_medical_entities(question):

    diseases = [
        "diabetes",
        "asthma",
        "cancer",
        "leukemia",
        "hypertension",
        "covid",
        "tuberculosis"
    ]

    symptoms = [
        "fever",
        "cough",
        "headache",
        "fatigue",
        "pain",
        "nausea"
    ]

    treatments = [
        "chemotherapy",
        "radiation",
        "insulin",
        "surgery",
        "medication"
    ]

    detected = []

    question = question.lower()

    for disease in diseases:
        if disease in question:
            detected.append(
                f"Disease: {disease}"
            )

    for symptom in symptoms:
        if symptom in question:
            detected.append(
                f"Symptom: {symptom}"
            )

    for treatment in treatments:
        if treatment in question:
            detected.append(
                f"Treatment: {treatment}"
            )

    return detected