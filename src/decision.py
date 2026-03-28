def get_decision(label, confidence):
    if label == "ambulance" and confidence >= 0.75:
        return "🚑 High Priority Emergency"

    if confidence >= 0.85:
        return "High Confidence"
    elif confidence >= 0.65:
        return "Needs Review"
    else:
        return "Uncertain"