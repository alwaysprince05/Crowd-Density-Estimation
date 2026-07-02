def calculate_density(count: int) -> int:
    return count

def classify_density(count: int) -> str:
    if count < 15:
        return "LOW"
    elif count < 40:
        return "MODERATE"
    elif count < 80:
        return "HIGH"
    else:
        return "CRITICAL"
