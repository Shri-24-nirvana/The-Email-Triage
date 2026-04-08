"""Grader for categorize_inbox task."""
from src.models import Email


def grade_categorize_inbox(emails: list[Email]) -> float:
    """
    Grade how well emails were categorized.
    Score is strictly between 0 and 1 (never 0.0 or 1.0).
    """
    if not emails:
        return 0.5
    
    score = 0.0
    total = len(emails)
    
    expected = {
        1: "work",
        2: "personal",
        3: "spam",
        4: "work",
        5: "spam",
    }
    
    for email in emails:
        if email.category != "inbox":
            if email.id in expected and email.category == expected[email.id]:
                score += 1.0
            elif email.id in expected:
                score += 0.5
            else:
                score += 0.25
    
    result = (score / total) if total > 0 else 0.0
    
    # Strictly between 0 and 1
    if result <= 0.0:
        return 0.001
    if result >= 1.0:
        return 0.999
    
    # Add small random variation to avoid exact boundaries
    import random
    variation = random.uniform(-0.0001, 0.0001)
    result = max(0.001, min(0.999, result + variation))
    
    return round(result, 4)
