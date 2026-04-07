"""Grader for categorize_inbox task."""
from src.models import Email


def grade_categorize_inbox(emails: list[Email]) -> float:
    """
    Grade how well emails were categorized.
    
    Expected categorizations based on sender/content:
    - boss@company.com -> work
    - colleague@company.com -> work  
    - mom@example.com -> personal
    - newsletter@spam.com, spam -> spam
    """
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
    
    return min(score / total if total > 0 else 0.0, 1.0)
