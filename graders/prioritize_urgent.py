"""Grader for prioritize_urgent task."""
from src.models import Email


def grade_prioritize_urgent(emails: list[Email]) -> float:
    """
    Grade how well urgent emails were prioritized.
    
    High priority (1-2) should be assigned to:
    - Emails from boss@company.com (id 1)
    - Emails with "URGENT" in subject (id 1)
    - Work-related emails that need quick response (id 1, 4)
    """
    urgent_senders = ["boss@company.com"]
    urgent_keywords = ["urgent", "asap", "important", "deadline"]
    
    expected_high_priority = {1, 4}
    
    correct = 0
    total = 0
    
    for email in emails:
        should_be_urgent = (
            email.sender in urgent_senders or
            any(kw in email.subject.lower() for kw in urgent_keywords)
        )
        
        is_actually_urgent = email.priority <= 2
        
        if should_be_urgent and is_actually_urgent:
            correct += 1
        elif not should_be_urgent and not is_actually_urgent:
            correct += 1
        elif should_be_urgent and not is_actually_urgent:
            correct += 0.25
        
        total += 1
    
    return min(correct / total if total > 0 else 0.0, 1.0)
