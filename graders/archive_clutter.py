"""Grader for archive_clutter task."""
from datetime import datetime, timedelta
from src.models import Email


def grade_archive_clutter(emails: list[Email]) -> float:
    """
    Grade how well old non-urgent emails were archived.
    
    Emails older than 7 days with priority > 2 should be archived.
    Urgent emails (priority 1-2) should NOT be archived.
    """
    try:
        cutoff = datetime.now() - timedelta(days=7)
    except:
        cutoff = datetime(2026, 4, 1)
    
    old_non_urgent = 0
    correctly_archived = 0
    incorrectly_archived = 0
    
    for email in emails:
        try:
            email_date = datetime.fromisoformat(email.timestamp.replace("Z", "+00:00"))
            if email_date.tzinfo:
                email_date = email_date.replace(tzinfo=None)
        except:
            email_date = datetime(2026, 3, 25)
        
        is_old = email_date < cutoff
        is_urgent = email.priority <= 2
        is_archived = email.category == "archived"
        
        if is_old and not is_urgent:
            old_non_urgent += 1
            if is_archived:
                correctly_archived += 1
        elif is_urgent and is_archived:
            incorrectly_archived += 1
    
    if old_non_urgent == 0:
        return 1.0
    
    precision = correctly_archived / (correctly_archived + incorrectly_archived) if (correctly_archived + incorrectly_archived) > 0 else 0
    recall = correctly_archived / old_non_urgent if old_non_urgent > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return min(f1, 1.0)
