"""
Email Triage Agent - Baseline Inference Script
Rule-based email triage (no LLM required).

Required environment variables:
- API_BASE_URL: The API endpoint for the LLM (optional, uses rules if not set)
- MODEL_NAME: The model identifier to use
- HF_TOKEN: Your Hugging Face / API key
"""

import os
import json
import sys

from src.environment import EmailTriageEnv
from src.models import Email
from graders import GRADERS

TASKS = ["categorize_inbox", "prioritize_urgent", "archive_clutter"]
MAX_STEPS = 20


def clamp_score(score: float) -> float:
    """Ensure score is strictly between 0 and 1 (not 0.0 or 1.0)."""
    if score <= 0.0:
        return 0.01
    elif score >= 1.0:
        return 0.99
    return score


def categorize_rule_based(emails: list[dict]) -> dict:
    """Rule-based categorization."""
    for email in emails:
        if email.get("category") == "inbox":
            sender = email.get("sender", "").lower()
            subject = email.get("subject", "").lower()
            
            if "boss" in sender or "colleague" in sender or "company" in sender:
                return {"action_type": "categorize", "email_id": email["id"], "category": "work"}
            elif "mom" in sender or "family" in sender or "personal" in sender:
                return {"action_type": "categorize", "email_id": email["id"], "category": "personal"}
            elif "spam" in sender or "newsletter" in sender or "won" in subject:
                return {"action_type": "categorize", "email_id": email["id"], "category": "spam"}
    
    return {"action_type": "categorize", "email_id": emails[0]["id"], "category": "work"}


def prioritize_rule_based(emails: list[dict]) -> dict:
    """Rule-based prioritization."""
    for email in emails:
        if email.get("category") == "inbox":
            sender = email.get("sender", "").lower()
            subject = email.get("subject", "").lower()
            
            if "boss" in sender or "urgent" in subject:
                return {"action_type": "prioritize", "email_id": email["id"], "priority": 1}
            elif "colleague" in sender:
                return {"action_type": "prioritize", "email_id": email["id"], "priority": 2}
    
    return {"action_type": "prioritize", "email_id": emails[0]["id"], "priority": 3}


def archive_rule_based(emails: list[dict]) -> dict:
    """Rule-based archiving."""
    for email in emails:
        if email.get("category") == "inbox":
            priority = email.get("priority", 3)
            sender = email.get("sender", "").lower()
            
            if priority > 2 and "boss" not in sender:
                return {"action_type": "archive", "email_id": email["id"]}
    
    return {"action_type": "archive", "email_id": emails[0]["id"]}


def run_task(task_id: str, env: EmailTriageEnv) -> dict:
    obs = env.reset(task_id)
    
    print(f"[START] task_id={task_id}")
    sys.stdout.flush()
    
    for step in range(MAX_STEPS):
        state = env.state()
        if state["done"]:
            break
        
        emails = state["observation"]["emails"]
        inbox_emails = [e for e in emails if e.get("category") == "inbox"]
        
        if not inbox_emails:
            break
        
        if task_id == "categorize_inbox":
            action = categorize_rule_based(inbox_emails)
        elif task_id == "prioritize_urgent":
            action = prioritize_rule_based(inbox_emails)
        else:
            action = archive_rule_based(inbox_emails)
        
        obs, reward, done = env.step(action)
        
        print(f"[STEP] step={step + 1} action={json.dumps(action)} reward={reward:.4f} done={done}")
        sys.stdout.flush()
        
        if done:
            break
    
    final_state = env.state()
    emails = [Email(**e) for e in final_state["observation"]["emails"]]
    grader = GRADERS.get(task_id, GRADERS["categorize_inbox"])
    raw_score = grader(emails)
    score = clamp_score(raw_score)
    
    print(f"[END] task_id={task_id} score={score:.4f}")
    sys.stdout.flush()
    
    return {"task_id": task_id, "score": score}


def main():
    print(f"[START] model=rule-based tasks={TASKS}")
    sys.stdout.flush()
    
    env = EmailTriageEnv(seed=42)
    results = []
    
    for task_id in TASKS:
        result = run_task(task_id, env)
        results.append(result)
    
    scores = [r["score"] for r in results]
    total_score = clamp_score(sum(scores) / len(scores))
    
    print(f"\n[SUMMARY] average_score={total_score:.4f}")
    print(f"[RESULTS] {json.dumps(results, indent=2)}")
    sys.stdout.flush()
    
    return {"average_score": total_score, "results": results}


if __name__ == "__main__":
    main()
