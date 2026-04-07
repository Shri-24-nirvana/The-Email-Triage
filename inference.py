"""
Email Triage Agent - Baseline Inference Script
Uses OpenAI Client for LLM-based email triage decisions.

Required environment variables:
- API_BASE_URL: The API endpoint for the LLM
- MODEL_NAME: The model identifier to use
- HF_TOKEN: Your Hugging Face / API key
"""

import os
import json
import sys
from typing import Any

from openai import OpenAI
from src.environment import EmailTriageEnv
from src.models import Email
from graders import GRADERS

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/gradients/latest")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

TASKS = ["categorize_inbox", "prioritize_urgent", "archive_clutter"]
MAX_STEPS = 20


def create_system_prompt(task_id: str) -> str:
    prompts = {
        "categorize_inbox": """You are an email triage assistant. Your task is to categorize emails into: work, personal, or spam.
Rules:
- Work emails are from colleagues, bosses, or about work matters
- Personal emails are from family and friends
- Spam emails are suspicious newsletters or scams
Output ONLY JSON with format: {"action_type": "categorize", "email_id": <id>, "category": "<category>"}""",
        
        "prioritize_urgent": """You are an email triage assistant. Your task is to prioritize emails on a scale of 1-5 (1=highest priority).
Rules:
- Priority 1-2: Urgent emails needing immediate attention
- Priority 3: Normal emails
- Priority 4-5: Low priority emails
Output ONLY JSON with format: {"action_type": "prioritize", "email_id": <id>, "priority": <1-5>}""",
        
        "archive_clutter": """You are an email triage assistant. Your task is to archive old non-urgent emails.
Rules:
- Archive emails older than 7 days with low priority (4-5)
- Do NOT archive urgent emails (priority 1-2)
Output ONLY JSON with format: {"action_type": "archive", "email_id": <id>}"""
    }
    return prompts.get(task_id, prompts["categorize_inbox"])


def create_user_prompt(observation: dict) -> str:
    emails = observation["emails"]
    inbox_emails = [e for e in emails if e.get("category", "inbox") == "inbox"]
    
    if not inbox_emails:
        return "No emails to triage. Task complete."
    
    email_list = "\n".join([
        f"ID: {e['id']} | From: {e['sender']} | Subject: {e['subject']} | Priority: {e.get('priority', 3)}"
        for e in inbox_emails[:5]
    ])
    
    return f"Emails to triage:\n{email_list}"


def parse_llm_response(response_text: str) -> dict | None:
    try:
        for line in response_text.split("\n"):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        try:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end != 0:
                return json.loads(response_text[start:end])
        except:
            pass
    return None


def run_task(task_id: str, env: EmailTriageEnv) -> dict:
    obs = env.reset(task_id)
    
    print(f"[START] task_id={task_id}")
    sys.stdout.flush()
    
    for step in range(MAX_STEPS):
        state = env.state()
        if state["done"]:
            break
        
        prompt = create_user_prompt(obs)
        system_prompt = create_system_prompt(task_id)
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.1,
            )
            
            response_text = response.choices[0].message.content
            action = parse_llm_response(response_text)
            
            if action is None:
                print(f"[STEP] step={step + 1} error='Failed to parse LLM response'")
                sys.stdout.flush()
                continue
            
            obs, reward, done = env.step(action)
            
            print(f"[STEP] step={step + 1} action={json.dumps(action)} reward={reward:.4f} done={done}")
            sys.stdout.flush()
            
            if done:
                break
                
        except Exception as e:
            print(f"[STEP] step={step + 1} error='{str(e)}'")
            sys.stdout.flush()
    
    final_state = env.state()
    emails = [Email(**e) for e in final_state["observation"]["emails"]]
    grader = GRADERS.get(task_id, GRADERS["categorize_inbox"])
    score = grader(emails)
    
    print(f"[END] task_id={task_id} score={score:.4f}")
    sys.stdout.flush()
    
    return {"task_id": task_id, "score": score}


def main():
    print(f"[START] model={MODEL_NAME} tasks={TASKS}")
    sys.stdout.flush()
    
    env = EmailTriageEnv(seed=42)
    results = []
    
    for task_id in TASKS:
        result = run_task(task_id, env)
        results.append(result)
    
    total_score = sum(r["score"] for r in results) / len(results)
    
    print(f"\n[SUMMARY] average_score={total_score:.4f}")
    print(f"[RESULTS] {json.dumps(results, indent=2)}")
    sys.stdout.flush()
    
    return {"average_score": total_score, "results": results}


if __name__ == "__main__":
    main()
