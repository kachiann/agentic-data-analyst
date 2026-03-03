from __future__ import annotations
from openai import OpenAI
from .config import OPENAI_MODEL, OPENAI_API_KEY

def generate_report(user_goal: str, tool_results: list[dict]) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    system = (
        "You are a data analyst. Write a concise report grounded ONLY in the provided tool results.\n"
        "Structure:\n"
        "1) Executive Summary (3-5 bullets)\n"
        "2) Key Findings\n"
        "3) Methods\n"
        "4) Limitations / Next Steps\n"
        "Do not invent numbers or claims not present in tool results."
    )
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Goal: {user_goal}\n\nTool results:\n{tool_results}"},
        ],
    )
    return resp.output_text.strip()