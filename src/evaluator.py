from __future__ import annotations
from pydantic import BaseModel, Field
from openai import OpenAI
from .config import OPENAI_MODEL, OPENAI_API_KEY

class EvalResult(BaseModel):
    verdict: str = Field(..., description="pass or revise")
    issues: list[str] = Field(default_factory=list)
    suggested_fixes: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)

def evaluate_report(user_goal: str, tool_summaries: list[str], draft_report: str) -> EvalResult:
    client = OpenAI(api_key=OPENAI_API_KEY)
    system = (
        "You are a strict evaluator for a tool-grounded data analysis agent.\n"
        "Rules: any numeric claim must be supported by the tool summaries.\n"
        "If unsupported claims exist, return verdict=revise and list issues + fixes."
    )
    resp = client.responses.parse(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Goal: {user_goal}\n\nTool summaries:\n- " + "\n- ".join(tool_summaries) + f"\n\nDraft report:\n{draft_report}"},
        ],
        text_format=EvalResult
    )
    return resp.output_parsed