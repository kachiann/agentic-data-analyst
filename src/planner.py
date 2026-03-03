from __future__ import annotations
from pydantic import BaseModel, Field
from openai import OpenAI
from .config import OPENAI_MODEL, OPENAI_API_KEY

class PlanStep(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)
    purpose: str

class AnalysisPlan(BaseModel):
    goal: str
    steps: list[PlanStep]

ALLOWED_TOOLS = [
    "preview", "schema", "missingness", "describe_numeric", "correlations",
    "sql_query", "train_regression", "train_classification"
]

def make_plan(user_goal: str, columns: list[str]) -> AnalysisPlan:
    client = OpenAI(api_key=OPENAI_API_KEY)
    system = (
        "You are a planning agent for a data analysis system.\n"
        "You must output a JSON plan with goal and steps.\n"
        f"Use ONLY these tools: {', '.join(ALLOWED_TOOLS)}.\n"
        "Prefer EDA steps first. Only include modeling if goal implies prediction/explanation and target is clear.\n"
        "If you propose modeling, you must choose target and <=8 features from provided columns."
    )
    resp = client.responses.parse(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Goal: {user_goal}\nColumns: {columns}\nCreate plan."},
        ],
        text_format=AnalysisPlan
    )
    return resp.output_parsed