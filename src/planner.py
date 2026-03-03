from __future__ import annotations
from typing import Literal, Optional, List
from pydantic import BaseModel, Field
from openai import OpenAI
from .config import OPENAI_MODEL, OPENAI_API_KEY

# ---- Tool argument schemas (closed objects; no free-form dict) ----

class PreviewArgs(BaseModel):
    n: int = Field(5, ge=1, le=50)

class CorrelationsArgs(BaseModel):
    top_k: int = Field(10, ge=1, le=50)

class SQLQueryArgs(BaseModel):
    query: str
    limit: int = Field(50, ge=1, le=200)

class TrainRegressionArgs(BaseModel):
    target: str
    features: List[str]
    model: Literal["linear", "tree"] = "linear"

class TrainClassificationArgs(BaseModel):
    target: str
    features: List[str]
    model: Literal["logreg", "tree"] = "logreg"

# Union-like container: one of these will be filled depending on tool
class StepArgs(BaseModel):
    preview: Optional[PreviewArgs] = None
    correlations: Optional[CorrelationsArgs] = None
    sql_query: Optional[SQLQueryArgs] = None
    train_regression: Optional[TrainRegressionArgs] = None
    train_classification: Optional[TrainClassificationArgs] = None

class PlanStep(BaseModel):
    tool: Literal[
        "preview",
        "schema",
        "missingness",
        "describe_numeric",
        "correlations",
        "sql_query",
        "train_regression",
        "train_classification",
    ]
    args: StepArgs = Field(default_factory=StepArgs)
    purpose: str

class AnalysisPlan(BaseModel):
    goal: str
    steps: List[PlanStep]

ALLOWED_TOOLS = [
    "preview", "schema", "missingness", "describe_numeric", "correlations",
    "sql_query", "train_regression", "train_classification"
]

def make_plan(user_goal: str, columns: list[str]) -> AnalysisPlan:
    client = OpenAI(api_key=OPENAI_API_KEY)
    system = (
        "You are a planning agent for a data analysis system.\n"
        "Return a JSON plan with goal and steps.\n"
        f"Use ONLY these tools: {', '.join(ALLOWED_TOOLS)}.\n"
        "Prefer EDA steps first.\n"
        "Only include modeling if the goal implies prediction and a target is clear.\n"
        "If you propose modeling, choose target and <=8 features from provided columns.\n\n"
        "IMPORTANT: Put tool arguments inside args.<tool_name>.\n"
        "Examples:\n"
        "- preview: args.preview={\"n\": 5}\n"
        "- correlations: args.correlations={\"top_k\": 10}\n"
        "- sql_query: args.sql_query={\"query\": \"SELECT ... FROM data\", \"limit\": 50}\n"
        "- train_regression: args.train_regression={\"target\":\"y\",\"features\":[...],\"model\":\"linear\"}\n"
    )

    resp = client.responses.parse(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Goal: {user_goal}\nColumns: {columns}\nCreate a plan."},
        ],
        text_format=AnalysisPlan
    )
    return resp.output_parsed