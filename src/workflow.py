from __future__ import annotations
from .tools import DataTools
from .planner import make_plan
from .report import generate_report
from .evaluator import evaluate_report

ALLOWED_TOOLS = {
    "preview", "schema", "missingness", "describe_numeric", "correlations",
    "sql_query", "train_regression", "train_classification"
}

def run_workflow(user_goal: str, csv_path: str):
    tools = DataTools()
    results = []

    # Always load first
    results.append(tools.load_csv(csv_path).__dict__)

    # Plan
    plan = make_plan(user_goal=user_goal, columns=tools.df.columns.tolist())

    # Execute plan steps
    for step in plan.steps:
        if step.tool not in ALLOWED_TOOLS:
            continue
        fn = getattr(tools, step.tool)
        tr = fn(**step.args)
        results.append(tr.__dict__)

    # Report
    report = generate_report(user_goal, results)

    # Evaluate
    eval_res = evaluate_report(
        user_goal=user_goal,
        tool_summaries=[r["summary"] for r in results],
        draft_report=report,
    )

    # If revise, regenerate once with evaluator feedback
    if eval_res.verdict == "revise":
        report = report + "\n\n---\nEvaluator issues:\n- " + "\n- ".join(eval_res.issues) + "\nSuggested fixes:\n- " + "\n- ".join(eval_res.suggested_fixes)

    return {"plan": plan.model_dump(), "results": results, "report": report, "evaluation": eval_res.model_dump()}