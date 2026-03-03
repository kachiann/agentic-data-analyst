import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from src.workflow import run_workflow

import streamlit as st
from src.workflow import run_workflow

st.set_page_config(page_title="Agentic Data Analyst", layout="wide")

st.title("🤖 Autonomous Data Analysis & Report Generation Agent")
st.write("Upload a CSV, provide a goal, and the agent will plan → run tools → generate a grounded report → self-evaluate.")

csv_file = st.file_uploader("Upload CSV", type=["csv"])
goal = st.text_area(
    "What do you want to learn from the data?",
    value="Analyze the dataset and summarize key patterns. If there is a clear target column, build a simple baseline model."
)
run = st.button("Run")

if run:
    if not csv_file:
        st.error("Please upload a CSV file.")
    else:
        path = f"/tmp/{csv_file.name}"
        with open(path, "wb") as f:
            f.write(csv_file.getbuffer())

        with st.spinner("Running agent workflow..."):
            out = run_workflow(goal, path)

        st.subheader("Plan")
        st.json(out["plan"])

        st.subheader("Tool Results (summaries)")
        for r in out["results"]:
            st.write(f"**{r['name']}** — {r['summary']}")

        st.subheader("Evaluation")
        st.json(out["evaluation"])

        st.subheader("Report")
        st.markdown(out["report"])