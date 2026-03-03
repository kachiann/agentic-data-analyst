from __future__ import annotations
import duckdb
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

@dataclass
class ToolResult:
    name: str
    summary: str
    data: dict

class DataTools:
    """
    Safe, bounded tools: no arbitrary code execution.
    """
    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.con = duckdb.connect(database=":memory:")

    def load_csv(self, path: str) -> ToolResult:
        df = pd.read_csv(path)
        self.df = df
        self.con.register("data", df)
        return ToolResult(
            name="load_csv",
            summary=f"Loaded CSV with shape={df.shape} and {df.shape[1]} columns.",
            data={"shape": df.shape, "columns": list(df.columns)}
        )

    def preview(self, n: int = 5) -> ToolResult:
        self._require_df()
        return ToolResult("preview", f"Previewed first {n} rows.", {"head": self.df.head(n).to_dict(orient="records")})

    def schema(self) -> ToolResult:
        self._require_df()
        dtypes = {c: str(t) for c, t in self.df.dtypes.items()}
        return ToolResult("schema", "Captured dataframe schema (dtypes).", {"dtypes": dtypes})

    def missingness(self) -> ToolResult:
        self._require_df()
        miss = self.df.isna().sum().sort_values(ascending=False)
        miss_pct = (miss / len(self.df)).round(4)
        out = pd.DataFrame({"missing": miss, "missing_pct": miss_pct})
        return ToolResult("missingness", "Computed missing values per column.", {"missing_table": out.reset_index(names="column").to_dict(orient="records")})

    def describe_numeric(self) -> ToolResult:
        self._require_df()
        num = self.df.select_dtypes(include=[np.number])
        if num.shape[1] == 0:
            return ToolResult("describe_numeric", "No numeric columns found.", {"describe": []})
        desc = num.describe().transpose()
        return ToolResult("describe_numeric", "Computed numeric descriptive statistics.", {"describe": desc.reset_index(names="column").to_dict(orient="records")})

    def correlations(self, top_k: int = 10) -> ToolResult:
        self._require_df()
        num = self.df.select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            return ToolResult("correlations", "Not enough numeric columns to compute correlations.", {"pairs": []})
        corr = num.corr(numeric_only=True).abs()
        pairs = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                pairs.append((cols[i], cols[j], float(corr.iloc[i, j])))
        pairs.sort(key=lambda x: x[2], reverse=True)
        top = pairs[:top_k]
        return ToolResult("correlations", f"Computed top {top_k} absolute correlations.", {"pairs": [{"col_a": a, "col_b": b, "abs_corr": v} for a, b, v in top]})

    def sql_query(self, query: str, limit: int = 50) -> ToolResult:
        self._require_df()
        q = query.strip().rstrip(";")
        res = self.con.execute(q).fetchdf().head(limit)
        return ToolResult("sql_query", f"Executed SQL query (up to {limit} rows).", {"rows": res.to_dict(orient="records")})

    def train_regression(self, target: str, features: list[str], model: str = "linear") -> ToolResult:
        self._require_df()
        X, y = self._prepare_xy(target, features)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        m = LinearRegression() if model == "linear" else DecisionTreeRegressor(random_state=42, max_depth=6)
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        mae = float(mean_absolute_error(y_test, pred))
        r2 = float(r2_score(y_test, pred))

        return ToolResult("train_regression", f"Trained {model} regression. MAE={mae:.4f}, R2={r2:.4f}", {"mae": mae, "r2": r2, "features": features, "target": target, "model": model})

    def train_classification(self, target: str, features: list[str], model: str = "logreg") -> ToolResult:
        self._require_df()
        X, y = self._prepare_xy(target, features)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)

        m = LogisticRegression(max_iter=1000) if model == "logreg" else DecisionTreeClassifier(random_state=42, max_depth=6)
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        acc = float(accuracy_score(y_test, pred))

        auc = None
        if y.nunique() == 2 and hasattr(m, "predict_proba"):
            proba = m.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, proba))

        return ToolResult("train_classification", f"Trained {model} classifier. Accuracy={acc:.4f}" + (f", ROC-AUC={auc:.4f}" if auc is not None else ""), {"accuracy": acc, "roc_auc": auc, "features": features, "target": target, "model": model})

    def _prepare_xy(self, target: str, features: list[str]):
        self._require_df()
        for c in [target] + features:
            if c not in self.df.columns:
                raise ValueError(f"Column not found: {c}")

        df = self.df[features + [target]].dropna().copy()
        y = df[target]
        X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
        return X, y

    def _require_df(self):
        if self.df is None:
            raise ValueError("No dataframe loaded. Call load_csv first.")