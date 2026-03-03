import numpy as np
import pandas as pd

np.random.seed(42)
n = 500

df = pd.DataFrame({
    "age": np.random.randint(18, 70, size=n),
    "tenure_months": np.random.randint(1, 72, size=n),
    "monthly_spend": np.random.normal(60, 20, size=n).clip(5, 200),
    "support_tickets": np.random.poisson(1.2, size=n),
    "region": np.random.choice(["north", "south", "east", "west"], size=n),
})

logit = -2.0 + 0.02*(df["monthly_spend"]) + 0.35*(df["support_tickets"]) - 0.02*(df["tenure_months"])
p = 1/(1+np.exp(-logit))
df["churned"] = (np.random.rand(n) < p).astype(int)

df.to_csv("data/sample.csv", index=False)
print("Wrote data/sample.csv")