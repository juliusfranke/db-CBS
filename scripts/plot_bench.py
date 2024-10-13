import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic

data = pd.read_parquet("../results/bench.parquet")
baseline = pd.read_parquet("../results/baseline.parquet")
data["score"] = data["duration_dbcbs"] ** 2 + data["cost"] ** 2
baseline["score"] = baseline["duration_dbcbs"] ** 2 + baseline["cost"] ** 2
baseline_mean = (
    baseline[baseline["mp_name"] == "Baseline"]
    .groupby("size")[["success", "cost", "duration_dbcbs", "score"]]
    .mean()
)
baseline_std = (
    baseline[baseline["mp_name"] == "Baseline"]
    .groupby("size")[["success", "cost", "duration_dbcbs", "score"]]
    .std()
)
ic(baseline_mean)
# baseline_n = len(data[(data["mp_name"] == "Baseline") & (data["size"] == 100)])
# ci_95 = 1.96 / np.sqrt(baseline_n)

data = data[data["mp_name"] != "Baseline"]
for size in data["size"].unique():
    for param in ["duration_dbcbs", "success", "cost", "score"]:
        data.loc[data["size"] == size, f"{param}_bl"] = (
            data.loc[data["size"] == size][param] - baseline_mean.loc[size, param]
        ) / baseline_mean.loc[size, param]


# breakpoint()
# sns.lmplot(data, x="duration_dbcbs", y="cost", hue="mp_name")
# plt.yscale("log")
# plt.xscale("log")
# plt.showee
# breakpoint()
data_mean = data.groupby(["size", "mp_name"])[
    ["success", "cost", "duration_dbcbs", "score"]
].mean()
ic(data_mean)
# breakpoint()
# sns.lineplot(data, x="size", y="success_bl", hue="mp_name")
sns.barplot(data, x="size", y="success_bl", hue="mp_name")
plt.axhline(y=0, color="red", linestyle="--", label="Baseline mean")
# plt.axhline(y=ci_95, color="red", linestyle="--", label="Baseline mean")
# plt.axhline(y=-ci_95, color="red", linestyle="--", label="Baseline mean")
plt.show()
# sns.lineplot(data, x="size", y="duration_dbcbs_bl", hue="mp_name")
sns.boxplot(data, x="size", y="duration_dbcbs_bl", hue="mp_name", showfliers=False)
plt.axhline(y=0, color="red", linestyle="--", label="Baseline mean")
# plt.yscale("log")
plt.show()
sns.boxplot(data, x="size", y="cost_bl", hue="mp_name", showfliers=False)
# sns.lineplot(data, x="size", y="cost_bl", hue="mp_name")
plt.axhline(y=0, color="red", linestyle="--", label="Baseline mean")
# plt.yscale("log")
plt.show()
# sns.boxplot(data, x="mp_name", y="duration_dbcbs", hue="cat")
# plt.show()
sns.boxplot(data, x="size", y="score_bl", hue="mp_name", showfliers=False)
# sns.lineplot(data, x="size", y="score_bl", hue="mp_name")
plt.axhline(y=0, color="red", linestyle="--", label="Baseline mean")
plt.show()

# breakpoint()
