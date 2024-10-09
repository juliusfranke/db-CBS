import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_parquet("../results/bench.parquet")
breakpoint()
# mps = data[data.cat != "Baseline"]
# baseline_mean = (
#     data[data.cat == "Baseline"][["size", "success", "cost", "duration_dbcbs"]]
#     .groupby("size")
#     .mean()
# ), hue="cat"
sns.barplot(data, x="size", y="success", hue="mp_name")
plt.show()
sns.boxplot(data, x="size", y="duration_dbcbs", hue="mp_name")
plt.show()
sns.boxplot(data, x="size", y="cost", hue="mp_name")
plt.show()
# sns.boxplot(data, x="mp_name", y="duration_dbcbs", hue="cat")
# plt.show()

breakpoint()
