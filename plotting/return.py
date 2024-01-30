import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

FILE = "data/return_perf.csv"

plt.figure(figsize=(5, 3))
sns.set()
sns.set_style("darkgrid")
sns.set_context('notebook')
data = pd.read_csv(FILE)
data = data.rename(columns={"Mode": "Method"})
data = data.replace({"Parallel": "Monoid"})
ax = sns.lineplot(data=data, x="Batch Size", y="Time (ms)", hue="Method", style="Max Ep. Length")
ax.set(yscale='log', xscale='log')
ax.set_title("Return Runtime")
#plt.legend(loc='center right', bbox_to_anchor=(1, 0.6))
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig("plots/return-runtime.pdf")
#plt.show()