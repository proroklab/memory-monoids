import numpy as np
import plotlib as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


runs, summaries = pl.build_projects(
    {"rdqn_paper": [100, 1000]}, 
    workdir="data/wandb", 
    clean=False, 
    multiprocess=True, 
    x_key="collect/train_epoch",
    metric_keys={
        "eval/collect/best_reward": "Return",
        "collect/train_epoch": "Train Epoch",
        "train/time_total": "Train Time",
        "train/time_this_epoch": "Train Time Per Epoch",
    },
    recategorize_keys=["Env", "Model"],
)

runs['Method'] = runs.apply(lambda x: x['Batch Mode'] + ' L=' + str(round(x['Segment Length'])) if x['Batch Mode'] == 'SBB' else x['Batch Mode'], axis=1)
# Delete unnecessary columns
runs = runs[["Train Epoch", "Return", "Method", "Env", "Model", "Train Time", "Train Time Per Epoch"]]

# If you ever have the weird problem where the plots are are all jagged and with no sd/ci
# MAKE SURE THE X VALUES ARE THE SAME FOR EACH RUN
# If the x values vary, multiple runs will be treated as one
sns.set()
sns.set_style("darkgrid")
sns.set_context('notebook')
ax = sns.relplot(
    runs, x="Train Epoch", y="Return", 
    col="Model", row="Env", hue="Method", kind="line",
    height=1.85,
    facet_kws={'sharey': False, 'sharex': False, "margin_titles": True}, errorbar="ci",
    hue_order=['TBB', 'SBB L=10', 'SBB L=20', 'SBB L=50', 'SBB L=100'],
    aspect=1.25,
)
ax.set_titles(col_template="{col_name}", row_template="{row_name}")
plt.savefig("plots/return_all.pdf")

rpf_runs = runs[runs['Env'] == 'Repeat First']
rpf_runs.groupby(['Method'])['Train Time Per Epoch'].mean()
rpf_runs.groupby(['Method'])['Train Time'].mean()



s5 = runs[
    (runs['Env'] == "Count Recall") & 
    (runs['Model'] == "S5")
]
linattn = runs[
    (runs['Env'] == "Pos. Cartpole") & 
    (runs['Model'] == "LinAttn")
]
lru = runs[
    (runs['Env'] == "Repeat First") & 
    (runs['Model'] == "LRU")
]
ffm = runs[
    (runs['Env'] == "Repeat Previous") & 
    (runs['Model'] == "FFM")
]
subruns = pd.concat([s5, linattn, lru, ffm])
subruns["Model/Env"] = subruns["Model"].astype(str) + "/" + subruns["Env"].astype(str)


ax = sns.relplot(
    subruns, x="Train Epoch", y="Return", 
    col="Model/Env", hue="Method", kind="line", col_wrap=2,
    height=1.85,
    facet_kws={'sharey': False, 'sharex': False, "margin_titles": True}, errorbar="ci",
    hue_order=['TBB', 'SBB L=10', 'SBB L=20', 'SBB L=50', 'SBB L=100'],
    aspect=1.25,
)
ax.set_titles(col_template="{col_name}", row_template="{row_name}")
sns.move_legend(ax, "center left", bbox_to_anchor=(0.8, 0.5))
plt.savefig("plots/return_small_2x2.pdf", bbox_extra_artists=(ax.legend,), bbox_inches='tight')