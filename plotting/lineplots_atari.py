import numpy as np
import plotlib as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


runs, summaries = pl.build_projects(
    {"monoids_atari_final": [100, 1000]}, 
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
lru = runs[
    (runs['Model'] == "LRU")
]
ffm = runs[
    (runs['Model'] == "FFM")
]
subruns = pd.concat([lru, ffm])
subruns["Model/Env"] = subruns["Model"].astype(str) + "-" + subruns["Env"].astype(str)

# If you ever have the weird problem where the plots are are all jagged and with no sd/ci
# MAKE SURE THE X VALUES ARE THE SAME FOR EACH RUN
# If the x values vary, multiple runs will be treated as one
sns.set()
sns.set_style("darkgrid")
sns.set_context('notebook')
ax = sns.relplot(
    subruns, x="Train Epoch", y="Return", 
    col="Model/Env", hue="Method", kind="line", col_wrap=2,
    height=4,
    facet_kws={'sharey': False, 'sharex': False, "margin_titles": True}, errorbar="ci",
    hue_order=['TBB', 'SBB L=80'],
    aspect=1.5,
)
ax.axes.flatten()[0].hlines(y=1020, xmin=0, xmax=10_000, color='r', ls='--')
ax.set_titles(col_template="{col_name}", row_template="{row_name}")
plt.savefig("plots/return_atari.pdf")