import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker

TBB_FILES = {
    "data/rpe-ffm_grads.csv": ["TBB", "FFM"],
    "data/rpe-linattn_grads.csv": ["TBB", "LinAttn"],
    "data/rpe-lru_grads.csv": ["TBB", "LRU"],
    "data/rpe-s5_grads.csv": ["TBB", "S5"],
}
SBB_FILES = {
    "data/segment-ffm-grad_grads.csv": ["SBB", "FFM"],
    "data/segment-linattn-grad_grads.csv": ["SBB", "LinAttn"],
    "data/segment-lru-grad_grads.csv": ["SBB", "LRU"],
    "data/segment-s5-grad_grads.csv": ["SBB", "S5"],
}
TBB_RFE_FILES = {
    "data/rfe-ffm_grads.csv": ["TBB", "FFM"],
    "data/rfe-linattn_grads.csv": ["TBB", "LinAttn"],
    "data/rfe-lru_grads.csv": ["TBB", "LRU"],
    "data/rfe-s5_grads.csv": ["TBB", "S5"],
}
SBB_RFE_FILES = {
    "data/segment-ffm-grad-rpf_grads.csv": ["SBB", "FFM"],
    "data/segment-linattn-grad-rpf_grads.csv": ["SBB", "LinAttn"],
    "data/segment-lru-grad-rpf_grads.csv": ["SBB", "LRU"],
    "data/segment-s5-grad-rpf_grads.csv": ["SBB", "S5"],
}
TBB_CPE_FILES = {
    "data/cpe-ffm_grads.csv": ["TBB", "FFM"],
    "data/cpe-linattn_grads.csv": ["TBB", "LinAttn"],
    "data/cpe-lru_grads.csv": ["TBB", "LRU"],
    "data/cpe-s5_grads.csv": ["TBB", "S5"],
}
SBB_CPE_FILES = {
    "data/segment-ffm-grad-cpe_grads.csv": ["SBB", "FFM"],
    "data/segment-linattn-grad-cpe_grads.csv": ["SBB", "LinAttn"],
    "data/segment-lru-grad-cpe_grads.csv": ["SBB", "LRU"],
    "data/segment-s5-grad-cpe_grads.csv": ["SBB", "S5"],
}
MAX_LEN = 102

for tbb, sbb in zip(TBB_FILES.items(), SBB_FILES.items()):
    #fig, (ax1, ax2) = plt.subplots(1,2)
    #fig, ax1 = plt.axes()

    sns.set()
    sns.set_style("darkgrid")
    sns.set_context('notebook')
    f, axes = plt.subplots(2, 2, height_ratios=[1, 0.75], figsize=(7,4))

    tbb_data = pd.read_csv(tbb[0])
    tbb_data = tbb_data.rename(columns={"Unnamed: 0": "Train Epoch"})
    tbb_data["Train Epoch"] = tbb_data["Train Epoch"].astype(int) * 10 # Convert eval epoch to training epoch
    tbb_data = tbb_data.set_index("Train Epoch").stack().reset_index()
    tbb_data = tbb_data.rename(columns={"level_1": "Observation Age", 0: "dQ(s_n, a_n)/(d o_i)"})
    tbb_data["Observation Age"] = tbb_data["Observation Age"].astype(int)

    tbb_pivot = tbb_data.copy().pivot(columns="Observation Age", index="Train Epoch", values="dQ(s_n, a_n)/(d o_i)")
    # normalize
    #tbb_pivot = tbb_pivot.div(tbb_pivot.sum(axis=1), axis=0)
    sns.heatmap(tbb_pivot, norm=LogNorm(), cbar=True, xticklabels=20, yticklabels=500, ax=axes[0, 0])
    axes[0,0].set_title(f"Sensitivity of TBB {tbb[1][1]} Q Values") 
    plt.tight_layout()

    sbb_data = pd.read_csv(sbb[0])
    sbb_data = sbb_data.rename(columns={"Unnamed: 0": "Train Epoch"})
    sbb_data["Train Epoch"] = sbb_data["Train Epoch"].astype(int) * 10 # Convert eval epoch to training epoch
    sbb_data = sbb_data.set_index("Train Epoch").stack().reset_index()
    sbb_data = sbb_data.rename(columns={"level_1": "Observation Age", 0: "dQ(s_n, a_n)/(d o_i)"})
    sbb_data["Observation Age"] = sbb_data["Observation Age"].astype(int)

    sbb_pivot = sbb_data.copy().pivot(columns="Observation Age", index="Train Epoch", values="dQ(s_n, a_n)/(d o_i)")
    # normalize
    #sbb_pivot = sbb_pivot.div(sbb_pivot.sum(axis=1), axis=0)
    sns.heatmap(sbb_pivot, norm=LogNorm(), cbar=True, xticklabels=20, yticklabels=500, ax=axes[0, 1])
    axes[0,1].set_title(f"Sensitivity of SBB {sbb[1][1]} L=10 Q Values") 
    plt.tight_layout()



    # cdf
    tbb_data = tbb_data.copy().pivot(index="Observation Age", columns="Train Epoch", values="dQ(s_n, a_n)/(d o_i)")
    tbb_cdf = tbb_data.cumsum(axis=0)[5000] / tbb_data.cumsum(axis=0)[5000].max()
    sns.lineplot(tbb_cdf.T, ax=axes[1,0])
    axes[1,0].set(ylim=(-0.05, 1.05))
    axes[1,0].axhline(y=tbb_cdf[-10], color="r", xmax=1 - (14/102))
    axes[1,0].plot(-10, tbb_cdf[-10], 'ro')  # Overlay red dot at x=-10, y=tbb_cdf[-10]
    axes[1,0].set_ylabel(r"Norm. Cumulative $\frac{\partial Q(s_n, a_n)}{\partial o_i}$")  
    plt.tight_layout()

    sbb_data = sbb_data.copy().pivot(index="Observation Age", columns="Train Epoch", values="dQ(s_n, a_n)/(d o_i)")
    sbb_cdf = sbb_data.cumsum(axis=0)[5000] / sbb_data.cumsum(axis=0)[5000].max()
    sns.lineplot(sbb_cdf.T, ax=axes[1,1])
    axes[1,1].set(ylim=(-0.05, 1.05))
    axes[1,1].axhline(y=sbb_cdf[-10], color="r", xmax=1 - (14/102))
    axes[1,1].plot(-10, sbb_cdf[-10], 'ro')  # Overlay red dot at x=-10, y=sbb_cdf[-10]
    axes[1,1].set_ylabel(r"Norm. Cumulative $\frac{\partial Q(s_n, a_n)}{\partial o_i}$")  
    plt.tight_layout()

    plt.savefig(f"plots/{tbb[1][0]}-{tbb[1][1]}-cdf.pdf")


MAX_LEN = 52
for tbb, sbb in zip(TBB_RFE_FILES.items(), SBB_RFE_FILES.items()):

    sns.set()
    sns.set_style("darkgrid")
    sns.set_context('notebook')
    f, axes = plt.subplots(2, 2, height_ratios=[1, 0.75], figsize=(7,4))

    tbb_data = pd.read_csv(tbb[0])
    tbb_data = tbb_data.rename(columns={"Unnamed: 0": "Train Epoch"})
    tbb_data["Train Epoch"] = tbb_data["Train Epoch"].astype(int) * 10 # Convert eval epoch to training epoch
    tbb_data = tbb_data.set_index("Train Epoch").stack().reset_index()
    tbb_data = tbb_data.rename(columns={"level_1": "Observation Age", 0: "dQ(s_n, a_n)/(d o_i)"})
    tbb_data["Observation Age"] = tbb_data["Observation Age"].astype(int)

    tbb_pivot = tbb_data.copy().pivot(columns="Observation Age", index="Train Epoch", values="dQ(s_n, a_n)/(d o_i)")
    # normalize
    #tbb_pivot = tbb_pivot.div(tbb_pivot.sum(axis=1), axis=0)
    sns.heatmap(tbb_pivot, norm=LogNorm(), cbar=True, xticklabels=20, yticklabels=500, ax=axes[0, 0])
    axes[0,0].set_title(f"Sensitivity of TBB {tbb[1][1]} Q Values") 
    plt.tight_layout()

    sbb_data = pd.read_csv(sbb[0])
    sbb_data = sbb_data.rename(columns={"Unnamed: 0": "Train Epoch"})
    sbb_data["Train Epoch"] = sbb_data["Train Epoch"].astype(int) * 10 # Convert eval epoch to training epoch
    sbb_data = sbb_data.set_index("Train Epoch").stack().reset_index()
    sbb_data = sbb_data.rename(columns={"level_1": "Observation Age", 0: "dQ(s_n, a_n)/(d o_i)"})
    sbb_data["Observation Age"] = sbb_data["Observation Age"].astype(int)

    sbb_pivot = sbb_data.copy().pivot(columns="Observation Age", index="Train Epoch", values="dQ(s_n, a_n)/(d o_i)")
    # normalize
    #sbb_pivot = sbb_pivot.div(sbb_pivot.sum(axis=1), axis=0)
    sns.heatmap(sbb_pivot, norm=LogNorm(), cbar=True, xticklabels=20, yticklabels=500, ax=axes[0, 1])
    axes[0,1].set_title(f"Sensitivity of SBB {sbb[1][1]} L=10 Q Values") 
    plt.tight_layout()



    # cdf
    tbb_data = tbb_data.copy().pivot(index="Observation Age", columns="Train Epoch", values="dQ(s_n, a_n)/(d o_i)")
    tbb_cdf = tbb_data.cumsum(axis=0)[5000] / tbb_data.cumsum(axis=0)[5000].max()
    sns.lineplot(tbb_cdf.T, ax=axes[1,0])
    axes[1,0].set(ylim=(-0.05, 1.05))
    axes[1,0].set_ylabel(r"Norm. Cumulative $\frac{\partial Q(s_n, a_n)}{\partial o_i}$")  
    plt.tight_layout()

    sbb_data = sbb_data.copy().pivot(index="Observation Age", columns="Train Epoch", values="dQ(s_n, a_n)/(d o_i)")
    sbb_cdf = sbb_data.cumsum(axis=0)[5000] / sbb_data.cumsum(axis=0)[5000].max()
    sns.lineplot(sbb_cdf.T, ax=axes[1,1])
    axes[1,1].set(ylim=(-0.05, 1.05))
    axes[1,1].axhline(y=sbb_cdf[-10], color="r", xmax=1 - (10/52))
    axes[1,1].plot(-10, sbb_cdf[-10], 'ro')  # Overlay red dot at x=-10, y=sbb_cdf[-10]
    axes[1,1].set_ylabel(r"Norm. Cumulative $\frac{\partial Q(s_n, a_n)}{\partial o_i}$")  
    plt.tight_layout()

    plt.savefig(f"plots/{tbb[1][0]}-{tbb[1][1]}-rpf-cdf.pdf")


MAX_LEN = 200
for tbb, sbb in zip(TBB_CPE_FILES.items(), SBB_CPE_FILES.items()):

    sns.set()
    sns.set_style("darkgrid")
    sns.set_context('notebook')
    f, axes = plt.subplots(2, 2, height_ratios=[1, 0.75], figsize=(7,4))

    tbb_data = pd.read_csv(tbb[0])
    tbb_data = tbb_data.rename(columns={"Unnamed: 0": "Train Epoch"})
    tbb_data["Train Epoch"] = tbb_data["Train Epoch"].astype(int) * 10 # Convert eval epoch to training epoch
    tbb_data = tbb_data.set_index("Train Epoch").stack().reset_index()
    tbb_data = tbb_data.rename(columns={"level_1": "Observation Age", 0: "dQ(s_n, a_n)/(d o_i)"})
    tbb_data["Observation Age"] = tbb_data["Observation Age"].astype(int)

    tbb_pivot = tbb_data.copy().pivot(columns="Observation Age", index="Train Epoch", values="dQ(s_n, a_n)/(d o_i)")
    # normalize
    #tbb_pivot = tbb_pivot.div(tbb_pivot.sum(axis=1), axis=0)
    sns.heatmap(tbb_pivot, norm=LogNorm(), cbar=True, xticklabels=50, yticklabels=500, ax=axes[0, 0])
    axes[0,0].set_title(f"Sensitivity of TBB {tbb[1][1]} Q Values") 
    plt.tight_layout()

    sbb_data = pd.read_csv(sbb[0])
    sbb_data = sbb_data.rename(columns={"Unnamed: 0": "Train Epoch"})
    sbb_data["Train Epoch"] = sbb_data["Train Epoch"].astype(int) * 10 # Convert eval epoch to training epoch
    sbb_data = sbb_data.set_index("Train Epoch").stack().reset_index()
    sbb_data = sbb_data.rename(columns={"level_1": "Observation Age", 0: "dQ(s_n, a_n)/(d o_i)"})
    sbb_data["Observation Age"] = sbb_data["Observation Age"].astype(int)

    sbb_pivot = sbb_data.copy().pivot(columns="Observation Age", index="Train Epoch", values="dQ(s_n, a_n)/(d o_i)")
    # normalize
    #sbb_pivot = sbb_pivot.div(sbb_pivot.sum(axis=1), axis=0)
    sns.heatmap(sbb_pivot, norm=LogNorm(), cbar=True, xticklabels=50, yticklabels=500, ax=axes[0, 1])
    axes[0,1].set_title(f"Sensitivity of SBB {sbb[1][1]} L=10 Q Values") 
    plt.tight_layout()



    # cdf
    tbb_data = tbb_data.copy().pivot(index="Observation Age", columns="Train Epoch", values="dQ(s_n, a_n)/(d o_i)")
    tbb_cdf = tbb_data.cumsum(axis=0)[10_000] / tbb_data.cumsum(axis=0)[10_000].max()
    sns.lineplot(tbb_cdf.T, ax=axes[1,0])
    axes[1,0].set(ylim=(-0.05, 1.05))
    #axes[1,0].axhline(y=tbb_cdf[-10], color="r", xmax=1 - (15 / 200))
    #axes[1,0].plot(-10, tbb_cdf[-10], 'ro')  # Overlay red dot at x=-10, y=sbb_cdf[-10]
    axes[1,0].set_ylabel(r"Norm. Cumulative $\frac{\partial Q(s_n, a_n)}{\partial o_i}$")  
    
    plt.tight_layout()

    sbb_data = sbb_data.copy().pivot(index="Observation Age", columns="Train Epoch", values="dQ(s_n, a_n)/(d o_i)")
    sbb_cdf = sbb_data.cumsum(axis=0)[10_000] / sbb_data.cumsum(axis=0)[10_000].max()
    sns.lineplot(sbb_cdf.T, ax=axes[1,1])
    axes[1,1].set(ylim=(-0.05, 1.05))
    axes[1,1].axhline(y=sbb_cdf[-10], color="r", xmax=1 - (15 / 200))
    axes[1,1].plot(-10, sbb_cdf[-10], 'ro')  # Overlay red dot at x=-10, y=sbb_cdf[-10]
    axes[1,1].set_ylabel(r"Norm. Cumulative $\frac{\partial Q(s_n, a_n)}{\partial o_i}$")  
    plt.tight_layout()

    plt.savefig(f"plots/{tbb[1][0]}-{tbb[1][1]}-cpe-cdf.pdf")