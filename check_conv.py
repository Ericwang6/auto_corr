import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob


def create_multi_barhs(ax, labels, datas, tick_step=1, group_gap=0.2, bar_gap=0, xerrs=None, legends=None):
    """
    https://blog.csdn.net/mighty13/article/details/113873617
    """
    ticks = np.arange(len(labels)) * tick_step
    group_num = len(datas)
    if legends is None:
        legends = [str(x) for x in range(group_num)]
    group_width = tick_step - group_gap
    bar_span = group_width / group_num
    bar_width = bar_span - bar_gap
    baseline_x = ticks - (group_width - bar_span) / 2
    for index, y in enumerate(datas):
        if xerrs is None:
            ax.barh(baseline_x + index*bar_span, y, bar_width, labels, label=legends[index])
        else:
            ax.barh(baseline_x + index*bar_span, y, bar_width, xerr=xerrs[index], label=legends[index])
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(ticks.min() - bar_span * group_num, ticks.max() + bar_span * group_num)
    ax.legend()


csvs = list(glob.glob("*_corr.csv"))
times = [int(csv.split("_")[0]) for csv in csvs]
times.sort()
datas = [pd.read_csv(f"{tt}_corr.csv", index_col=0) for tt in times]
labels = list(datas[0].T.columns)
legends = [f"{tt/1000:.1f}ns" for tt in times]

solvated_diff = np.array([data.T.loc["solvated_diff", :] for data in datas])
solvated_std = np.array([data.T.loc["solvated_std", :] for data in datas])
complex_diff = np.array([data.T.loc["complex_diff", :] for data in datas])
complex_std = np.array([data.T.loc["complex_std", :] for data in datas])

# solvated 
plt.rcParams['font.size'] = 14
fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(10, 30))
create_multi_barhs(ax, labels, solvated_diff, xerrs=solvated_std, legends=legends)
ax.set_xlabel("Energy (kcal/mol)")
ax.set_ylabel("Ligands")
ax.set_title("Solvated Free Energy Difference")
fig.savefig("solvated.png", dpi=300)

# complex
plt.rcParams['font.size'] = 14
fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(10, 30))
create_multi_barhs(ax, labels, complex_diff, xerrs=complex_std, legends=legends)
ax.set_xlabel("Energy (kcal/mol)")
ax.set_ylabel("Ligands")
ax.set_title("Complex Free Energy Difference")
fig.savefig("complex.png", dpi=300)