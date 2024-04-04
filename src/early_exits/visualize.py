import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")


def visualize_ee_results(results, path):
    plt.figure()
    plt.clf()
    plt.cla()

    exit_costs = results["exit_costs"] / 10**9
    baseline_cost = results["baseline_cost"]
    per_ic_acc = results["per_ic_acc"]["tag"]
    per_th_acc = results["per_th_acc"]["tag"]
    per_th_exits = results["per_th_exit_cnt"]["tag"]
    per_th_exit_costs = (per_th_exits * np.expand_dims(exit_costs, 0)).sum(1)

    plot = sns.lineplot(x=per_th_exit_costs, y=per_th_acc)
    plot = sns.scatterplot(x=exit_costs, y=per_ic_acc)
    plot.set(xlabel="GFLOPs", ylabel="Accuracy")
    plot.get_figure().savefig(str(path))
