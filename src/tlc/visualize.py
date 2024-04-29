from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch import Tensor


def plot_heatmap(data: Tensor, output_path: Path):
    plt.cla()
    plt.clf()
    plt.figure()
    plot = sns.heatmap(data.detach().cpu().numpy(), annot=True, fmt=".2f", cmap="Blues")
    plot.get_figure().savefig(str(output_path))


def plot_loss(losses: List[float], output_path: Path):
    plt.cla()
    plt.clf()
    plt.figure()
    plot = sns.lineplot(y=losses, x=list(range(len(losses))))
    plot.set_yscale("log")
    plot.get_figure().savefig(str(output_path))


def plot_cost_vs_acc(df: pd.DataFrame, output_path: Path):
    plt.clf()
    plt.cla()
    plt.figure()
    plot = sns.lineplot(data=df, x="cost", y="acc", hue="method")
    plot.set_title("Cost vs Accuracy")
    plot.get_figure().savefig(str(output_path))
