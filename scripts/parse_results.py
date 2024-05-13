from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

sns.set_style("whitegrid")


def best_th_acc(costs, accs, cost_th):
    cost_acc = [(c, a) for c, a in zip(costs, accs)]
    cost_acc_below = [(c, a) for c, a in cost_acc if c <= cost_th]
    if len(cost_acc_below) == 0:
        return 0
    best_acc = max(cost_acc_below, key=lambda x: x[1])[1]
    return best_acc


def aggregate_ee_data(th_data, ic_data):
    output_th_data = []
    output_ic_data = []

    run_names = list(set(list(th_data["run_name"].unique())))
    for run_name in run_names:
        run_th_data = th_data[th_data["run_name"] == run_name].sort_values(
            by=["seed", "cost"]
        )
        seeds = list(run_th_data["seed"].unique())

        costs_th = []
        accs_th = []
        for seed in seeds:
            cost = run_th_data[run_th_data["seed"] == seed]["cost"].values
            acc = run_th_data[run_th_data["seed"] == seed]["acc"].values
            costs_th.append(cost)
            accs_th.append(acc)
        costs_th = np.stack(costs_th, axis=0)
        accs_th = np.stack(accs_th, axis=0)
        cost_mean = costs_th.mean(axis=0)
        cost_std = costs_th.std(axis=0)
        acc_mean = accs_th.mean(axis=0)
        acc_std = accs_th.std(axis=0)

        base_run_th = deepcopy(run_th_data[run_th_data["seed"] == seeds[0]])
        base_run_th["cost_avg"] = cost_mean
        base_run_th["cost_std"] = cost_std
        base_run_th["acc_avg"] = acc_mean
        base_run_th["acc_std"] = acc_std
        output_th_data.append(base_run_th)

        run_ic_data = ic_data[ic_data["run_name"] == run_name]
        if len(run_ic_data) == 0:
            continue
        costs_ic = []
        accs_ic = []
        for seed in seeds:
            cost = run_ic_data[run_ic_data["seed"] == seed]["cost"].values
            acc = run_ic_data[run_ic_data["seed"] == seed]["acc"].values
            costs_ic.append(cost)
            accs_ic.append(acc)
        costs_ic = np.stack(costs_ic, axis=0)
        accs_ic = np.stack(accs_ic, axis=0)

        cost_mean = costs_ic.mean(axis=0)
        cost_std = costs_ic.std(axis=0)
        acc_mean = accs_ic.mean(axis=0)
        acc_std = accs_ic.std(axis=0)

        base_run_ic = deepcopy(run_ic_data[run_ic_data["seed"] == seeds[0]])
        base_run_ic["cost_avg"] = cost_mean
        base_run_ic["cost_std"] = cost_std
        base_run_ic["acc_avg"] = acc_mean
        base_run_ic["acc_std"] = acc_std
        output_ic_data.append(base_run_ic)

    ic_data = pd.concat(output_ic_data)
    th_data = pd.concat(output_th_data)
    return th_data, ic_data


def plot_compare(th_data, ic_data, output_path):
    plt.cla()
    plt.clf()
    plt.figure()

    th_data, ic_data = aggregate_ee_data(th_data, ic_data)
    th_data = th_data.sort_values(by=["ee"])

    if len(th_data) > 0 and len(ic_data) > 0:
        colors = sns.color_palette("tab10")
        keys = list(
            set(list(th_data["run_name"].unique()) + list(ic_data["run_name"].unique()))
        )
        palette = {k: colors[i] for i, k in enumerate(keys)}
    else:
        palette = None
    if len(th_data) > 0:
        plot = sns.lineplot(
            x="cost_avg", y="acc_avg", hue="run_name", data=th_data, palette=palette
        )
        for run_name in th_data["run_name"].unique():
            run_data = th_data[th_data["run_name"] == run_name]
            run_data = run_data.sort_values(by=["cost_avg"])
            avg_cost = run_data["cost_avg"]
            y_1 = run_data["acc_avg"] - run_data["acc_std"]
            y_2 = run_data["acc_avg"] + run_data["acc_std"]
            if any(y_1.values != y_2.values):
                plt.fill_between(
                    avg_cost,
                    y_1,
                    y_2,
                    alpha=0.3,
                    color=palette[run_name],
                    interpolate=True,
                )

    if len(ic_data) > 0:
        plot = sns.scatterplot(
            x="cost_avg",
            y="acc_avg",
            hue="run_name",
            data=ic_data,
            legend=False,
            palette=palette,
        )
        for run_name in ic_data["run_name"].unique():
            run_data = ic_data[ic_data["run_name"] == run_name]
            if len(run_data) == 0:
                continue
            run_data = run_data.sort_values(by=["cost_avg"])
            y_1 = run_data["acc_avg"] - run_data["acc_std"]
            y_2 = run_data["acc_avg"] + run_data["acc_std"]
            if any(y_1.values != y_2.values):
                plt.errorbar(
                    run_data["cost_avg"],
                    run_data["acc_avg"],
                    yerr=run_data["acc_std"],
                    fmt="none",
                    color=palette[run_name],
                )
    plot.get_figure().savefig(str(output_path))


if __name__ == "__main__":
    root_name = "results"
    output_dir = Path("parsed_results") / "semi_final_batch"
    output_dir.mkdir(parents=True, exist_ok=True)

    root_dir = Path(root_name)
    result_dirs = list(root_dir.rglob("*/results"))
    assert len(result_dirs) > 0, f"Didn't find any results in {root_dir}"

    ic_plot_data = []
    th_plot_data = []
    output_data = []

    result_dirs = sorted(list(result_dirs))
    for r in tqdm(result_dirs):
        results = list(r.glob("*"))
        if len(results) == 0:
            print(f"Didn't find any results in {r}")
            continue

        ee_filenames = list(r.glob("ee_eval.npy"))
        final_results_filenames = list(r.glob("avg_accs_tag*"))
        if not (len(ee_filenames) == 1 or len(final_results_filenames) == 1):
            print(f"Didn't find any results for {r}")
            continue

        assert len(ee_filenames) == 1 or len(final_results_filenames) == 1

        run_name = str(r.parent.parent).replace(root_name, "")
        dataset = run_name.strip("/").split("/")[0]
        run_name = "_".join(run_name.strip("/").split("/")[1:])
        seed = int(run_name.split("seed")[-1])
        run_name = run_name.replace(f"_seed{seed}", "")
        method = run_name.split("_")[0]
        if "ex" in run_name:
            method += "_ex" + run_name.split("ex")[1].split("_")[0]
        if len(ee_filenames) == 1:
            ee_filename = ee_filenames[0]
            ee_data = np.load(ee_filename, allow_pickle=True).item()["avg"]
            exit_costs = ee_data["exit_costs"]
            baseline_cost = ee_data["baseline_cost"]
            per_ic_acc = ee_data["per_ic_acc"]["tag"] * 100
            per_th_acc = ee_data["per_th_acc"]["tag"] * 100
            per_th_exit_cnt = ee_data["per_th_exit_cnt"]["tag"]
            per_th_cost = (exit_costs[None, :] * per_th_exit_cnt).sum(1) / baseline_cost
            ic_accs = {f"ic_{i}": acc for i, acc in enumerate(per_ic_acc.tolist())}
            th_accs = {
                "th_025": best_th_acc(per_th_cost, per_th_acc, 0.25),
                "th_050": best_th_acc(per_th_cost, per_th_acc, 0.5),
                "th_075": best_th_acc(per_th_cost, per_th_acc, 0.75),
                "th_100": best_th_acc(per_th_cost, per_th_acc, 1.0),
                "th_none": max(per_ic_acc),
            }
            output = {
                "dataset": dataset,
                "method": method,
                "run_name": run_name,
                "seed": seed,
            }
            output.update(ic_accs)
            output.update(th_accs)
            output_data.append(output)

            ic_costs = exit_costs / baseline_cost
            for ic_acc, ic_cost in zip(per_ic_acc.tolist(), ic_costs.tolist()):
                ic_plot_data.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "run_name": run_name,
                        "cost": ic_cost,
                        "acc": ic_acc,
                        "ee": True,
                        "seed": seed,
                    }
                )
            for th_acc, th_cost in zip(per_th_acc.tolist(), per_th_cost.tolist()):
                th_plot_data.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "run_name": run_name,
                        "cost": th_cost,
                        "acc": th_acc,
                        "ee": True,
                        "seed": seed,
                    }
                )
        else:
            final_results_filename = final_results_filenames[0]
            with final_results_filename.open("r") as f:
                data = f.read().strip()
            final_val = 100 * float(data.split("\t")[-1])
            output = {
                "dataset": dataset,
                "method": method,
                "run_name": run_name,
                "no_ee_acc": final_val,
                "seed": seed,
            }
            output_data.append(output)

            th_plot_data.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "run_name": run_name,
                    "cost": 0.2,
                    "acc": final_val,
                    "ee": False,
                    "seed": seed,
                }
            )
            th_plot_data.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "run_name": run_name,
                    "cost": 1,
                    "acc": final_val,
                    "ee": False,
                    "seed": seed,
                }
            )

    data = pd.DataFrame(output_data)
    data = data.sort_values(by=["dataset", "method", "run_name"])
    # Data to csv formatting floats to 3 points
    data.to_csv(output_dir / "data.csv", index=False, float_format="%.3f")

    th_plot_data = pd.DataFrame(th_plot_data)
    ic_plot_data = pd.DataFrame(ic_plot_data)

    datasets = data["dataset"].unique()
    method_names = data["method"].unique()
    dataset_method_combinations = [(d, m) for d in datasets for m in method_names]
    for dataset_name, method_name in tqdm(
        dataset_method_combinations, "Plotting early exit scores..."
    ):
        dataset_output_dir = output_dir / dataset_name
        dataset_output_dir.mkdir(exist_ok=True, parents=True)

        dataset_method_th_data = th_plot_data[
            (th_plot_data["dataset"] == dataset_name)
            & (th_plot_data["method"] == method_name)
        ]
        dataset_method_ic_data = ic_plot_data[
            (ic_plot_data["dataset"] == dataset_name)
            & (ic_plot_data["method"] == method_name)
        ]

        if len(dataset_method_ic_data) == 0 and len(dataset_method_th_data) == 0:
            print(f"Skipping {method_name} for {dataset_name} due to no data")
            continue

        plot_compare(
            dataset_method_th_data,
            dataset_method_ic_data,
            dataset_output_dir / (method_name + ".png"),
        )
